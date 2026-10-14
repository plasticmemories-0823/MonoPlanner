"""在按既有规则绑定 Docker 任务前，准备可复用的项目环境。

只有环境准备流程可以下载依赖；不在本机安装，不以可写方式挂载用户源码，
也不向任务开放网络连接。
"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid
import weakref

from agent.execution.docker import (
    docker_executable,
    proxy_environment,
    PROXY_VARIABLES,
    dependency_cache_usable,
    LOG_OPTIONS,
)
from agent.execution.managed_plan import (
    PreparationError,
    REVISION,
    inspect_project,
    _digest,
    _link,
)
from agent.storage.preferences import atomic_write, state_root

MAX_LOG = 16 * 1024 * 1024
SETUP_TIMEOUT = 1800
PROXY_LIFETIME = SETUP_TIMEOUT + 300
IMAGE_ID = re.compile(r'sha256:[0-9a-f]{64}')
BUILDKIT_IMAGE = 'moby/buildkit:v0.34.0'
_LOCKS = weakref.WeakValueDictionary()
_LOCKS_GUARD = threading.Lock()
_DOWNLOAD_RELAY = ContextVar('monoplanner_preparation_download_relay', default=None)
_CONTAINER_INSPECT = (
    '[{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},'
    '"State":{{json .State}},"Config":{"Labels":{{json .Config.Labels}}},'
    '"HostConfig":{"Memory":{{json .HostConfig.Memory}},'
    '"LogConfig":{{json .HostConfig.LogConfig}}}}]'
)


@contextmanager
def _default_downloads():
    from agent.execution.download_relay import DownloadRelay

    with DownloadRelay() as relay:
        token = _DOWNLOAD_RELAY.set(relay)
        try:
            yield relay
        finally:
            _DOWNLOAD_RELAY.reset(token)


@contextmanager
def _preparation_log(path):
    try:
        with path.open('wb') as output:
            yield output
    finally:
        relay = _DOWNLOAD_RELAY.get()
        if relay is not None and path.is_file():
            relay.redact_log(path)


# ============================================================
# 取消信号与准备记录
# ============================================================
class PreparationCancelled(PreparationError):
    pass


def _check_cancel(cancelled):
    from agent.storage.resources import owner_cancelled

    if owner_cancelled() or (cancelled and cancelled()):
        raise PreparationCancelled('环境准备已取消，尚未开始执行任务。')


def _save(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8'))


def _load(path):
    if not path.exists():
        return None
    if _link(path):
        raise PreparationError('环境记录不能是链接。')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError('环境记录不完整，请保留现场后重新准备。') from exc


# ============================================================
# 项目环境的独占准备锁
# ============================================================
@contextmanager
def _lock(path, cancelled=None):
    """进程退出后由操作系统释放锁，不删除所谓过期锁文件。"""
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    while not lock.acquire(timeout=0.1):
        _check_cancel(cancelled)
    stream = None
    acquired = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and _link(path):
            raise PreparationError('环境锁不能是链接。')
        stream = path.open('a+b')
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        while True:
            _check_cancel(cancelled)
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.1)
        yield
    finally:
        if stream:
            if acquired:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        lock.release()


# ============================================================
# 可复用环境的查询与准备
# ============================================================
class ManagedEnvironments:
    def __init__(self, root=None):
        from agent.storage.resources import current_root

        self.root = current_root(root) / 'managed-environments'

    def _record(self, repo):
        return self.root / 'projects' / (_digest(str(Path(repo).resolve())) + '.json')

    def _register_binding(self, repo, binding, work):
        from agent.storage.resources import resource_catalog

        catalog = resource_catalog(self.root.parent)
        owner = 'project:' + hashlib.sha256(str(Path(repo).resolve()).encode()).hexdigest()
        image = binding['image']
        inspected = self._image(image, work) or {}
        metadata = {'repo': str(Path(repo).resolve()),
                    'fingerprint': binding.get('managed', {}).get('fingerprint')}
        identity = catalog.register('image:' + image, 'image', image,
                                    size_bytes=inspected.get('Size'),
                                    metadata=metadata, owner=owner)
        base = binding.get('managed', {}).get('base_image')
        if base:
            catalog.link('environment:' + image, 'image:' + base)
        # 新环境提交后才释放同项目旧版本；旧任务自己的引用仍然有效。
        with catalog._db() as db:
            db.execute('DELETE FROM resource_refs WHERE owner=? AND resource<>?', (owner, identity))
        usage = catalog.usage()
        if usage['idle_bytes'] > usage['cache_limit_bytes'] or usage['idle_unknown_count']:
            from agent.storage.resources import delete_owned_docker_resource

            catalog.cleanup(docker=delete_owned_docker_resource)
        return binding

    def _stop_owned_container(self, name, metadata, owner):
        from agent.storage.resources import stop_recorded_container

        # 原件同时留在内存；磁盘已满时，仍能按原 ID、标签和镜像终止安装。
        return stop_recorded_container(name, metadata, owner, self.root.parent)

    def _finish_resources(self, owner, identities, work, *, completed):
        from agent.storage.resources import resource_catalog, delete_owned_docker_resource

        def query(arguments):
            return self._run(arguments, check=False,
                log=work / ('cleanup-' + _digest(arguments)[:16] + '.log'))
        result = resource_catalog(self.root.parent).finish_resources(
            owner, identities, completed=completed, docker=lambda kind, identity, metadata:
            delete_owned_docker_resource(kind, identity, metadata, query=query, stop_running=True))
        if result.get('errors'):
            raise PreparationError('环境资源收尾未完成，现场已保留。')
        return result

    def _space_guard(self, *, initial=False):
        from agent.storage.resources import preflight_space, resource_catalog
        from agent.storage.disk_space import require_docker_space

        reserve = resource_catalog(self.root.parent).usage()['reserve_bytes']
        preflight_space(self.root, 0, reserve_bytes=reserve)
        self._docker_storage = require_docker_space(
            reserve, previous=None if initial else getattr(self, '_docker_storage', None),
            verify_context=initial or not hasattr(self, '_docker_storage'),
        )

    def _build_base(self, build, tag, work, proxy_arguments, cancelled):
        with _default_downloads():
            return self._build_base_with_downloads(build, tag, work, proxy_arguments, cancelled)

    def _build_base_with_downloads(self, build, tag, work, proxy_arguments, cancelled):
        """具名构建容器带固定镜像、日志限额和精确身份，不改用户默认 builder。"""
        from agent.storage.resources import (
            current_owner, register_active_container, unregister_active_container,
            resource_catalog, owner_cancelled, ownership_labels,
        )

        name = 'monoplanner-build-' + uuid.uuid4().hex
        node = name + '0'
        container, volume = 'buildx_buildkit_' + node, 'buildx_buildkit_' + node + '_state'
        owner = current_owner('prepare:' + name)
        labels = ownership_labels(owner, self.root.parent)
        label_args = [part for key, value in labels.items() for part in ('--label', key + '=' + value)]
        catalog = resource_catalog(self.root.parent)
        identity, volume_identity = 'container:' + container, 'volume:' + volume
        actual_id = None
        image = self._image(BUILDKIT_IMAGE, work, cancelled)
        downloaded = image is None
        if image is None:
            from agent.execution.download_relay import docker_proxy_modes

            modes = docker_proxy_modes()
            _save(work / 'builder-network.json', {
                'bootstrap': 'docker-desktop-daemon', 'proxy_modes': modes,
                'build_and_install': 'current-system-network',
            })
            try:
                self._run(['pull', BUILDKIT_IMAGE], log=work / 'builder-pull.log',
                          cancelled=cancelled, timeout=SETUP_TIMEOUT)
            except PreparationCancelled:
                raise
            except PreparationError as exc:
                raise PreparationError(
                    '首次下载构建工具失败：这一步由 Docker Desktop 下载，使用它已有的代理设置'
                    f'（桌面：{modes["desktop"]}；容器：{modes["containers"]}）。'
                    '系统代理开关无法覆盖 Docker 的手动代理。请在 Docker Desktop 的代理设置中'
                    '核对系统代理模式或现有手动配置；本次不会修改全局设置或自动重试。'
                    '详细错误已保存到 builder-pull.log。'
                ) from exc
            image = self._image(BUILDKIT_IMAGE, work, cancelled)
        if image is None:
            raise PreparationError('无法核验固定版本 BuildKit 镜像。')
        with catalog._db() as db:
            existing = db.execute('SELECT owned FROM resources WHERE id=?',
                                  ('image:' + image['Id'],)).fetchone()
        catalog.register('image:' + image['Id'], 'image', image['Id'],
                         owned=bool(existing['owned']) if existing else downloaded,
                         size_bytes=image.get('Size'), metadata={'downloaded': BUILDKIT_IMAGE},
                         owner=owner)
        metadata = {'builder': name, 'builder_node': node, 'builder_volume': volume,
                    'container_id': None, 'image': image['Id'], 'labels': labels}
        catalog.register(identity, 'container', container, owner=owner, metadata=metadata)
        control = register_active_container(
            owner, container, lambda: self._stop_owned_container(container, metadata, owner),
            root=self.root.parent,
        )
        complete = False
        def receipt(status):
            _save(work / 'builder.json', {'name': name, 'node': node, 'container': container,
                'owner': owner, 'status': status, 'container_id': actual_id,
                'volume': volume, 'image': image['Id'], 'labels': labels})
        receipt('starting')
        try:
            self._run(['volume', 'create', *label_args, '--label',
                       'com.monoplanner.builder=' + name, volume],
                      log=work / 'builder-volume.log', cancelled=cancelled)
            catalog.register(volume_identity, 'volume', volume, owner=owner,
                             metadata={'builder': name, 'labels': labels})
            created = self._run([
                'create', '--name', container, '--pull', 'never', '--init',
                '--privileged', '--memory', '2g', '--memory-swap', '2g', '--cpus', '2',
                '--pids-limit', '256', '--restart', 'no', *LOG_OPTIONS, *label_args,
                '--mount', 'type=volume,source=' + volume + ',target=/var/lib/buildkit',
                *proxy_environment(_DOWNLOAD_RELAY.get().url(container=True)), image['Id'],
            ], log=work / 'builder-container.log', cancelled=cancelled)
            actual_id = created.stdout.strip()
            if not re.fullmatch(r'[0-9a-f]{64}', actual_id):
                raise PreparationError('构建容器身份未确认，已保留现场。')
            metadata['container_id'] = actual_id
            catalog.register(identity, 'container', container, owner=owner, metadata=metadata)
            receipt('created')
            config = work / 'buildkitd.toml'
            config.write_text(
                '[worker.oci]\nmax-parallelism = 1\ngc = true\nreservedSpace = "512MB"\n'
                'maxUsedSpace = "20GB"\nminFreeSpace = "5GB"\n', encoding='utf-8')
            self._run(['cp', str(config), container + ':/etc/buildkit/buildkitd.toml'],
                      log=work / 'builder-config.log', cancelled=cancelled)
            self._run([
                'buildx', 'create', '--name', name, '--node', node, '--driver', 'docker-container',
                '--driver-opt', 'image=' + BUILDKIT_IMAGE,
                '--driver-opt', 'memory=2g', '--driver-opt', 'cpu-period=100000',
                '--driver-opt', 'cpu-quota=200000', '--driver-opt', 'restart-policy=no',
            ], log=work / 'builder-create.log', cancelled=cancelled)
            self._run(['buildx', 'inspect', '--bootstrap', name],
                      log=work / 'builder-bootstrap.log', cancelled=cancelled)
            inspected = self._run(['container', 'inspect', actual_id],
                                 log=work / 'builder-inspect.log', cancelled=cancelled)
            actual = json.loads(inspected.stdout)[0]
            host = actual.get('HostConfig', {})
            if (actual.get('Id') != actual_id or actual.get('Image') != image['Id']
                    or any(actual.get('Config', {}).get('Labels', {}).get(key) != value
                           for key, value in labels.items())
                    or host.get('LogConfig') != {
                        'Type': 'local', 'Config': {'max-size': '10m', 'max-file': '3'}}
                    or host.get('Memory') != 2 * 1024**3):
                raise PreparationError('构建容器身份或资源限制与记录不符。')
            receipt('building')
            self._run([
                'buildx', 'build', '--builder', name, '--load', '--network', 'none',
                *proxy_arguments, '--tag', tag, str(build),
            ], log=work / 'base-build.log', cancelled=cancelled, timeout=SETUP_TIMEOUT)
            _check_cancel(cancelled)
            complete = True
        finally:
            try:
                if complete and not owner_cancelled(owner, self.root.parent):
                    self._finish_resources(owner, [identity, volume_identity], work, completed=True)
                else:
                    try:
                        stopped = self._stop_owned_container(container, metadata, owner)
                        catalog.record_container_stop(owner, container, stopped or {})
                    except (PreparationError, OSError, ValueError, subprocess.SubprocessError):
                        pass
                receipt('completed' if complete else 'interrupted')
            finally:
                unregister_active_container(control)


    def status(self, repo):
        plan = inspect_project(repo)
        record = _load(self._record(plan.root))
        current = record and record.get('fingerprint') == plan.fingerprint
        if current:
            from agent.storage.resources import resource_catalog

            with resource_catalog(self.root.parent)._db() as db:
                resource = db.execute('SELECT status FROM resources WHERE id=?',
                                      ('image:' + record['binding']['image'],)).fetchone()
            if resource and resource['status'] != 'ready':
                current = False
        return {
            **plan.public(),
            'state': 'ready' if current else 'needs_preparation',
            **(record.get('binding', {}) if current else {}),
            'cached': bool(current),
            'materials': record.get('materials') if current else None,
            'image': record.get('binding', {}).get('image') if current else None,
            'message': (
                '已保存环境，将在执行前核验并复用。'
                if current
                else '首次工作时将自动准备项目环境。'
            ),
        }

    def verify(self, binding):
        """用户明确刷新时核验保存环境；不创建容器、下载镜像或重建依赖。"""
        deadline = time.monotonic() + 5
        image = binding.get('image', '')
        verified = []
        if not isinstance(image, str) or not IMAGE_ID.fullmatch(image):
            return {'state': 'unknown', 'message': '保存的环境身份不完整，请核对配置。'}

        def inspect(kind, identity):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('environment verification deadline')
            result = subprocess.run(
                [docker_executable(), kind, 'inspect', identity],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=remaining,
                **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}),
            )
            if result.returncode:
                error = (result.stdout + result.stderr).lower()
                if 'no such ' + kind in error or 'no such object' in error:
                    from agent.storage.resources import resource_catalog

                    with resource_catalog(self.root.parent)._db() as db:
                        db.execute("UPDATE resources SET status='deleted',touched=? "
                                   'WHERE kind=? AND location=?',
                                   (time.time(), kind, identity))
                    return None
                raise PreparationError('Docker 查询未完成。')
            values = json.loads(result.stdout)
            if not isinstance(values, list) or len(values) != 1:
                raise PreparationError('Docker 资源记录不完整。')
            verified.append((kind, identity))
            return values

        try:
            inspected = inspect('image', image)
            if inspected is None:
                return {'state': 'missing', 'message':
                        '保存的项目镜像已不存在，请明确重新准备；已有任务仍保留原绑定。'}
            actual = inspected[0]
            if (actual.get('Id') != image or actual.get('Os') != 'linux'
                    or actual.get('Architecture') != 'amd64'):
                return {'state': 'unknown', 'message': '项目镜像身份或平台不一致，请核对配置。'}
            usable = dependency_cache_usable(
                image, binding.get('managed'), root=self.root.parent,
                inspect_volume=lambda name: inspect('volume', name),
                cancelled=lambda: time.monotonic() >= deadline,
            )
            if not usable:
                return {'state': 'missing', 'message':
                        '保存的依赖缓存缺失或与记录不一致，请明确重新准备；原任务现场已保留。'}
            from agent.storage.resources import resource_catalog

            # 原身份重新出现且完整核验通过后，同步展示台账；不恢复任何执行游标。
            with resource_catalog(self.root.parent)._db() as db:
                db.executemany("UPDATE resources SET status='ready',touched=? "
                               "WHERE kind=? AND location=? AND status='deleted'",
                               [(time.time(), kind, identity) for kind, identity in verified])
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            return {'state': 'unknown', 'message':
                    '暂时无法核验保存的环境，请检查 Docker 后重新连接；不会自动下载或准备。',
                    'error_type': type(exc).__name__}
        return {'state': 'ready', 'message': '已核验保存的项目环境，可以复用。'}

    def _run(self, arguments, *, log, cancelled=None, timeout=60, check=True):
        _check_cancel(cancelled)
        # 身份核验不需要 Env/Cmd；一次性中继口令不随 inspect 写入本地日志。
        if arguments[:2] == ['container', 'inspect'] and len(arguments) == 3:
            arguments = [*arguments[:2], '--format', _CONTAINER_INSPECT, arguments[2]]
        log.parent.mkdir(parents=True, exist_ok=True)
        with _preparation_log(log) as output:
            try:
                process = subprocess.Popen(
                    [docker_executable(), *arguments],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=(_DOWNLOAD_RELAY.get().environment()
                         if _DOWNLOAD_RELAY.get() is not None else None),
                    **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}),
                )
            except OSError as exc:
                raise PreparationError('无法连接 Docker，请启动 Docker Desktop 后重试。') from exc
            started = time.monotonic()
            checked_space = started
            try:
                while process.poll() is None:
                    _check_cancel(cancelled)
                    if hasattr(self, '_docker_storage') and time.monotonic() - checked_space >= 1:
                        self._space_guard()
                        checked_space = time.monotonic()
                    if time.monotonic() - started > timeout:
                        raise PreparationError(
                            '环境准备超时。已保留日志；请检查网络或项目的安装步骤后重试。'
                        )
                    if log.stat().st_size > MAX_LOG:
                        raise PreparationError('安装程序输出过多，已停止准备并保留日志。')
                    time.sleep(0.1)
            except BaseException:
                process.kill()
                process.wait(timeout=10)
                raise
        if log.stat().st_size > MAX_LOG:
            raise PreparationError('安装程序输出过多，已停止准备并保留日志。')
        text = log.read_text(encoding='utf-8', errors='replace')
        if check and process.returncode:
            # 本地保留日志以便诊断；不要向模型或界面返回任意安装输出，
            # 因为其中可能包含项目数据。
            if 'Public download destination required' in text:
                raise PreparationError(
                    '下载地址未通过公网安全校验，地址解析结果不安全或无效；'
                    '未开始执行项目，详细原因已保存在本机安装日志。'
                )
            raise PreparationError(
                '自动准备未完成。请检查 Docker、网络或项目依赖声明；详细原因已保存在本机安装日志。'
            )
        return subprocess.CompletedProcess(arguments, process.returncode, text, '')

    def _image(self, image, work, cancelled=None):
        result = self._run(
            ['image', 'inspect', image], log=work / 'inspect.log', cancelled=cancelled, check=False
        )
        if result.returncode:
            return None
        try:
            value = json.loads(result.stdout)[0]
            if (
                not IMAGE_ID.fullmatch(value['Id'])
                or value['Os'] != 'linux'
                or value['Architecture'] != 'amd64'
            ):
                raise ValueError('incompatible image')
            return value
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise PreparationError('Docker 返回的环境信息无效，未开始执行。') from exc

    def _dependencies_usable(self, binding, work, cancelled):
        def inspect_volume(name):
            result = self._run(
                ['volume', 'inspect', name],
                log=work / ('volume-' + hashlib.sha256(name.encode()).hexdigest()[:16] + '.log'),
                cancelled=cancelled,
                check=False,
            )
            if result.returncode:
                if 'no such volume' in result.stdout.lower():
                    return None
                raise PreparationError('无法核验已保存的依赖，请检查 Docker 后重试。')
            try:
                return json.loads(result.stdout)
            except ValueError as exc:
                raise PreparationError('Docker 返回的依赖信息不完整，未开始执行。') from exc

        return dependency_cache_usable(
            binding['image'],
            binding.get('managed'),
            root=self.root.parent,
            inspect_volume=inspect_volume,
            cancelled=cancelled,
        )

    # ============================================================
    # 基础镜像与安装材料
    # ============================================================
    def _base(self, plan, work, progress, cancelled):
        recipe = {'revision': REVISION, 'python': plan.python, 'node': plan.node}
        key = _digest(recipe)[:24]
        tag = 'monoplanner-base:' + key
        def record(image):
            from agent.storage.resources import resource_catalog, current_owner

            resource_catalog(self.root.parent).register(
                'image:' + image['Id'], 'image', image['Id'], size_bytes=image.get('Size'),
                metadata={'base_key': key}, owner=current_owner())
            return image['Id']
        with _lock(self.root / 'locks' / ('base-' + key + '.lock'), cancelled):
            image = self._image(tag, work, cancelled)
            if image:
                return record(image)
            progress('downloading', '正在下载 Python、Node.js 和常用开发工具。')
            build = work / 'base'
            build.mkdir()
            # Docker 构建上下文中只包含受信任的安装方案，
            # 不包含项目文件、用户 Dockerfile 或本机凭据文件。
            dockerfile = f'''FROM node:{plan.node}-bookworm AS node
FROM python:{plan.python}-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \\
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && \\
    python -m venv /opt/monoplanner/venv && mkdir -p /opt/monoplanner/node
ENV PATH="/opt/monoplanner/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PYTHONUNBUFFERED=1
ENV COREPACK_HOME=/opt/monoplanner/corepack
WORKDIR /testbed
LABEL org.monoplanner.managed-base="{key}"
'''
            (build / 'Dockerfile').write_text(dockerfile, encoding='utf-8')
            try:
                proxy_arguments = [
                    arg for name in PROXY_VARIABLES for arg in ('--build-arg', name + '=')
                ]
                self._build_base(build, tag, work, proxy_arguments, cancelled)
            except PreparationCancelled:
                raise
            except PreparationError as exc:
                raise PreparationError(
                    '无法下载或准备基础工具。' + str(exc)
                ) from exc
            image = self._image(tag, work, cancelled)
            if not image:
                raise PreparationError('基础环境生成后无法读取，未开始执行。')
            return record(image)

    def _copy(self, plan, work, cancelled):
        target = work / 'source'
        target.mkdir()
        from agent.execution.installation_materials import collect
        from agent.execution.managed_plan import safe_path

        if plan.materials is None:
            plan.materials = collect(plan, cancelled)
        from agent.storage.resources import preflight_space

        preflight_space(work, sum(item['size'] for item in plan.materials.values()))
        _save(work / 'installation-materials.json', plan.materials)
        for name, item in plan.materials.items():
            relative = Path(name)
            data = safe_path(plan.root, relative).read_bytes()
            if len(data) != item['size'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise PreparationError(f'安装资料在复制前发生变化：{name}')
            _check_cancel(cancelled)
            if (
                relative.as_posix() in plan.files
                and hashlib.sha256(data).hexdigest() != plan.files[relative.as_posix()]
            ):
                raise PreparationError('项目依赖声明在准备期间发生变化，请保存后重试。')
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for relative, expected in plan.files.items() if plan.projects else []:
            copied = target / relative
            if not copied.is_file() or hashlib.sha256(copied.read_bytes()).hexdigest() != expected:
                raise PreparationError('项目依赖资料未完整复制，未开始安装。')
        return target

    # ============================================================
    # 隔离容器中的依赖安装
    # ============================================================
    def _setup(self, plan, base, work, progress, cancelled):
        with _default_downloads():
            return self._setup_with_downloads(plan, base, work, progress, cancelled)

    def _setup_with_downloads(self, plan, base, work, progress, cancelled):
        identity = uuid.uuid4().hex
        network, proxy, setup = (
            'monoplanner-prepare-' + identity + suffix for suffix in ('-net', '-proxy', '-work')
        )
        from agent.storage.resources import (
            current_owner, register_active_container, unregister_active_container,
            owner_cancelled, resource_catalog, ownership_labels,
        )

        control_keys = []
        owner = current_owner('prepare:' + identity)
        labels = ownership_labels(owner, self.root.parent)
        label_args = [part for key, value in labels.items() for part in ('--label', key + '=' + value)]
        catalog = resource_catalog(self.root.parent)
        owned_records = {}
        container_records = {}
        complete = False

        def register(name):
            metadata = {'container_id': None, 'preparation': str(work),
                        'image': base, 'labels': labels}
            container_records[name] = metadata
            owned_records[name] = catalog.register(
                'container:' + name, 'container', name, owner=owner,
                metadata=metadata,
            )
            def stop():
                return self._stop_owned_container(name, container_records[name], owner)
            control_keys.append(register_active_container(
                owner, name, stop, root=self.root.parent
            ))

        def record_identity(name, result):
            container_id = result.stdout.strip()
            if re.fullmatch(r'[0-9a-f]{64}', container_id):
                container_records[name]['container_id'] = container_id
                catalog.register(owned_records[name], 'container', name, owner=owner,
                                 metadata=container_records[name])
            else:
                raise PreparationError('环境准备容器身份未确认，已保留现场。')
        source = self._copy(plan, work, cancelled)
        from agent.execution import managed_proxy, managed_setup

        proxy_code = Path(managed_proxy.__file__).read_text(encoding='utf-8')
        setup_code = Path(managed_setup.__file__).read_bytes()
        (work / 'setup.py').write_bytes(setup_code)
        _save(work / 'plan.json', {'projects': plan.projects})
        constraints = [
            '--cap-drop',
            'ALL',
            '--security-opt',
            'no-new-privileges',
            '--pids-limit',
            '256',
            '--cpus',
            '2',
            '--memory',
            '3g',
            *LOG_OPTIONS,
            *label_args,
        ]
        try:
            # 分派前先记录名称：取消或 CLI 失败时，Docker 可能已经
            # 创建了这个名称唯一、由程序管理的资源。
            owned_records[network] = catalog.register('network:' + network, 'network', network,
                owner=owner, metadata={'network_id': None, 'labels': labels})
            self._run(
                [
                    'network',
                    'create',
                    *label_args,
                    '--internal',
                    '--opt',
                    'com.docker.network.bridge.gateway_mode_ipv4=isolated',
                    network,
                ],
                log=work / 'network.log',
                cancelled=cancelled,
            )
            inspected = self._run(
                ['network', 'inspect', network],
                log=work / 'network-inspect.log',
                cancelled=cancelled,
            )
            network_state = json.loads(inspected.stdout)[0]
            if (
                network_state.get('Internal') is not True
                or network_state.get('EnableIPv6') is not False
                or network_state.get('Driver') != 'bridge'
                or network_state.get('Options', {}).get(
                    'com.docker.network.bridge.gateway_mode_ipv4'
                )
                != 'isolated'
            ):
                raise PreparationError(
                    'Docker 未提供所需的准备网络隔离，请更新 Docker Desktop 后重试。'
                )
            network_id = network_state.get('Id')
            if not isinstance(network_id, str) or not re.fullmatch(r'[0-9a-f]{64}', network_id):
                raise PreparationError('准备网络身份未确认，现场已保留。')
            catalog.register(owned_records[network], 'network', network, owner=owner,
                             metadata={'network_id': network_id, 'labels': labels})
            register(proxy)
            # 即使后端意外退出，容器内部的超时控制仍有效。
            # 主进程退出后，init 也会结束容器中的子进程命名空间。
            proxy_result = self._run(
                [
                    'run',
                    '--detach',
                    '--pull',
                    'never',
                    '--name',
                    proxy,
                    '--read-only',
                    '--tmpfs',
                    '/tmp:rw,nosuid,nodev,size=16777216',
                    *constraints,
                    '--init',
                    *proxy_environment(),
                    '--env',
                    'MONOPLANNER_DOWNLOAD_RELAY=' + _DOWNLOAD_RELAY.get().url(container=True),
                    '--entrypoint',
                    '/usr/bin/timeout',
                    base,
                    '--signal=TERM',
                    '--kill-after=10s',
                    str(PROXY_LIFETIME) + 's',
                    '/usr/local/bin/python',
                    '-c',
                    proxy_code,
                ],
                log=work / 'proxy.log',
                cancelled=cancelled,
            )
            record_identity(proxy, proxy_result)
            self._run(
                ['network', 'connect', '--alias', 'download-proxy', network, proxy],
                log=work / 'proxy-connect.log',
                cancelled=cancelled,
            )
            register(setup)
            setup_result = self._run(
                [
                    'create',
                    '--name',
                    setup,
                    '--pull',
                    'never',
                    '--network',
                    network,
                    *constraints,
                    '--init',
                    *proxy_environment('http://download-proxy:3128'),
                    '--env',
                    'npm_config_audit=false',
                    '--env',
                    'npm_config_fund=false',
                    '--env',
                    'YARN_NODE_LINKER=node-modules',
                    '--entrypoint',
                    '/usr/bin/timeout',
                    base,
                    '--signal=TERM',
                    '--kill-after=10s',
                    str(SETUP_TIMEOUT) + 's',
                    '/opt/monoplanner/venv/bin/python',
                    '/opt/monoplanner/setup.py',
                ],
                log=work / 'setup-create.log',
                cancelled=cancelled,
            )
            record_identity(setup, setup_result)
            for local, destination in (
                (source, '/monoplanner-source'),
                (work / 'setup.py', '/opt/monoplanner/setup.py'),
                (work / 'plan.json', '/monoplanner-plan.json'),
            ):
                self._run(
                    ['cp', str(local), setup + ':' + destination],
                    log=work / ('copy-' + local.name + '.log'),
                    cancelled=cancelled,
                )
            progress('installing', '正在按项目配置安装依赖，安装过程在独立容器中运行。')
            self._run(
                ['start', '--attach', setup],
                log=work / 'install.log',
                cancelled=cancelled,
                timeout=SETUP_TIMEOUT,
            )
            # docker start --attach 可能在容器随后失败的情况下仍返回零；
            # 应以记录中的容器退出码为准。
            completed = self._run(
                ['inspect', '--format', '{{.State.ExitCode}}', setup],
                log=work / 'exit.log',
                cancelled=cancelled,
            )
            if completed.stdout.strip() != '0':
                raise PreparationError(
                    '项目依赖安装失败，已保留本机安装日志；不会在本机执行或忽略缺失依赖。'
                )
            self._run(
                ['cp', setup + ':/monoplanner-result.json', str(work / 'result.json')],
                log=work / 'copy-result.log',
                cancelled=cancelled,
            )
            result = _load(work / 'result.json')
            modules = result.get('node_modules') if isinstance(result, dict) else None
            if not isinstance(modules, list) or len(modules) > 64:
                raise PreparationError('项目环境检查结果无效。')
            seen = set()
            for item in modules:
                relative = item.get('relative') if isinstance(item, dict) else None
                if (
                    not isinstance(relative, str)
                    or relative in seen
                    or '\\' in relative
                    or ':' in relative
                    or relative.startswith('/')
                    or '..' in Path(relative).parts
                    or item.get('image_path')
                    != '/opt/monoplanner/node/'
                    + ('' if relative == '.' else relative + '/')
                    + 'node_modules'
                ):
                    raise PreparationError('项目依赖目录检查未通过。')
                seen.add(relative)
            progress('saving', '正在保存可复用环境；后续执行保持无网络隔离。')
            tag = 'monoplanner-project:' + plan.fingerprint[:32]
            commit = self._run(
                [
                    'commit',
                    '--change',
                    'ENV ' + ' '.join(name + '=' for name in PROXY_VARIABLES),
                    '--change',
                    'LABEL org.monoplanner.project-fingerprint=' + plan.fingerprint,
                    '--change',
                    'ENTRYPOINT ["/bin/bash"]',
                    '--change',
                    'CMD ["-lc", "sleep infinity"]',
                    setup,
                    tag,
                ],
                log=work / 'commit.log',
                cancelled=cancelled,
                timeout=300,
            )
            image = self._image(commit.stdout.strip(), work, cancelled)
            if not image or not IMAGE_ID.fullmatch(commit.stdout.strip()):
                raise PreparationError('已准备环境无法固定版本，未开始执行。')
            complete = True
            return {
                'backend': 'docker',
                'image': image['Id'],
                'runtime': {'python': '/opt/monoplanner/venv/bin/python', 'conda_env': ''},
                'managed': {
                    'revision': REVISION,
                    'fingerprint': plan.fingerprint,
                    'python': plan.python,
                    'node': plan.node,
                    'node_modules': modules,
                    'manifest_count': len(plan.files),
                    'base_image': base,
                },
            }
        finally:
            # 只能清理本次尝试生成的资源名称。
            interrupted = (not complete or owner_cancelled(owner, self.root.parent)
                           or (cancelled and cancelled()))
            if interrupted:
                from agent.storage.resources import cancel_owned_containers

                cancel_owned_containers(owner, root=self.root.parent)
            try:
                self._finish_resources(owner, list(owned_records.values()), work, completed=not interrupted)
            finally:
                for key in control_keys:
                    unregister_active_container(key)

    def _empty_environment(self, plan, base, work, progress, cancelled):
        """无项目依赖时，共享同一套已准备的基础工具。"""
        from agent.execution.managed_plan import ProjectPlan

        empty = ProjectPlan(
            root=self.root / 'base-project',
            python=plan.python,
            node=plan.node,
            recipe=plan.recipe,
            materials={},
        )
        key = empty.fingerprint
        record = self.root / 'bases' / (key + '.json')
        with _lock(self.root / 'locks' / ('empty-' + key + '.lock'), cancelled):
            saved = _load(record)
            if saved and saved.get('fingerprint') == key:
                binding = saved['binding']
                image = self._image(binding['image'], work, cancelled)
                if (
                    image
                    and image['Id'] == binding['image']
                    and self._dependencies_usable(binding, work, cancelled)
                ):
                    progress('ready', '已复用基础工具环境，无需复制项目资料。')
                    return binding
            binding = self._setup(empty, base, work, progress, cancelled)
            _save(record, {'fingerprint': key, 'binding': binding})
            return binding

    # ============================================================
    # 环境准备的主流程
    # ============================================================
    def prepare(self, repo, progress=None, cancelled=None):
        plan = inspect_project(repo)
        record = self._record(plan.root)
        project_key = record.stem
        with _lock(self.root / 'locks' / (project_key + '.lock'), cancelled):
            _check_cancel(cancelled)
            # 其他等待者可能在获取锁前读到了旧记录。
            plan = inspect_project(repo)
            work = self.root / 'attempts' / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex)
            work.mkdir(parents=True)

            def emit(stage, message):
                value = {
                    'stage': stage,
                    'message': message,
                    'fingerprint': plan.fingerprint,
                    'projects': plan.public()['projects'],
                    'materials': (
                        {
                            'file_count': len(plan.materials),
                            'total_bytes': sum(v['size'] for v in plan.materials.values()),
                        }
                        if plan.materials is not None
                        else None
                    ),
                    'updated_at': time.time(),
                }
                _save(work / 'status.json', value)
                if progress:
                    try:
                        progress(value)
                    except Exception:
                        # 显示订阅方不能取消安装进程。
                        pass

            emit('checking', '正在检查项目文件和已保存的环境。')
            try:
                cached = _load(record)
                if cached and cached.get('fingerprint') == plan.fingerprint:
                    image = self._image(cached.get('binding', {}).get('image', ''), work, cancelled)
                    if (
                        image
                        and image['Id'] == cached['binding']['image']
                        and self._dependencies_usable(cached['binding'], work, cancelled)
                    ):
                        emit('ready', '已复用这个项目的环境。')
                        return self._register_binding(repo, cached['binding'], work)
                # 镜像下载前同时核验记录盘与 Docker 数据盘；缓存复用无需下载。
                self._space_guard(initial=True)
                self._run(
                    ['info', '--format', '{{.OSType}}'],
                    log=work / 'docker-info.log',
                    cancelled=cancelled,
                )
                from agent.execution.installation_materials import collect, summary

                plan.materials = collect(plan, cancelled)
                materials = summary(plan.materials)
                emit(
                    'materials',
                    f"已确定安装资料：{materials['file_count']} 个文件，{materials['total_bytes']:,} 字节。",
                )
                base = self._base(plan, work, emit, cancelled)
                binding = (
                    self._setup(plan, base, work, emit, cancelled)
                    if plan.projects
                    else self._empty_environment(plan, base, work, emit, cancelled)
                )
                if inspect_project(repo).fingerprint != plan.fingerprint:
                    raise PreparationError(
                        '依赖配置在安装过程中变化。已保留旧环境；请保存配置后重新准备。'
                    )
                _save(
                    record,
                    {
                        'fingerprint': plan.fingerprint,
                        'binding': binding,
                        'prepared_at': time.time(),
                        'attempt': work.name,
                        'materials': materials,
                    },
                )
                emit('ready', '项目环境已就绪，下次可以直接复用。')
                return self._register_binding(repo, binding, work)
            except BaseException as exc:
                emit(
                    'cancelled' if isinstance(exc, PreparationCancelled) else 'failed',
                    (
                        str(exc)
                        if isinstance(exc, PreparationError)
                        else '环境准备未完成，已保留日志。'
                    ),
                )
                raise
            finally:
                source = work / 'source'
                import sys
                from agent.storage.resources import owner_cancelled

                if source.exists() and sys.exc_info()[0] is None and not owner_cancelled():
                    # 这里只处理上面创建的临时目录，而不是用户选择的路径；
                    # 递归清理前必须拒绝链接。
                    if _link(source) or not source.resolve().is_relative_to(self.root.resolve()):
                        raise PreparationError('临时资料路径异常，已保留现场。')
                    shutil.rmtree(source)


# ============================================================
# 对外查询与准备入口
# ============================================================
def status(repo):
    return ManagedEnvironments().status(repo)


def prepare(repo, progress=None, cancelled=None):
    return ManagedEnvironments().prepare(repo, progress=progress, cancelled=cancelled)
