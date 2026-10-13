"""Prepare reusable project environments before the unchanged Docker task binding.

Only this preparation path downloads dependencies. It never installs into the
host, mounts user source writable, or grants a task a network connection.
"""
from contextlib import contextmanager
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

from agent.execution.docker import (docker_executable, proxy_environment, PROXY_VARIABLES,
                                    dependency_cache_usable)
from agent.execution.managed_plan import (PreparationError, REVISION, inspect_project,
                                          source_files, _digest, _link)
from agent.storage.preferences import atomic_write, state_root

MAX_LOG = 16 * 1024 * 1024
SETUP_TIMEOUT = 1800
PROXY_LIFETIME = SETUP_TIMEOUT + 300
IMAGE_ID = re.compile(r'sha256:[0-9a-f]{64}')
_LOCKS = weakref.WeakValueDictionary()
_LOCKS_GUARD = threading.Lock()


class PreparationCancelled(PreparationError):
    pass


def _check_cancel(cancelled):
    if cancelled and cancelled():
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


@contextmanager
def _lock(path, cancelled=None):
    """OS-owned lock is released on process death; no stale lock-file deletion."""
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    while not lock.acquire(timeout=.1):
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
                time.sleep(.1)
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


class ManagedEnvironments:
    def __init__(self, root=None):
        self.root = (Path(root) if root else state_root()) / 'managed-environments'

    def _record(self, repo):
        return self.root / 'projects' / (_digest(str(Path(repo).resolve())) + '.json')

    def status(self, repo):
        plan = inspect_project(repo)
        record = _load(self._record(plan.root))
        current = record and record.get('fingerprint') == plan.fingerprint
        return {**plan.public(), 'state': 'ready' if current else 'needs_preparation',
                **(record.get('binding', {}) if current else {}),
                'cached': bool(current), 'image': record.get('binding', {}).get('image') if current else None,
                'message': '已保存环境，将在执行前核验并复用。' if current else '首次工作时将自动准备项目环境。'}

    def _run(self, arguments, *, log, cancelled=None, timeout=60, check=True):
        _check_cancel(cancelled)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('wb') as output:
            try:
                process = subprocess.Popen([docker_executable(), *arguments], stdout=output,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))
            except OSError as exc:
                raise PreparationError('无法连接 Docker，请启动 Docker Desktop 后重试。') from exc
            started = time.monotonic()
            try:
                while process.poll() is None:
                    _check_cancel(cancelled)
                    if time.monotonic() - started > timeout:
                        raise PreparationError('环境准备超时。已保留日志；请检查网络或项目的安装步骤后重试。')
                    if log.stat().st_size > MAX_LOG:
                        raise PreparationError('安装程序输出过多，已停止准备并保留日志。')
                    time.sleep(.1)
            except BaseException:
                process.kill()
                process.wait(timeout=10)
                raise
        text = log.read_text(encoding='utf-8', errors='replace')
        if check and process.returncode:
            # The local log is available for diagnosis. Do not return arbitrary
            # installer output (possibly containing project data) to the model/UI.
            raise PreparationError('自动准备未完成。请检查 Docker、网络或项目依赖声明；详细原因已保存在本机安装日志。')
        return subprocess.CompletedProcess(arguments, process.returncode, text, '')

    def _image(self, image, work, cancelled=None):
        result = self._run(['image', 'inspect', image], log=work / 'inspect.log',
                           cancelled=cancelled, check=False)
        if result.returncode:
            return None
        try:
            value = json.loads(result.stdout)[0]
            if not IMAGE_ID.fullmatch(value['Id']) or value['Os'] != 'linux' or value['Architecture'] != 'amd64':
                raise ValueError('incompatible image')
            return value
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise PreparationError('Docker 返回的环境信息无效，未开始执行。') from exc

    def _dependencies_usable(self, binding, work, cancelled):
        def inspect_volume(name):
            result = self._run(['volume', 'inspect', name],
                               log=work / ('volume-' + hashlib.sha256(name.encode()).hexdigest()[:16] + '.log'),
                               cancelled=cancelled, check=False)
            if result.returncode:
                if 'no such volume' in result.stdout.lower():
                    return None
                raise PreparationError('无法核验已保存的依赖，请检查 Docker 后重试。')
            try:
                return json.loads(result.stdout)
            except ValueError as exc:
                raise PreparationError('Docker 返回的依赖信息不完整，未开始执行。') from exc
        return dependency_cache_usable(binding['image'], binding.get('managed'), root=self.root.parent,
                                       inspect_volume=inspect_volume, cancelled=cancelled)

    def _base(self, plan, work, progress, cancelled):
        recipe = {'revision': REVISION, 'python': plan.python, 'node': plan.node}
        key = _digest(recipe)[:24]
        tag = 'monoplanner-base:' + key
        with _lock(self.root / 'locks' / ('base-' + key + '.lock'), cancelled):
            image = self._image(tag, work, cancelled)
            if image:
                return image['Id']
            progress('downloading', '正在下载 Python、Node.js 和常用开发工具。')
            build = work / 'base'
            build.mkdir()
            # Only the trusted recipe enters Docker's build context. Project
            # files, user Dockerfiles and host credential files are not included.
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
                proxy_arguments = [arg for name in PROXY_VARIABLES for arg in ('--build-arg', name + '=')]
                self._run(['build', '--network', 'none', *proxy_arguments, '--tag', tag, str(build)],
                          log=work / 'base-build.log', cancelled=cancelled, timeout=SETUP_TIMEOUT)
            except PreparationCancelled:
                raise
            except PreparationError as exc:
                raise PreparationError('无法下载或准备基础工具。请检查 Docker Desktop 的网络或代理设置后重试；详细原因已保存在本机日志。') from exc
            image = self._image(tag, work, cancelled)
            if not image:
                raise PreparationError('基础环境生成后无法读取，未开始执行。')
            return image['Id']

    def _copy(self, plan, work, cancelled):
        target = work / 'source'
        target.mkdir()
        for relative, data in source_files(plan.root):
            _check_cancel(cancelled)
            if relative.as_posix() in plan.files and hashlib.sha256(data).hexdigest() != plan.files[relative.as_posix()]:
                raise PreparationError('项目依赖声明在准备期间发生变化，请保存后重试。')
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for relative, expected in plan.files.items():
            copied = target / relative
            if not copied.is_file() or hashlib.sha256(copied.read_bytes()).hexdigest() != expected:
                raise PreparationError('项目依赖资料未完整复制，未开始安装。')
        return target

    def _setup(self, plan, base, work, progress, cancelled):
        identity = uuid.uuid4().hex
        network, proxy, setup = ('monoplanner-prepare-' + identity + suffix for suffix in ('-net', '-proxy', '-work'))
        owned = {'network': False, 'proxy': False, 'setup': False}
        source = self._copy(plan, work, cancelled)
        from agent.execution import managed_proxy, managed_setup
        proxy_code = Path(managed_proxy.__file__).read_text(encoding='utf-8')
        setup_code = Path(managed_setup.__file__).read_bytes()
        (work / 'setup.py').write_bytes(setup_code)
        _save(work / 'plan.json', {'projects': plan.projects})
        constraints = ['--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                       '--pids-limit', '256', '--cpus', '2', '--memory', '3g']
        try:
            # Mark before dispatch: cancellation or a CLI failure may occur after
            # Docker created the uniquely named owned resource.
            owned['network'] = True
            self._run(['network', 'create', '--internal', '--opt',
                       'com.docker.network.bridge.gateway_mode_ipv4=isolated', network],
                      log=work / 'network.log', cancelled=cancelled)
            inspected = self._run(['network', 'inspect', network],
                                  log=work / 'network-inspect.log', cancelled=cancelled)
            network_state = json.loads(inspected.stdout)[0]
            if (network_state.get('Internal') is not True
                    or network_state.get('EnableIPv6') is not False
                    or network_state.get('Driver') != 'bridge'
                    or network_state.get('Options', {}).get('com.docker.network.bridge.gateway_mode_ipv4') != 'isolated'):
                raise PreparationError('Docker 未提供所需的准备网络隔离，请更新 Docker Desktop 后重试。')
            owned['proxy'] = True
            # Container-owned deadlines survive an abrupt backend exit. When the
            # main process exits, init also ends the container's child namespace.
            self._run(['run', '--detach', '--pull', 'never', '--name', proxy,
                       '--read-only', '--tmpfs', '/tmp:rw,nosuid,nodev,size=16777216',
                       *constraints, '--init', *proxy_environment(),
                       '--entrypoint', '/usr/bin/timeout', base, '--signal=TERM', '--kill-after=10s',
                       str(PROXY_LIFETIME) + 's', '/usr/local/bin/python', '-c', proxy_code],
                      log=work / 'proxy.log', cancelled=cancelled)
            self._run(['network', 'connect', '--alias', 'download-proxy', network, proxy],
                      log=work / 'proxy-connect.log', cancelled=cancelled)
            owned['setup'] = True
            self._run(['create', '--name', setup, '--pull', 'never', '--network', network,
                       *constraints, '--init', *proxy_environment('http://download-proxy:3128'),
                       '--env', 'npm_config_audit=false', '--env', 'npm_config_fund=false',
                       '--env', 'YARN_NODE_LINKER=node-modules',
                       '--entrypoint', '/usr/bin/timeout', base, '--signal=TERM', '--kill-after=10s',
                       str(SETUP_TIMEOUT) + 's', '/opt/monoplanner/venv/bin/python', '/opt/monoplanner/setup.py'],
                      log=work / 'setup-create.log', cancelled=cancelled)
            for local, destination in ((source, '/monoplanner-source'), (work / 'setup.py', '/opt/monoplanner/setup.py'),
                                       (work / 'plan.json', '/monoplanner-plan.json')):
                self._run(['cp', str(local), setup + ':' + destination], log=work / ('copy-' + local.name + '.log'), cancelled=cancelled)
            progress('installing', '正在按项目配置安装依赖，安装过程在独立容器中运行。')
            self._run(['start', '--attach', setup], log=work / 'install.log', cancelled=cancelled, timeout=SETUP_TIMEOUT)
            # docker start --attach may return zero when the daemon started a
            # container that subsequently failed. Its recorded exit is definitive.
            completed = self._run(['inspect', '--format', '{{.State.ExitCode}}', setup],
                                  log=work / 'exit.log', cancelled=cancelled)
            if completed.stdout.strip() != '0':
                raise PreparationError('项目依赖安装失败，已保留本机安装日志；不会在本机执行或忽略缺失依赖。')
            self._run(['cp', setup + ':/monoplanner-result.json', str(work / 'result.json')],
                      log=work / 'copy-result.log', cancelled=cancelled)
            result = _load(work / 'result.json')
            modules = result.get('node_modules') if isinstance(result, dict) else None
            if not isinstance(modules, list) or len(modules) > 64:
                raise PreparationError('项目环境检查结果无效。')
            seen = set()
            for item in modules:
                relative = item.get('relative') if isinstance(item, dict) else None
                if (not isinstance(relative, str) or relative in seen or '\\' in relative or ':' in relative
                        or relative.startswith('/') or '..' in Path(relative).parts
                        or item.get('image_path') != '/opt/monoplanner/node/' + ('' if relative == '.' else relative + '/') + 'node_modules'):
                    raise PreparationError('项目依赖目录检查未通过。')
                seen.add(relative)
            progress('saving', '正在保存可复用环境；后续执行保持无网络隔离。')
            tag = 'monoplanner-project:' + plan.fingerprint[:32]
            commit = self._run(['commit', '--change', 'ENV ' + ' '.join(name + '=' for name in PROXY_VARIABLES),
                '--change', 'LABEL org.monoplanner.project-fingerprint=' + plan.fingerprint,
                '--change', 'ENTRYPOINT ["/bin/bash"]', '--change', 'CMD ["-lc", "sleep infinity"]', setup, tag],
                log=work / 'commit.log', cancelled=cancelled, timeout=300)
            image = self._image(commit.stdout.strip(), work, cancelled)
            if not image or not IMAGE_ID.fullmatch(commit.stdout.strip()):
                raise PreparationError('已准备环境无法固定版本，未开始执行。')
            return {'backend': 'docker', 'image': image['Id'],
                    'runtime': {'python': '/opt/monoplanner/venv/bin/python', 'conda_env': ''},
                    'managed': {'revision': REVISION, 'fingerprint': plan.fingerprint,
                                'python': plan.python, 'node': plan.node, 'node_modules': modules,
                                'manifest_count': len(plan.files)}}
        finally:
            # Only names generated for this attempt are eligible for cleanup.
            for kind, name in (('setup', setup), ('proxy', proxy)):
                if owned[kind]:
                    try:
                        self._run(['rm', '--force', name], log=work / ('cleanup-' + kind + '.log'), check=False)
                    except (PreparationError, OSError):
                        pass
            if owned['network']:
                try:
                    self._run(['network', 'rm', network], log=work / 'cleanup-network.log', check=False)
                except (PreparationError, OSError):
                    pass

    def prepare(self, repo, progress=None, cancelled=None):
        plan = inspect_project(repo)
        record = self._record(plan.root)
        project_key = record.stem
        with _lock(self.root / 'locks' / (project_key + '.lock'), cancelled):
            _check_cancel(cancelled)
            # Another waiter may have observed older files before acquiring lock.
            plan = inspect_project(repo)
            work = self.root / 'attempts' / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex)
            work.mkdir(parents=True)

            def emit(stage, message):
                value = {'stage': stage, 'message': message, 'fingerprint': plan.fingerprint,
                         'updated_at': time.time()}
                _save(work / 'status.json', value)
                if progress:
                    try:
                        progress(value)
                    except Exception:
                        # A display subscriber must not cancel an installer.
                        pass

            emit('checking', '正在检查项目文件和已保存的环境。')
            try:
                cached = _load(record)
                if cached and cached.get('fingerprint') == plan.fingerprint:
                    image = self._image(cached.get('binding', {}).get('image', ''), work, cancelled)
                    if (image and image['Id'] == cached['binding']['image']
                            and self._dependencies_usable(cached['binding'], work, cancelled)):
                        emit('ready', '已复用这个项目的环境。')
                        return cached['binding']
                self._run(['info', '--format', '{{.OSType}}'], log=work / 'docker-info.log', cancelled=cancelled)
                base = self._base(plan, work, emit, cancelled)
                binding = self._setup(plan, base, work, emit, cancelled)
                if inspect_project(repo).fingerprint != plan.fingerprint:
                    raise PreparationError('依赖配置在安装过程中变化。已保留旧环境；请保存配置后重新准备。')
                _save(record, {'fingerprint': plan.fingerprint, 'binding': binding,
                               'prepared_at': time.time(), 'attempt': work.name})
                emit('ready', '项目环境已就绪，下次可以直接复用。')
                return binding
            except BaseException as exc:
                emit('cancelled' if isinstance(exc, PreparationCancelled) else 'failed',
                     str(exc) if isinstance(exc, PreparationError) else '环境准备未完成，已保留日志。')
                raise
            finally:
                source = work / 'source'
                if source.exists():
                    # This is the exact scratch directory created above, not a
                    # user-selected path; reject links before recursive removal.
                    if _link(source) or not source.resolve().is_relative_to(self.root.resolve()):
                        raise PreparationError('临时资料路径异常，已保留现场。')
                    shutil.rmtree(source)


def status(repo):
    return ManagedEnvironments().status(repo)


def prepare(repo, progress=None, cancelled=None):
    return ManagedEnvironments().prepare(repo, progress=progress, cancelled=cancelled)
