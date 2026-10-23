"""在同一个 Docker 容器中运行任务源码和验收命令。

模型和文件查询工具运行于本机；仅挂载任务源码与执行记录，
不挂载凭据或 Docker Socket。
"""

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from agent.execution.process import log_excerpt
from agent.storage.records import digest, save

PROXY_VARIABLES = tuple(
    name
    for protocol in ('HTTP', 'HTTPS', 'FTP', 'ALL', 'NO')
    for name in (protocol + '_PROXY', protocol.lower() + '_proxy')
)
DEPENDENCY_CACHE_OPTIONS = 'rw,exec,nosuid,nodev,size=67108864,mode=1777'
LOG_OPTIONS = ['--log-driver', 'local', '--log-opt', 'max-size=10m', '--log-opt', 'max-file=3']


# ============================================================
# 代理与依赖挂载校验
# ============================================================
def proxy_environment(proxy=None):
    """覆盖 Docker CLI 默认注入的代理配置，避免其中的凭据进入容器。"""
    result = []
    for name in PROXY_VARIABLES:
        value = proxy if proxy and name.lower() in {'http_proxy', 'https_proxy'} else ''
        result.extend(['--env', name + '=' + value])
    return result


def managed_dependencies(image, managed, *, root=None):
    """只有程序管理的镜像内路径才能成为项目依赖挂载。"""
    rows = (managed or {}).get('node_modules', [])
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('Invalid managed dependency list')
    dependencies, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'relative', 'image_path'}:
            raise ValueError('Invalid managed dependency descriptor')
        relative = row['relative']
        if (
            not isinstance(relative, str)
            or not relative
            or '\\' in relative
            or any(c in relative for c in '\0\r\n,:')
            or PurePosixPath(relative).is_absolute()
            or str(PurePosixPath(relative)) != relative
            or any(
                p in {'..', '.git', '.venv', 'node_modules'} or p.startswith('.env')
                for p in PurePosixPath(relative).parts
            )
        ):
            raise ValueError('Unsafe managed dependency directory')
        expected = (
            '/opt/monoplanner/node/' + (relative + '/' if relative != '.' else '') + 'node_modules'
        )
        if row['image_path'] != expected or relative in seen:
            raise ValueError('Managed dependency image path differs from project directory')
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Managed dependencies require an immutable image ID')
        seen.add(relative)
        from agent.storage.resources import current_root

        owner = str(current_root(root))
        identity = hashlib.sha256(
            (owner + '\n' + image + '\n' + expected).encode('utf-8')
        ).hexdigest()
        dependencies.append(
            {
                'relative': relative,
                'image_path': expected,
                'identity': identity,
                'volume': 'monoplanner-node-' + identity,
                'target': '/testbed/'
                + (relative + '/' if relative != '.' else '')
                + 'node_modules',
            }
        )
    return dependencies


def dependency_mountpoint(repo, relative):
    """只创建空的挂载目录，不替换或填充本机依赖。"""
    from agent.source.snapshots import _link

    root = Path(repo).resolve()
    current = root
    parts = (() if relative == '.' else PurePosixPath(relative).parts) + ('node_modules',)
    for part in parts:
        current /= part
        if current.exists() or current.is_symlink():
            if _link(current) or not current.is_dir() or not current.resolve().is_relative_to(root):
                raise ValueError('Dependency mount contains a link or non-directory')
        else:
            current.mkdir()
    return current


def dependency_cache_usable(image, managed, *, root, inspect_volume, cancelled=None):
    """检查缓存能否用于后续任务，不修改旧数据卷记录。

    首次执行可以初始化新数据卷。已经尝试过的数据卷必须具备完整日志，
    且当前身份与记录一致；否则准备新的镜像版本，不能原地修补已绑定的任务。
    """
    from agent.execution.managed import _lock
    from agent.source.snapshots import _link

    try:
        dependencies = managed_dependencies(image, managed, root=root)
    except (TypeError, ValueError):
        return False
    root = Path(root)
    for item in dependencies:
        with _lock(root / 'managed-volume-locks' / (item['identity'] + '.lock'), cancelled):
            folder = root / 'managed-volumes' / item['identity']
            record = folder / 'completion.json'
            if folder.exists() and (_link(folder) or not record.is_file()):
                return False
            actual = inspect_volume(item['volume'])
            if not record.exists():
                if actual:
                    return False
                continue
            if _link(record):
                return False
            try:
                stored = json.loads(record.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                return False
            expected = {'status': 'completed', 'image': image, **item}
            labels = {
                'com.monoplanner.dependency': item['identity'],
                'com.monoplanner.image': image,
            }
            if (
                stored != expected
                or not isinstance(actual, list)
                or len(actual) != 1
                or actual[0].get('Name') != item['volume']
                or actual[0].get('Driver') != 'local'
                or actual[0].get('Options')
                or actual[0].get('Labels') != labels
            ):
                return False
    return True


# ============================================================
# Docker 命令与运行时配置
# ============================================================
def docker_executable():
    # 便携启动会主动从 PATH 中排除系统安装的语言运行时。
    if shutil.which('docker'):
        return 'docker'
    installed = (
        Path(os.environ.get('ProgramFiles', 'C:/Program Files'))
        / 'Docker/Docker/resources/bin/docker.exe'
    )
    return str(installed) if os.name == 'nt' and installed.is_file() else 'docker'


def runtime_profile(value=None):
    value = dict(
        value or {'python': '/opt/miniconda3/envs/testbed/bin/python', 'conda_env': 'testbed'}
    )
    if set(value) != {'python', 'conda_env'}:
        raise ValueError('运行环境仅接受 python 和 conda_env')
    python, conda = value['python'], value['conda_env']
    if (
        not isinstance(python, str)
        or not python.startswith('/')
        or '\\' in python
        or '..' in python.split('/')
        or any(c in python for c in '\0\r\n')
    ):
        raise ValueError('python 必须是容器内的绝对路径')
    if not isinstance(conda, str) or (conda and not re.fullmatch(r'[A-Za-z0-9_.-]+', conda)):
        raise ValueError('conda_env 必须是环境名称或空字符串')
    return value


# 此脚本必须兼容评测镜像中的旧版 Python。
# 超时控制在容器内部生效；不能依赖终止 Docker CLI
# 来结束真实命令及其子进程。
_PROCESS_RUNNER = r'''
import base64
import json
import os
import signal
import subprocess
import sys
import traceback

request = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
status_path = request["status_path"]
result = {"exit_code": None, "timed_out": False, "pid": None,
          "request_digest": request.get("request_digest")}

def save():
    temporary = status_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, status_path)

try:
    # POSIX accepts byte arguments. Avoid Python 3.6's ASCII locale encoding
    # before the child even starts; keep the benchmark interpreter and locale.
    argv = [arg.encode("utf-8") for arg in request["argv"]]
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    out = open(request["stdout_path"], "wb") if request.get("stdout_path") else None
    err = open(request["stderr_path"], "wb") if request.get("stderr_path") else None
    process = subprocess.Popen(argv, cwd="/testbed", env=child_env, start_new_session=True,
                               stdout=out, stderr=err, stdin=subprocess.DEVNULL)
    result["pid"] = process.pid
    save()
    try:
        result["exit_code"] = process.wait(timeout=request["timeout"])
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        result["exit_code"] = 124
except BaseException as exc:
    result.update(exit_code=127, error="{}: {}".format(type(exc).__name__, exc))
    traceback.print_exc()
finally:
    for stream in (locals().get("out"), locals().get("err")):
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    save()
sys.exit(result["exit_code"] or 0)
'''

_CLEANUP_RUNNER = r'''
import json
import os
import signal
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as source:
        status = json.load(source)
    if status.get("exit_code") is None:
        if not isinstance(status.get("pid"), int) or status["pid"] <= 1:
            sys.exit(2)
        os.killpg(status["pid"], signal.SIGKILL)
except FileNotFoundError:
    sys.exit(2)
except ProcessLookupError:
    pass
'''


# ============================================================
# 任务容器的创建与生命周期
# ============================================================
class DockerEnvironment:
    """每个任务管理自己的容器；停止容器时保留本机运行证据。"""

    python_executable = "/opt/miniconda3/envs/testbed/bin/python"

    def __init__(
        self,
        repo_root,
        image,
        container_name,
        timeout=120,
        artifact_root=None,
        *,
        runtime=None,
        material_mounts=None,
        managed=None,
        project_scope=None,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(f"工作区不存在：{self.repo_root}")
        if not self.repo_root.is_dir():
            raise NotADirectoryError(f"工作区必须是目录：{self.repo_root}")
        if not isinstance(image, str) or not image.strip() or image.startswith("-"):
            raise ValueError("Docker image 必须是非空镜像名称。")
        if not isinstance(container_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", container_name
        ):
            raise ValueError("Docker container_name 格式无效。")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("命令超时必须大于 0。")
        self.image = image
        self.runtime = runtime_profile(runtime)
        self.python_executable = self.runtime['python']
        self.material_mounts = dict(material_mounts or {})
        from agent.source.scope import project_scope as resolve_scope

        self.project_scope = resolve_scope(self.repo_root, project_scope)
        self.project_masks = {}
        self.project_mask_identity = None
        self._project_mask_registered = False
        self.project_paths = self._project_path_identities()
        self.dependencies = managed_dependencies(image, managed)
        for item in self.dependencies:
            self.project_scope.require(item['relative'] + '/node_modules')
        self.dependency_mounts = {item['target']: item['volume'] for item in self.dependencies}
        self.dependency_caches = {
            target + '/' + name: DEPENDENCY_CACHE_OPTIONS
            for target in self.dependency_mounts
            for name in ('.cache', '.vite', '.vite-temp')
        }
        self._dependencies_checked = False
        self.runtime_facts = {
            'backend': 'Docker',
            'python_executable': self.python_executable,
            'conda_env': self.runtime['conda_env'],
            'source': '/testbed',
            'network': 'none',
        }
        self.container_name = container_name
        self.timeout = timeout
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve()
            if artifact_root is not None
            else Path(tempfile.mkdtemp(prefix="coding-agent-docker-"))
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        # --mount 使用逗号分隔参数；含冲突字符的路径应拒绝，不能重新解释。
        if any("," in str(path) for path in (self.repo_root, self.artifact_root)):
            raise ValueError("Docker 挂载路径暂不支持逗号。")
        self._container_id = None
        self._closed = False
        from agent.storage.resources import current_owner, current_root

        self.resource_owner = current_owner()
        self.resource_root = current_root()
        self._control_key = None
        self._execution_unknown = False
        self.disposable = False
        self._lifetime = None

    def lifetime(self):
        if self._lifetime is None:
            from agent.runtime.liveness import Lease

            root = self.resource_root if self.resource_owner else self.artifact_root.parent
            self._lifetime = Lease(root, self.container_name)
        return self._lifetime

    def _check_cancelled(self):
        from agent.storage.resources import owner_cancelled

        if self.resource_owner and owner_cancelled(self.resource_owner, self.resource_root):
            raise InterruptedError('任务已立即暂停；容器及操作现场已保留。')

    def _kill_now(self):
        if self._lifetime is not None:
            self._lifetime.revoke()
        self._execution_unknown = True
        result = self._docker(
            ['kill', self._container_id or self.container_name], timeout=8,
            capture_output=True, text=True, encoding='utf-8', errors='replace',
        )
        # 创建尚未返回时可能暂时不存在；创建返回后的取消检查会再次终止它。
        if result.returncode and not any(text in result.stderr.lower()
                                        for text in ('no such container', 'is not running')):
            raise RuntimeError('Docker 未确认任务容器已停止。')
        return self._confirm_stopped()

    def _confirm_stopped(self):
        target = self._container_id or self.container_name
        result = self._docker(['inspect', target], timeout=5, capture_output=True,
                              text=True, encoding='utf-8', errors='replace')
        if result.returncode:
            if 'no such' in (result.stdout + result.stderr).lower():
                return {'stopped': True, 'resource_state': 'missing', 'container_id': self._container_id}
            raise RuntimeError('无法核验容器停止状态；租约已撤销。')
        actual = json.loads(result.stdout)[0]
        if (self._container_id and actual.get('Id') != self._container_id
                or actual.get('State', {}).get('Running') is not False):
            raise RuntimeError('原容器仍在运行或身份变化，不能报告已停止。')
        return {'stopped': True, 'resource_state': 'stopped', 'container_id': actual['Id']}

    def _register_control(self):
        if not self.resource_owner:
            return
        from agent.storage.resources import register_active_container, resource_catalog, ownership_labels

        if self._control_key is None:
            self._control_key = register_active_container(
                self.resource_owner, self.container_name, self._kill_now, root=self.resource_root
            )
        resource_catalog(self.resource_root).register(
            'container:' + self.container_name, 'container', self.container_name,
            metadata={'container_id': self._container_id, 'image': self.image,
                      'labels': ownership_labels(self.resource_owner, self.resource_root)},
            owner=self.resource_owner,
        )

    def clone_options(self):
        # 调查/验收副本已按源码清单排除其他项目，不携带原目录绑定或遮罩。
        options = {'runtime': self.runtime, 'material_mounts': self.material_mounts}
        if self.dependencies:
            options['managed'] = {
                'node_modules': [
                    {k: item[k] for k in ('relative', 'image_path')} for item in self.dependencies
                ]
            }
        return options

    def _project_path_identities(self):
        result = {}
        for relative in ('.', *self.project_scope.excluded_projects):
            info = (self.repo_root / relative).stat()
            result[relative] = {'device': info.st_dev, 'inode': info.st_ino}
        return result

    def _prepare_project_masks(self, *, allow_create=True, full_check=False):
        """排除项目在容器中只留下空只读目录，遮罩原件不通过其他挂载暴露。"""
        self.project_scope.validate(full_check=full_check)
        if self.project_paths != self._project_path_identities():
            raise ValueError('项目目录身份已变化，不能继续沿用执行范围。')
        if not self.project_scope.excluded_projects:
            return
        from agent.source.snapshots import _link

        mask = self.artifact_root.with_name(self.artifact_root.name + '-project-mask')
        exposed = [self.repo_root, self.artifact_root, *map(Path, self.material_mounts.values())]
        if any(mask.is_relative_to(path.resolve()) for path in exposed):
            raise ValueError('项目遮罩不能位于容器可访问的原始目录中。')
        for path in (mask, *mask.parents):
            if (path.exists() or path.is_symlink()) and _link(path):
                raise ValueError('项目遮罩目录不能包含链接。')
        if not mask.exists():
            if not allow_create:
                raise ValueError('已绑定的项目遮罩丢失，不会自动替换。')
            mask.mkdir()
        if not mask.is_dir() or next(mask.iterdir(), None) is not None:
            raise ValueError('项目遮罩必须是空目录。')
        info = mask.stat()
        identity = {'path': str(mask), 'device': info.st_dev, 'inode': info.st_ino}
        if self.project_mask_identity is not None and self.project_mask_identity != identity:
            raise ValueError('项目遮罩身份已变化，不能继续。')
        self.project_mask_identity = identity
        self.project_masks = {
            '/testbed/' + name: str(mask) for name in self.project_scope.excluded_projects
        }
        if any(',' in target for target in self.project_masks):
            raise ValueError('Docker 项目排除路径暂不支持逗号。')
        if self.resource_owner and not self._project_mask_registered:
            from agent.storage.resources import resource_catalog

            resource_catalog(self.resource_root).register(
                'project-mask:' + self.container_name, 'temporary', mask,
                owner=self.resource_owner, size_bytes=0,
                metadata={'container': self.container_name},
            )
            self._project_mask_registered = True

    @staticmethod
    def _process_options():
        return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    def _docker(self, arguments, timeout=30, **kwargs):
        return subprocess.run(
            [docker_executable(), *arguments],
            timeout=timeout,
            **self._process_options(),
            **kwargs,
        )

    def _dependency_query(self, arguments, *, timeout=30, check_after=True):
        self._check_cancelled()
        result = self._docker(
            arguments,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        if result.returncode:
            raise RuntimeError(
                '项目依赖准备失败；保留记录，不自动重试：' + result.stderr.strip()[:500]
            )
        if check_after:
            self._check_cancelled()
        return result.stdout

    def _finish_resources(self, owner, identities, *, completed):
        from agent.storage.resources import resource_catalog, delete_owned_docker_resource

        def query(arguments):
            return self._docker(arguments, timeout=15, capture_output=True,
                                text=True, encoding='utf-8', errors='replace')
        result = resource_catalog(self.resource_root).finish_resources(
            owner, identities, completed=completed, docker=lambda kind, identity, metadata:
            delete_owned_docker_resource(kind, identity, metadata, query=query, stop_running=True))
        if result.get('errors'):
            raise RuntimeError('容器资源收尾未完成，现场已保留。')
        return result

    def _seed_dependency(self, item):
        from agent.storage.resources import (
            ownership_labels, resource_catalog, register_active_container,
            unregister_active_container, owner_cancelled,
        )

        name = 'monoplanner-seed-' + uuid.uuid4().hex
        owner = self.resource_owner or 'environment:' + self.image
        labels = ownership_labels(owner, self.resource_root)
        catalog = resource_catalog(self.resource_root)
        identity = 'container:' + name
        container_id = None

        def record():
            catalog.register(identity, 'container', name, owner=owner, metadata={
                'container_id': container_id, 'image': self.image, 'labels': labels,
            })

        def stop():
            result = self._docker(['kill', container_id or name], timeout=8,
                                 capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode and not any(message in result.stderr.lower()
                                             for message in ('no such container', 'is not running')):
                raise RuntimeError('依赖复制容器停止未确认。')

        record()
        control = register_active_container(owner, name, stop, root=self.resource_root)
        complete = False
        from agent.runtime.liveness import Lease

        guard = Lease(self.resource_root, name, cancelled=lambda: owner_cancelled(owner, self.resource_root))
        try:
            arguments = [
                'create', '--pull', 'never', '--name', name, '--network', 'none', '--read-only',
                '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64',
                '--cpus', '2', '--memory', '512m', '--user', '0:0', *LOG_OPTIONS,
                *[arg for key, value in labels.items() for arg in ('--label', key + '=' + value)],
                *proxy_environment(), '--mount',
                'type=volume,source=' + item['volume'] + ',target=/prepared,volume-nocopy',
                '--entrypoint', '/bin/cp', self.image, '-R', '--no-preserve=ownership',
                item['image_path'] + '/.', '/prepared/',
            ]
            arguments = guard.wrap(arguments, self.image)
            container_id = self._dependency_query(arguments, check_after=False).strip()
            if not re.fullmatch(r'[0-9a-f]{64}', container_id):
                raise RuntimeError('依赖复制容器身份未确认。')
            record()
            self._check_cancelled()
            self._dependency_query(['start', '--attach', container_id], timeout=120)
            exit_code = self._dependency_query(
                ['inspect', '--format', '{{.State.ExitCode}}', container_id]
            ).strip()
            if exit_code != '0':
                raise RuntimeError('项目依赖准备失败；复制结果未确认。')
            complete = True
        finally:
            guard.revoke()
            try:
                if complete and not owner_cancelled(owner, self.resource_root):
                    self._finish_resources(owner, [identity], completed=True)
                else:
                    stop()
            finally:
                unregister_active_container(control)

    def _dependency_space_budget(self):
        from agent.storage.resources import resource_catalog, preflight_space
        from agent.storage.disk_space import require_docker_space

        reserve = resource_catalog(self.resource_root).usage()['reserve_bytes']
        storage = require_docker_space(reserve)
        image = json.loads(self._dependency_query(['image', 'inspect', self.image]))[0]
        size = image.get('Size')
        if image.get('Id') != self.image or type(size) is not int or size < 0:
            raise RuntimeError('依赖复制大小未确认，尚未创建数据卷。')
        # 镜像整体大小是准备量估计；额外记录估算口径，不把它当卷的实占。
        return {**preflight_space(storage['location'], size, reserve_bytes=reserve),
                'estimated': True, 'basis': 'source_image_size'}

    def _verify_dependency_volume(self, item, labels):
        actual = json.loads(self._dependency_query(['volume', 'inspect', item['volume']]))
        if (
            len(actual) != 1
            or actual[0].get('Name') != item['volume']
            or actual[0].get('Driver') != 'local'
            or actual[0].get('Options')
            or actual[0].get('Labels') != labels
        ):
            raise ValueError('项目依赖副本身份或存储方式已改变。')

    # ============================================================
    # 只读依赖数据卷的准备
    # ============================================================
    def _prepare_dependencies(self, *, allow_create=True):
        if self._dependencies_checked:
            return
        if not self.dependencies:
            self._dependencies_checked = True
            return
        from agent.execution.managed import _lock
        from agent.storage.preferences import state_root

        for item in self.dependencies:
            with _lock(self.resource_root / 'managed-volume-locks' / (item['identity'] + '.lock')):
                self._prepare_dependency(item, allow_create=allow_create)
        self._dependencies_checked = True

    def _prepare_dependency(self, item, *, allow_create):
        from agent.storage.preferences import state_root, atomic_write

        # 只读调查副本也需要校验并创建挂载点。
        # 这些是程序生成的空目录，依赖内容仍保存在 Docker 中。
        dependency_mountpoint(self.repo_root, item['relative'])
        folder = self.resource_root / 'managed-volumes' / item['identity']
        record = folder / 'completion.json'
        expected = {'image': self.image, **item}
        labels = {
            'com.monoplanner.dependency': item['identity'],
            'com.monoplanner.image': self.image,
        }
        if record.exists():
            stored = json.loads(record.read_text(encoding='utf-8'))
            if stored != {'status': 'completed', **expected}:
                raise RuntimeError('项目依赖副本准备结果不明；不会自动重复准备。')
        else:
            if not allow_create:
                raise RuntimeError('已绑定任务缺少项目依赖副本记录；不会自动重建。')
            space = self._dependency_space_budget()
            # 独占预留同时防止两个任务并发初始化同一个数据卷。
            # 已经放弃的预留不能自动重放。
            try:
                folder.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise RuntimeError('项目依赖正在准备或记录不完整；请稍后重试读取状态。') from exc
            atomic_write(record, json.dumps({'status': 'pending', **expected}).encode('utf-8'))
            atomic_write(folder / 'space-budget.json', json.dumps(space).encode('utf-8'))
            existing = self._docker(
                ['volume', 'inspect', item['volume']],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            if not existing.returncode:
                raise RuntimeError('发现没有完成记录的项目依赖副本；不会自动覆盖。')
            self._dependency_query(
                [
                    'volume',
                    'create',
                    '--label',
                    'com.monoplanner.dependency=' + item['identity'],
                    '--label',
                    'com.monoplanner.image=' + self.image,
                    item['volume'],
                ]
            )
            self._verify_dependency_volume(item, labels)
            from agent.storage.resources import resource_catalog

            resource_catalog(self.resource_root).register(
                'volume:' + item['volume'], 'volume', item['volume'],
                metadata={'identity': item['identity'], 'image': self.image},
                owner=self.resource_owner or 'environment:' + self.image,
            )
            # 这里不挂载项目、运行记录、凭据或 Docker Socket。
            # cp 会保留依赖中的符号链接，包括工作区引用；
            # 任务只读挂载最终的依赖目录。
            self._seed_dependency(item)
            atomic_write(record, json.dumps({'status': 'completed', **expected}).encode('utf-8'))
        self._verify_dependency_volume(item, labels)
        if self.resource_owner:
            from agent.storage.resources import resource_catalog

            catalog = resource_catalog(self.resource_root)
            identity = catalog.register(
                'volume:' + item['volume'], 'volume', item['volume'],
                metadata={'identity': item['identity'], 'image': self.image},
                owner=self.resource_owner,
            )
            catalog.link('environment:' + self.image, identity)

    # ============================================================
    # 启动容器并绑定工作目录
    # ============================================================
    def start(self):
        self._check_cancelled()
        self._prepare_project_masks(
            allow_create=self._container_id is None, full_check=self._container_id is None,
        )
        if self._closed:
            raise RuntimeError("此环境已关闭，请为新任务创建新环境。")
        if self._container_id is not None:
            return self
        self._register_control()
        self._prepare_dependencies()
        arguments = [
            "run",
            "--detach",
            "--pull",
            "never",
            "--name",
            self.container_name,
            "--network",
            "none",
            "--init",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--cpus",
            "2",
            "--memory",
            "3g",
            *LOG_OPTIONS,
            *proxy_environment(),
            "--mount",
            f"type=bind,source={self.repo_root},target=/testbed"
            + (",readonly" if getattr(self, "repo_read_only", False) else ""),
            "--mount",
            f"type=bind,source={self.artifact_root},target=/agent-artifacts",
            "--workdir",
            "/testbed",
            "--entrypoint",
            "/bin/bash",
            self.image,
            "-c",
            "exec sleep infinity",
        ]
        if self.resource_owner:
            from agent.storage.resources import ownership_labels

            arguments[1:1] = [arg for key, value in ownership_labels(
                self.resource_owner, self.resource_root
            ).items() for arg in ('--label', key + '=' + value)]
        if getattr(self, "root_read_only", False):
            # Docker 默认以 noexec 挂载 tmpfs；编译调查需要能够
            # 在隔离的临时文件系统中运行生成的程序。
            arguments[1:1] = ["--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=268435456"]
        for index, source in enumerate(self.material_mounts.values()):
            if ',' in str(source):
                raise ValueError('资料挂载路径暂不支持逗号')
            arguments[1:1] = [
                '--mount',
                f'type=bind,source={source},target=/agent-materials/{index},readonly',
            ]
        for target, volume in self.dependency_mounts.items():
            arguments[1:1] = [
                '--mount',
                f'type=volume,source={volume},target={target},readonly,volume-nocopy',
            ]
        for target, options in self.dependency_caches.items():
            arguments[1:1] = ['--tmpfs', target + ':' + options]
        for target, source in self.project_masks.items():
            arguments[1:1] = [
                '--mount', f'type=bind,source={source},target={target},readonly',
            ]
        arguments = self.lifetime().wrap(arguments, self.image)
        result = self._docker(
            arguments, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        (self.artifact_root / "container-start.stdout.log").write_text(
            result.stdout, encoding="utf-8"
        )
        (self.artifact_root / "container-start.stderr.log").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode != 0:
            raise RuntimeError(f"无法启动任务容器：{result.stderr.strip()}")
        container_id = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            raise RuntimeError("Docker 未返回有效容器 ID。")
        self._container_id = container_id
        self._register_control()
        from agent.storage.resources import owner_cancelled

        if self.resource_owner and owner_cancelled(self.resource_owner, self.resource_root):
            self._kill_now()
            self._check_cancelled()
        (self.artifact_root / "container.json").write_text(
            json.dumps(
                {
                    "container_id": container_id,
                    "container_name": self.container_name,
                    "image": self.image,
                    "repo_root": str(self.repo_root),
                    "artifact_root": str(self.artifact_root),
                    "network": "none",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self

    def host_to_container(self, path):
        resolved = Path(path).expanduser().resolve()
        if resolved.is_relative_to(self.repo_root):
            relative = resolved.relative_to(self.repo_root)
            if relative != Path('.'):
                self.project_scope.require(relative.as_posix())
        for index, source in enumerate(self.material_mounts):
            root = Path(source)
            if resolved == root:
                return f'/agent-materials/{index}'
            if root.is_dir() and resolved.is_relative_to(root):
                return f'/agent-materials/{index}/{resolved.relative_to(root).as_posix()}'
        for root, target in (
            (self.artifact_root, "/agent-artifacts"),
            (self.repo_root, "/testbed"),
        ):
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            return target if relative == Path(".") else f"{target}/{relative.as_posix()}"
        raise ValueError(f"路径不在任务仓库或记录目录中：{resolved}")

    def _activated_argv(self, command, shell):
        activation = (
            'source /opt/miniconda3/bin/activate ' + shlex.quote(self.runtime['conda_env']) + ' && '
            if self.runtime['conda_env']
            else ''
        )
        if shell:
            if not isinstance(command, str) or not command.strip():
                raise ValueError("shell=True 时 command 必须是非空文本。")
            return [
                "/bin/bash",
                "-c",
                activation + 'exec /bin/bash -e -o pipefail -c "$1"',
                "agent-command",
                command,
            ]
        if (
            not isinstance(command, (list, tuple))
            or not command
            or any(not isinstance(arg, str) for arg in command)
        ):
            raise ValueError("shell=False 时 command 必须是非空字符串参数列表。")
        return ["/bin/bash", "-c", activation + 'exec "$@"', "agent-command", *command]

    def _cleanup_process_group(self, status_path):
        result = self._docker(
            [
                "exec",
                self._container_id,
                self.python_executable,
                "-c",
                _CLEANUP_RUNNER,
                self.host_to_container(status_path),
            ],
            timeout=10,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0

    # ============================================================
    # 执行进程并保存完整结果
    # ============================================================
    def run_process(
        self,
        command,
        *,
        stdout_path,
        stderr_path,
        timeout=None,
        shell=False,
        on_started=None,
        isolated_launcher=False,
    ):
        timeout = self.timeout if timeout is None else timeout
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("命令超时必须大于 0。")
        argv = self._activated_argv(command, shell)
        stdout_path, stderr_path = Path(stdout_path), Path(stderr_path)
        # 日志和协议文件必须位于本次运行挂载的证据目录。
        for path in (stdout_path, stderr_path):
            path.resolve().relative_to(self.artifact_root)
            path.parent.mkdir(parents=True, exist_ok=True)
        self.start()
        if on_started:
            on_started(self._container_id)
        status_path = self.artifact_root / f"process-{uuid.uuid4().hex}.json"
        request = {
            "argv": argv,
            "timeout": timeout,
            "status_path": self.host_to_container(status_path),
            "stdout_path": self.host_to_container(stdout_path),
            "stderr_path": self.host_to_container(stderr_path),
        }
        request["request_digest"] = digest(request)
        save(stdout_path.with_suffix(".intent.json"), {
            "command": command, "shell": shell, "container_id": self._container_id,
            "process_result": str(status_path), "request_digest": request["request_digest"],
        })
        payload = base64.b64encode(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        result = {
            "exit_code": None,
            "timed_out": False,
            "stdout_log": str(stdout_path.resolve()),
            "stderr_log": str(stderr_path.resolve()),
            "process_result": str(status_path),
            'container_id': self._container_id,
        }
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                completed = self._docker(
                    [
                        "exec",
                        self._container_id,
                        self.python_executable,
                        *(['-I'] if isolated_launcher else []),
                        "-c",
                        _PROCESS_RUNNER,
                        payload.decode("ascii"),
                    ],
                    timeout=timeout + 15,
                    stdout=stdout,
                    stderr=stderr,
                )
                result["exit_code"] = completed.returncode
            except subprocess.TimeoutExpired:
                # Docker 通信卡住时，也不能让任务继续在后台运行。
                try:
                    cleaned = self._cleanup_process_group(status_path)
                except (OSError, subprocess.SubprocessError):
                    cleaned = False
                result.update(
                    exit_code=124,
                    timed_out=True,
                    error="Docker 执行连接超时。",
                    process_cleanup_confirmed=cleaned,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                result.update(error=f"{type(exc).__name__}: {exc}")
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not result.get("error"):
                if not isinstance(status.get("exit_code"), int):
                    raise ValueError("容器命令没有完成记录。")
                result.update(
                    {
                        key: status[key]
                        for key in ("exit_code", "timed_out", "error")
                        if key in status
                    }
                )
        except (OSError, ValueError, AttributeError) as exc:
            result.setdefault("error", f"没有有效的容器执行结果：{exc}")
            if not result["timed_out"]:
                result["exit_code"] = None
        save(self.artifact_root / f"{status_path.stem}.host.json", result)
        self._execution_unknown |= bool(result.get('error') or result.get('exit_code') is None)
        return result

    @staticmethod
    def _summary(path):
        return log_excerpt(path, notice="[前文已截断；完整内容见日志]\n")

    def execute(self, command):
        stem = f"command-{uuid.uuid4().hex}"
        stdout_path = self.artifact_root / f"{stem}.stdout.log"
        stderr_path = self.artifact_root / f"{stem}.stderr.log"
        result = self.run_process(
            command, stdout_path=stdout_path, stderr_path=stderr_path, shell=True
        )
        result.update(
            command=command, stdout=self._summary(stdout_path), stderr=self._summary(stderr_path)
        )
        if result.get("error"):
            result["stderr"] += f"\n{result['error']}"
        if result["timed_out"]:
            result["stderr"] += f"\n命令执行超过 {self.timeout} 秒。"
        return result

    # ============================================================
    # 停止容器并保留现场
    # ============================================================
    def close(self):
        """只停止本实例管理的容器，保留容器和全部本机记录。"""
        if self._lifetime is not None:
            self._lifetime.revoke()
        if self._closed:
            return
        if self._container_id is not None:
            result = self._docker(
                ["stop", "--time", "5", self._container_id],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            (self.artifact_root / "container-stop.stdout.log").write_text(
                result.stdout, encoding="utf-8"
            )
            (self.artifact_root / "container-stop.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
            if result.returncode != 0:
                raise RuntimeError(f"无法停止任务容器：{result.stderr.strip()}")
            save(self.artifact_root / 'container-stopped.json', self._confirm_stopped())
        self._closed = True
        from agent.storage.resources import owner_cancelled, unregister_active_container

        if self.disposable and not self._execution_unknown and not (
            self.resource_owner and owner_cancelled(self.resource_owner, self.resource_root)
        ) and self._container_id is not None:
            if self.resource_owner:
                self._finish_resources(self.resource_owner,
                                       ['container:' + self.container_name], completed=True)
        if self._control_key:
            unregister_active_container(self._control_key)
            self._control_key = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
