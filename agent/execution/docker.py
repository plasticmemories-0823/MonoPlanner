"""Run a task checkout and its acceptance commands in the same Docker container.

The model and file-navigation tools stay on the host. Only the task checkout
and run artifacts are mounted; credentials and the Docker socket are not.
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

PROXY_VARIABLES = tuple(name for protocol in ('HTTP', 'HTTPS', 'FTP', 'ALL', 'NO')
                        for name in (protocol + '_PROXY', protocol.lower() + '_proxy'))
DEPENDENCY_CACHE_OPTIONS = 'rw,exec,nosuid,nodev,size=67108864,mode=1777'


def proxy_environment(proxy=None):
    """Override Docker CLI's implicit credential-bearing proxy injection."""
    result = []
    for name in PROXY_VARIABLES:
        value = proxy if proxy and name.lower() in {'http_proxy', 'https_proxy'} else ''
        result.extend(['--env', name + '=' + value])
    return result


def managed_dependencies(image, managed, *, root=None):
    """Only program-owned image paths may become project dependency mounts."""
    rows = (managed or {}).get('node_modules', [])
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('Invalid managed dependency list')
    dependencies, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'relative', 'image_path'}:
            raise ValueError('Invalid managed dependency descriptor')
        relative = row['relative']
        if (not isinstance(relative, str) or not relative or '\\' in relative
                or any(c in relative for c in '\0\r\n,:')
                or PurePosixPath(relative).is_absolute()
                or str(PurePosixPath(relative)) != relative
                or any(p in {'..', '.git', '.venv', 'node_modules'} or p.startswith('.env')
                       for p in PurePosixPath(relative).parts)):
            raise ValueError('Unsafe managed dependency directory')
        expected = '/opt/monoplanner/node/' + (relative + '/' if relative != '.' else '') + 'node_modules'
        if row['image_path'] != expected or relative in seen:
            raise ValueError('Managed dependency image path differs from project directory')
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Managed dependencies require an immutable image ID')
        seen.add(relative)
        from agent.storage.preferences import state_root
        owner = str(Path(root).resolve() if root is not None else state_root())
        identity = hashlib.sha256((owner + '\n' + image + '\n' + expected).encode('utf-8')).hexdigest()
        dependencies.append({'relative': relative, 'image_path': expected,
            'identity': identity, 'volume': 'monoplanner-node-' + identity,
            'target': '/testbed/' + (relative + '/' if relative != '.' else '') + 'node_modules'})
    return dependencies


def dependency_mountpoint(repo, relative):
    """Create empty mount directories, never replace or populate host packages."""
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
    """Check future-task cache eligibility without changing old volume records.

    A first execution may seed a new volume. Previously attempted volumes must
    have a complete journal and matching live identity; otherwise preparation
    builds a new image generation instead of repairing a bound task in place.
    """
    from agent.execution.managed import _lock
    from agent.source.snapshots import _link
    try:
        dependencies = managed_dependencies(image, managed, root=root)
    except (TypeError, ValueError):
        return False
    root = Path(root)
    for item in dependencies:
        with _lock(root/'managed-volume-locks'/(item['identity'] + '.lock'), cancelled):
            folder = root/'managed-volumes'/item['identity']
            record = folder/'completion.json'
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
            labels = {'com.monoplanner.dependency': item['identity'], 'com.monoplanner.image': image}
            if (stored != expected or not isinstance(actual, list) or len(actual) != 1
                    or actual[0].get('Name') != item['volume'] or actual[0].get('Driver') != 'local'
                    or actual[0].get('Options') or actual[0].get('Labels') != labels):
                return False
    return True


def docker_executable():
    # Portable startup intentionally omits system language runtimes from PATH.
    if shutil.which('docker'):
        return 'docker'
    installed = Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Docker/Docker/resources/bin/docker.exe'
    return str(installed) if os.name == 'nt' and installed.is_file() else 'docker'


def runtime_profile(value=None):
    value = dict(value or {'python': '/opt/miniconda3/envs/testbed/bin/python', 'conda_env': 'testbed'})
    if set(value) != {'python', 'conda_env'}:
        raise ValueError('运行环境仅接受 python 和 conda_env')
    python, conda = value['python'], value['conda_env']
    if (not isinstance(python, str) or not python.startswith('/') or '\\' in python
            or '..' in python.split('/') or any(c in python for c in '\0\r\n')):
        raise ValueError('python 必须是容器内的绝对路径')
    if not isinstance(conda, str) or (conda and not re.fullmatch(r'[A-Za-z0-9_.-]+', conda)):
        raise ValueError('conda_env 必须是环境名称或空字符串')
    return value


# Keep this script compatible with the older Python versions in benchmark images.
# The timeout lives inside the container, so killing the Docker CLI is not relied
# on to stop the actual command or its children.
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
result = {"exit_code": None, "timed_out": False, "pid": None}

def save():
    temporary = status_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False)
    os.replace(temporary, status_path)

try:
    # POSIX accepts byte arguments. Avoid Python 3.6's ASCII locale encoding
    # before the child even starts; keep the benchmark interpreter and locale.
    argv = [arg.encode("utf-8") for arg in request["argv"]]
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(argv, cwd="/testbed", env=child_env, start_new_session=True)
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


class DockerEnvironment:
    """One owned container per task; stopping it preserves host run evidence."""

    python_executable = "/opt/miniconda3/envs/testbed/bin/python"

    def __init__(self, repo_root, image, container_name, timeout=120,
                 artifact_root=None, *, runtime=None, material_mounts=None, managed=None):
        self.repo_root = Path(repo_root).expanduser().resolve()
        if not self.repo_root.exists():
            raise FileNotFoundError(f"工作区不存在：{self.repo_root}")
        if not self.repo_root.is_dir():
            raise NotADirectoryError(f"工作区必须是目录：{self.repo_root}")
        if not isinstance(image, str) or not image.strip() or image.startswith("-"):
            raise ValueError("Docker image 必须是非空镜像名称。")
        if not isinstance(container_name, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", container_name):
            raise ValueError("Docker container_name 格式无效。")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("命令超时必须大于 0。")
        self.image = image
        self.runtime = runtime_profile(runtime)
        self.python_executable = self.runtime['python']
        self.material_mounts = dict(material_mounts or {})
        self.dependencies = managed_dependencies(image, managed)
        self.dependency_mounts = {item['target']: item['volume'] for item in self.dependencies}
        self.dependency_caches = {target + '/' + name: DEPENDENCY_CACHE_OPTIONS
                                 for target in self.dependency_mounts
                                 for name in ('.cache', '.vite', '.vite-temp')}
        self._dependencies_checked = False
        self.runtime_facts = {'backend': 'Docker', 'python_executable': self.python_executable,
                              'conda_env': self.runtime['conda_env'], 'source': '/testbed', 'network': 'none'}
        self.container_name = container_name
        self.timeout = timeout
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve() if artifact_root is not None
            else Path(tempfile.mkdtemp(prefix="coding-agent-docker-"))
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        # --mount uses commas as separators. Reject rather than reinterpret a path.
        if any("," in str(path) for path in (self.repo_root, self.artifact_root)):
            raise ValueError("Docker 挂载路径暂不支持逗号。")
        self._container_id = None
        self._closed = False

    def clone_options(self):
        options = {'runtime': self.runtime, 'material_mounts': self.material_mounts}
        if self.dependencies:
            options['managed'] = {'node_modules': [{k: item[k] for k in ('relative', 'image_path')}
                                                   for item in self.dependencies]}
        return options

    @staticmethod
    def _process_options():
        return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    def _docker(self, arguments, timeout=30, **kwargs):
        return subprocess.run(
            [docker_executable(), *arguments], timeout=timeout,
            **self._process_options(), **kwargs,
        )

    def _dependency_query(self, arguments, *, timeout=30):
        result = self._docker(arguments, timeout=timeout, capture_output=True,
                              text=True, encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError('项目依赖准备失败；保留记录，不自动重试：' + result.stderr.strip()[:500])
        return result.stdout

    def _verify_dependency_volume(self, item, labels):
        actual = json.loads(self._dependency_query(['volume', 'inspect', item['volume']]))
        if (len(actual) != 1 or actual[0].get('Name') != item['volume']
                or actual[0].get('Driver') != 'local' or actual[0].get('Options')
                or actual[0].get('Labels') != labels):
            raise ValueError('项目依赖副本身份或存储方式已改变。')

    def _prepare_dependencies(self, *, allow_create=True):
        if self._dependencies_checked:
            return
        if not self.dependencies:
            self._dependencies_checked = True
            return
        from agent.execution.managed import _lock
        from agent.storage.preferences import state_root
        for item in self.dependencies:
            with _lock(state_root()/'managed-volume-locks'/(item['identity'] + '.lock')):
                self._prepare_dependency(item, allow_create=allow_create)
        self._dependencies_checked = True

    def _prepare_dependency(self, item, *, allow_create):
        from agent.storage.preferences import state_root, atomic_write
        # Validate/create mount points even for readonly investigation copies.
        # These are empty generated directories; package bytes stay in Docker.
        dependency_mountpoint(self.repo_root, item['relative'])
        folder = state_root()/'managed-volumes'/item['identity']
        record = folder/'completion.json'
        expected = {'image': self.image, **item}
        labels = {'com.monoplanner.dependency': item['identity'],
                  'com.monoplanner.image': self.image}
        if record.exists():
            stored = json.loads(record.read_text(encoding='utf-8'))
            if stored != {'status': 'completed', **expected}:
                raise RuntimeError('项目依赖副本准备结果不明；不会自动重复准备。')
        else:
            if not allow_create:
                raise RuntimeError('已绑定任务缺少项目依赖副本记录；不会自动重建。')
            # Exclusive reservation also prevents two tasks from seeding one
            # volume concurrently. An abandoned reservation is not replayed.
            try:
                folder.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise RuntimeError('项目依赖正在准备或记录不完整；请稍后重试读取状态。') from exc
            atomic_write(record, json.dumps({'status': 'pending', **expected}).encode('utf-8'))
            existing = self._docker(['volume', 'inspect', item['volume']], capture_output=True,
                text=True, encoding='utf-8', errors='replace')
            if not existing.returncode:
                raise RuntimeError('发现没有完成记录的项目依赖副本；不会自动覆盖。')
            self._dependency_query(['volume', 'create', '--label',
                'com.monoplanner.dependency=' + item['identity'], '--label',
                'com.monoplanner.image=' + self.image, item['volume']])
            self._verify_dependency_volume(item, labels)
            # No project, record, credential or socket is mounted here.
            # cp preserves package symlinks, including workspace references;
            # the task mounts the final package directory read-only.
            name = 'monoplanner-seed-' + uuid.uuid4().hex
            arguments = ['run', '--rm', '--pull', 'never', '--name', name,
                '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                '--security-opt', 'no-new-privileges', '--pids-limit', '64',
                '--cpus', '2', '--memory', '512m', '--user', '0:0',
                *proxy_environment(),
                '--mount', 'type=volume,source=' + item['volume'] + ',target=/prepared,volume-nocopy',
                '--entrypoint', '/bin/cp', self.image, '-R', '--no-preserve=ownership',
                item['image_path'] + '/.', '/prepared/']
            try:
                self._dependency_query(arguments, timeout=120)
            except BaseException:
                # This random name belongs only to this preparation attempt.
                # Removing the seed container cannot remove its retained volume.
                try:
                    self._docker(['rm', '--force', name], timeout=15, capture_output=True,
                                 text=True, encoding='utf-8', errors='replace')
                except (OSError, subprocess.SubprocessError):
                    pass
                raise
            atomic_write(record, json.dumps({'status': 'completed', **expected}).encode('utf-8'))
        self._verify_dependency_volume(item, labels)

    def start(self):
        if self._closed:
            raise RuntimeError("此环境已关闭，请为新任务创建新环境。")
        if self._container_id is not None:
            return self
        self._prepare_dependencies()
        arguments = [
            "run", "--detach", "--pull", "never", "--name", self.container_name,
            "--network", "none", "--init", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "256",
            "--cpus", "2", "--memory", "3g",
            *proxy_environment(),
            "--mount", f"type=bind,source={self.repo_root},target=/testbed" + (",readonly" if getattr(self, "repo_read_only", False) else ""),
            "--mount", f"type=bind,source={self.artifact_root},target=/agent-artifacts",
            "--workdir", "/testbed", "--entrypoint", "/bin/bash", self.image,
            "-c", "exec sleep infinity",
        ]
        if getattr(self, "root_read_only", False):
            # Docker defaults tmpfs to noexec. Compiler probes must be able to
            # run their temporary products inside this isolated filesystem.
            arguments[1:1] = ["--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=268435456"]
        for index, source in enumerate(self.material_mounts.values()):
            if ',' in str(source):
                raise ValueError('资料挂载路径暂不支持逗号')
            arguments[1:1] = ['--mount', f'type=bind,source={source},target=/agent-materials/{index},readonly']
        for target, volume in self.dependency_mounts.items():
            arguments[1:1] = ['--mount', f'type=volume,source={volume},target={target},readonly,volume-nocopy']
        for target, options in self.dependency_caches.items():
            arguments[1:1] = ['--tmpfs', target + ':' + options]
        result = self._docker(arguments, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        (self.artifact_root / "container-start.stdout.log").write_text(
            result.stdout, encoding="utf-8")
        (self.artifact_root / "container-start.stderr.log").write_text(
            result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"无法启动任务容器：{result.stderr.strip()}")
        container_id = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            raise RuntimeError("Docker 未返回有效容器 ID。")
        self._container_id = container_id
        (self.artifact_root / "container.json").write_text(json.dumps({
            "container_id": container_id, "container_name": self.container_name,
            "image": self.image, "repo_root": str(self.repo_root),
            "artifact_root": str(self.artifact_root), "network": "none",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return self

    def host_to_container(self, path):
        resolved = Path(path).expanduser().resolve()
        for index, source in enumerate(self.material_mounts):
            root = Path(source)
            if resolved == root:
                return f'/agent-materials/{index}'
            if root.is_dir() and resolved.is_relative_to(root):
                return f'/agent-materials/{index}/{resolved.relative_to(root).as_posix()}'
        for root, target in ((self.artifact_root, "/agent-artifacts"),
                             (self.repo_root, "/testbed")):
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            return target if relative == Path(".") else f"{target}/{relative.as_posix()}"
        raise ValueError(f"路径不在任务仓库或记录目录中：{resolved}")

    def _activated_argv(self, command, shell):
        activation = ('source /opt/miniconda3/bin/activate ' + shlex.quote(self.runtime['conda_env']) + ' && '
                      if self.runtime['conda_env'] else '')
        if shell:
            if not isinstance(command, str) or not command.strip():
                raise ValueError("shell=True 时 command 必须是非空文本。")
            return ["/bin/bash", "-c",
                    activation + 'exec /bin/bash -e -o pipefail -c "$1"',
                    "agent-command", command]
        if (not isinstance(command, (list, tuple)) or not command
                or any(not isinstance(arg, str) for arg in command)):
            raise ValueError("shell=False 时 command 必须是非空字符串参数列表。")
        return ["/bin/bash", "-c",
                activation + 'exec "$@"',
                "agent-command", *command]

    def _cleanup_process_group(self, status_path):
        result = self._docker([
            "exec", self._container_id, self.python_executable,
            "-c", _CLEANUP_RUNNER, self.host_to_container(status_path),
        ], timeout=10, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return result.returncode == 0

    def run_process(self, command, *, stdout_path, stderr_path, timeout=None,
                    shell=False, on_started=None, isolated_launcher=False):
        timeout = self.timeout if timeout is None else timeout
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("命令超时必须大于 0。")
        argv = self._activated_argv(command, shell)
        stdout_path, stderr_path = Path(stdout_path), Path(stderr_path)
        # Logs and protocol files must belong to this run's mounted evidence.
        for path in (stdout_path, stderr_path):
            path.resolve().relative_to(self.artifact_root)
            path.parent.mkdir(parents=True, exist_ok=True)
        self.start()
        if on_started:
            on_started(self._container_id)
        status_path = self.artifact_root / f"process-{uuid.uuid4().hex}.json"
        request = {"argv": argv, "timeout": timeout,
                   "status_path": self.host_to_container(status_path)}
        payload = base64.b64encode(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        result = {
            "exit_code": None, "timed_out": False,
            "stdout_log": str(stdout_path.resolve()),
            "stderr_log": str(stderr_path.resolve()),
            "process_result": str(status_path),
            'container_id': self._container_id,
        }
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                completed = self._docker([
                    "exec", self._container_id, self.python_executable,
                    *(['-I'] if isolated_launcher else []),
                    "-c", _PROCESS_RUNNER, payload.decode("ascii"),
                ], timeout=timeout + 15, stdout=stdout, stderr=stderr)
                result["exit_code"] = completed.returncode
            except subprocess.TimeoutExpired:
                # A stalled Docker transport must not leave the task running.
                try:
                    cleaned = self._cleanup_process_group(status_path)
                except (OSError, subprocess.SubprocessError):
                    cleaned = False
                result.update(exit_code=124, timed_out=True,
                              error="Docker 执行连接超时。", process_cleanup_confirmed=cleaned)
            except (OSError, subprocess.SubprocessError) as exc:
                result.update(error=f"{type(exc).__name__}: {exc}")
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not result.get("error"):
                if not isinstance(status.get("exit_code"), int):
                    raise ValueError("容器命令没有完成记录。")
                result.update({key: status[key] for key in ("exit_code", "timed_out", "error")
                               if key in status})
        except (OSError, ValueError, AttributeError) as exc:
            result.setdefault("error", f"没有有效的容器执行结果：{exc}")
            if not result["timed_out"]:
                result["exit_code"] = None
        (self.artifact_root / f"{status_path.stem}.host.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _summary(path):
        return log_excerpt(path, notice="[前文已截断；完整内容见日志]\n")

    def execute(self, command):
        stem = f"command-{uuid.uuid4().hex}"
        stdout_path = self.artifact_root / f"{stem}.stdout.log"
        stderr_path = self.artifact_root / f"{stem}.stderr.log"
        result = self.run_process(command, stdout_path=stdout_path,
                                  stderr_path=stderr_path, shell=True)
        result.update(command=command, stdout=self._summary(stdout_path),
                      stderr=self._summary(stderr_path))
        if result.get("error"):
            result["stderr"] += f"\n{result['error']}"
        if result["timed_out"]:
            result["stderr"] += f"\n命令执行超过 {self.timeout} 秒。"
        return result

    def close(self):
        """Stop only this instance's container; retain it and all host artifacts."""
        if self._closed:
            return
        if self._container_id is not None:
            result = self._docker(["stop", "--time", "5", self._container_id],
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            (self.artifact_root / "container-stop.stdout.log").write_text(
                result.stdout, encoding="utf-8")
            (self.artifact_root / "container-stop.stderr.log").write_text(
                result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"无法停止任务容器：{result.stderr.strip()}")
        self._closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
