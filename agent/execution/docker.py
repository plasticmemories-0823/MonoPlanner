"""Run a task checkout and its acceptance commands in the same Docker container.

The model and file-navigation tools stay on the host. Only the task checkout
and run artifacts are mounted; credentials and the Docker socket are not.
"""

import base64
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid


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
                 artifact_root=None):
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

    @staticmethod
    def _process_options():
        return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    def _docker(self, arguments, timeout=30, **kwargs):
        return subprocess.run(
            ["docker", *arguments], timeout=timeout,
            **self._process_options(), **kwargs,
        )

    def start(self):
        if self._closed:
            raise RuntimeError("此环境已关闭，请为新任务创建新环境。")
        if self._container_id is not None:
            return self
        arguments = [
            "run", "--detach", "--pull", "never", "--name", self.container_name,
            "--network", "none", "--init", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "256",
            "--cpus", "2", "--memory", "3g",
            "--mount", f"type=bind,source={self.repo_root},target=/testbed" + (",readonly" if getattr(self, "repo_read_only", False) else ""),
            "--mount", f"type=bind,source={self.artifact_root},target=/agent-artifacts",
            "--workdir", "/testbed", "--entrypoint", "/bin/bash", self.image,
            "-c", "exec sleep infinity",
        ]
        if getattr(self, "root_read_only", False):
            arguments[1:1] = ["--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=268435456"]
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
        for root, target in ((self.artifact_root, "/agent-artifacts"),
                             (self.repo_root, "/testbed")):
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            return target if relative == Path(".") else f"{target}/{relative.as_posix()}"
        raise ValueError(f"路径不在任务仓库或记录目录中：{resolved}")

    @staticmethod
    def _activated_argv(command, shell):
        if shell:
            if not isinstance(command, str) or not command.strip():
                raise ValueError("shell=True 时 command 必须是非空文本。")
            return ["/bin/bash", "-c",
                    'source /opt/miniconda3/bin/activate testbed && exec /bin/bash -e -o pipefail -c "$1"',
                    "agent-command", command]
        if (not isinstance(command, (list, tuple)) or not command
                or any(not isinstance(arg, str) for arg in command)):
            raise ValueError("shell=False 时 command 必须是非空字符串参数列表。")
        return ["/bin/bash", "-c",
                'source /opt/miniconda3/bin/activate testbed && exec "$@"',
                "agent-command", *command]

    def _cleanup_process_group(self, status_path):
        result = self._docker([
            "exec", self._container_id, self.python_executable,
            "-c", _CLEANUP_RUNNER, self.host_to_container(status_path),
        ], timeout=10, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return result.returncode == 0

    def run_process(self, command, *, stdout_path, stderr_path, timeout=None,
                    shell=False):
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
        status_path = self.artifact_root / f"process-{uuid.uuid4().hex}.json"
        request = {"argv": argv, "timeout": timeout,
                   "status_path": self.host_to_container(status_path)}
        payload = base64.b64encode(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        result = {
            "exit_code": None, "timed_out": False,
            "stdout_log": str(stdout_path.resolve()),
            "stderr_log": str(stderr_path.resolve()),
            "process_result": str(status_path),
        }
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                completed = self._docker([
                    "exec", self._container_id, self.python_executable,
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
        path = Path(path)
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 6000))
            tail = stream.read().decode("utf-8", errors="replace")
        return ("[前文已截断；完整内容见日志]\n" if size > 6000 else "") + tail

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
