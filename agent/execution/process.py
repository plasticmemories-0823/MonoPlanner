"""Local process execution shared by commands and acceptance tests."""

import os
from pathlib import Path
import signal
import subprocess


def shell_arguments(command):
    if not isinstance(command, str) or not command.strip():
        raise ValueError("shell 命令必须是非空文本")
    if os.name != "nt":
        return ["/bin/bash", "-e", "-o", "pipefail", "-c", command], False
    # cmd.exe has no pipefail. Reject top-level pipelines rather than report
    # the final consumer's exit code as the whole pipeline's test result.
    quoted, escaped = False, False
    for char in command:
        if escaped:
            escaped = False
        elif char == "^" and not quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == "|" and not quoted:
            raise ValueError("Windows 命令不支持可靠判定顶层管道结果，请拆成独立命令并检查每次退出码。")
    return command, True


def stop_process(process):
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def run_process(command, *, cwd, stdout_path, stderr_path, timeout, shell=False, env=None, on_started=None):
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if shell:
        command, shell = shell_arguments(command)
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
               if os.name == "nt" else {"start_new_session": True})
    result = {"exit_code": None, "timed_out": False,
              "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
    for path in (stdout_path, stderr_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(stdout_path).open("wb") as out, Path(stderr_path).open("wb") as err:
        process = subprocess.Popen(command, cwd=str(cwd), shell=shell, stdout=out, stderr=err, env=env, **options)
        try:
            if on_started:
                on_started(process.pid)
            result["exit_code"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process(process)
            result.update(exit_code=124, timed_out=True)
        except BaseException:
            stop_process(process)
            raise
    return result


def log_excerpt(path, limit=6000, *, notice="[前文已节选，完整输出见日志]\n"):
    path = Path(path)
    with path.open("rb") as stream:
        size = path.stat().st_size
        stream.seek(max(0, size - limit))
        value = stream.read().decode("utf-8", errors="replace")
    return (notice if size > limit else "") + value
