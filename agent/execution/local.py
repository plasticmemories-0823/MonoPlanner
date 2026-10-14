from pathlib import Path
import sys
import tempfile
import uuid

from agent.execution.process import run_process, log_excerpt


class LocalEnvironment:

    def __init__(
        self,
        repo_root: str,
        timeout: int = 120,
        artifact_root=None,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()

        if not self.repo_root.exists():
            raise FileNotFoundError(f"工作区不存在：{self.repo_root}")

        if not self.repo_root.is_dir():
            raise NotADirectoryError(f"工作区必须是目录：{self.repo_root}")

        self.timeout = timeout
        self.artifact_root = (
            Path(artifact_root)
            if artifact_root
            else Path(tempfile.mkdtemp(prefix="coding-agent-local-"))
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.python_executable = sys.executable

    @staticmethod
    def host_to_container(path):
        """本地后端直接使用本机路径作为执行路径。"""
        return str(path)

    def run_process(self, command, *, stdout_path, stderr_path, timeout=None, shell=False):
        return run_process(
            command,
            cwd=self.repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=self.timeout if timeout is None else timeout,
            shell=shell,
        )

    def execute(
        self,
        command: str,
    ) -> dict:

        tag = "command-" + uuid.uuid4().hex
        stdout, stderr = self.artifact_root / (tag + ".stdout.log"), self.artifact_root / (
            tag + ".stderr.log"
        )
        result = self.run_process(command, stdout_path=stdout, stderr_path=stderr, shell=True)
        result.update(command=command, stdout=log_excerpt(stdout), stderr=log_excerpt(stderr))
        return result
