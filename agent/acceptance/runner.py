"""运行冻结的验收规则并保存证据；本地执行器不提供安全沙箱。"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from agent.acceptance import results
from agent.acceptance.results import _read_test_result as read_test_result
from agent.acceptance.contracts import VerificationSpec, tests_ready_for_candidate
from agent.acceptance.drivers import _BOOTSTRAP, _REPOSITORY_BOOTSTRAP
from agent.acceptance.execution import validate_execution_result
from agent.execution.process import log_excerpt
from agent.settings import PROJECT_ROOT
from agent.model.errors import ModelRequestNotSent


# ============================================================
# 验收原件与运行环境
# ============================================================
class AcceptanceRun:
    """同一对象的基线和各轮运行使用同一份验收标准。"""

    def __init__(
        self,
        repo_root: Path,
        spec: VerificationSpec,
        artifacts_root: Path | None = None,
        timeout: int = 120,
        environment=None,
        repository_baseline=None,
    ):
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"验收工作区不存在：{self.repo_root}")
        if timeout <= 0:
            raise ValueError("验收超时必须大于 0。")
        if environment is None or not callable(getattr(environment, 'run_process', None)):
            raise ValueError('An explicit execution backend is required; host fallback is disabled')
        self.spec = spec
        self.version = hashlib.sha256(
            json.dumps(
                {
                    "criteria": spec.criteria,
                    "command": spec.command,
                    "test_code": spec.test_code,
                    "test_driver": spec.test_driver,
                    "execution_checks": spec.execution_checks,
                    "requirement_digest": spec.requirement_digest,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self._test_ids = None
        self.timeout = timeout
        self.environment = environment
        self.repository_baseline = repository_baseline
        if artifacts_root is None:
            base = Path(os.getenv("LOCALAPPDATA") or Path.home() / ".local" / "state")
            artifacts_root = base / "langgraph-simple-agent" / "runs"
        artifacts_root = Path(artifacts_root).resolve()
        artifacts_root.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = Path(tempfile.mkdtemp(prefix="acceptance-", dir=artifacts_root))
        self.test_path = self.artifact_dir / "test_acceptance.py"
        self.bootstrap_path = self.artifact_dir / "acceptance_runner.py"
        self.plan_path = self.artifact_dir / "plan.json"
        plan = {
            "repo_root": str(self.repo_root),
            "criteria": list(spec.criteria),
            "command": spec.command,
            "test_file": self.test_path.name if spec.test_code is not None else None,
            "test_driver": spec.test_driver,
            "requirement_digest": spec.requirement_digest,
        }
        self.plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        paths = [self.plan_path]
        if spec.execution_checks:
            self.execution_path = self.artifact_dir / "execution-checks.json"
            self.execution_runner = self.artifact_dir / "execution_runner.py"
            self.execution_path.write_text(
                json.dumps(spec.execution_checks, ensure_ascii=False), encoding="utf-8"
            )
            self.execution_runner.write_bytes(
                (PROJECT_ROOT / "resources/drivers/execution.py").read_bytes()
            )
            paths.extend((self.execution_path, self.execution_runner))
        if spec.test_code is not None:
            self.test_path.write_text(spec.test_code, encoding="utf-8")
            # 源码文件使用 UTF-8，即使任务运行于 ASCII 区域设置下的 Python 3.6。
            # 不要通过进程参数直接传递含 Unicode 的程序正文。
            self.bootstrap_path.write_text(_BOOTSTRAP, encoding="utf-8")
            paths.extend((self.test_path, self.bootstrap_path))
            if spec.test_driver:
                self.repository_runner = self.artifact_dir / "repository_runner.py"
                self.repository_runner.write_text(_REPOSITORY_BOOTSTRAP, encoding="utf-8")
                paths.append(self.repository_runner)
        self._hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in paths}

    def _unchanged(self):
        try:
            return all(
                hashlib.sha256(path.read_bytes()).digest() == digest
                for path, digest in self._hashes.items()
            )
        except OSError:
            return False

    @staticmethod
    def _summary(path: Path):
        # 完整输出已直接写入文件；只把尾部放回 Agent 上下文。
        return log_excerpt(path, notice="[前文已截断；完整内容见日志]\n")

    # ============================================================
    # 执行冻结验收并保留真实结果
    # ============================================================
    def run(self, phase: str) -> dict:
        started = time.monotonic()
        if not isinstance(phase, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", phase):
            raise ValueError("验收阶段名称只能包含字母、数字、下划线和连字符。")
        if self.spec.execution_checks and (self.artifact_dir / (phase + ".execution.json")).exists():
            raise ModelRequestNotSent(
                "execution_unknown", "execution_checks receipt already exists; preserve it without replay"
            )
        stdout_path = self.artifact_dir / f"{phase}.stdout.log"
        stderr_path = self.artifact_dir / f"{phase}.stderr.log"
        outcome_path = self.artifact_dir / f"{phase}.tests.json"
        report = {
            "status": "error",
            "exit_code": None,
            "reason": "验收未执行。",
            "stdout": "",
            "stderr": "",
            "tests_run": None,
            "failures": None,
            "errors": None,
            "skipped": None,
            "artifact_dir": str(self.artifact_dir),
            "phase": phase,
            "integrity_ok": True,
            "verification_version": self.version,
            "test_ids": None,
            "test_results": None,
            "test_evidence_valid": False,
            "error_kind": None,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        # 同一 phase 可显式重跑，不能把上次的结果误当成本次结果。
        outcome_path.unlink(missing_ok=True)
        # 日志只由执行它的后端打开；预先清空后即关闭，避免 Windows 上重复持有文件。
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        if not self._unchanged():
            report["integrity_ok"] = False
            report["error_kind"] = "integrity_error"
            report["reason"] = "冻结的验收文件被修改或删除，已停止验收。"
        else:
            process_finished = False
            try:
                command = self.spec.command
                backend = self.environment
                map_path = backend.host_to_container
                if command is None:
                    # -B 只阻止写缓存；独立目录同时避免读取仓库中的旧 .pyc。
                    cache_dir = tempfile.mkdtemp(prefix=f"{phase}.cache-", dir=self.artifact_dir)
                    command = [
                        backend.python_executable,
                        "-B",
                        "-X",
                        "utf8",
                        "-X",
                        f"pycache_prefix={map_path(cache_dir)}",
                        map_path(self.bootstrap_path),
                        map_path(self.test_path),
                        map_path(self.repo_root),
                        map_path(outcome_path),
                    ]
                    if self.spec.test_driver:
                        command[6] = map_path(self.repository_runner)
                        command.append(json.dumps(self.spec.test_driver))
                outcome = backend.run_process(
                    command,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    timeout=self.timeout,
                    shell=self.spec.command is not None,
                )
                report["exit_code"] = outcome["exit_code"]
                timed_out = outcome["timed_out"]
                backend_error = outcome.get("error")
                process_finished = True
                if timed_out:
                    report["error_kind"] = "timeout"
                    termination = (
                        "无法确认容器内执行已终止。"
                        if backend and outcome.get("process_cleanup_confirmed") is False
                        else "已终止执行。"
                    )
                    report["reason"] = f"验收超过 {self.timeout} 秒，{termination}"
                elif backend_error:
                    report["error_kind"] = "environment_error"
                    report["reason"] = f"验收环境执行失败：{backend_error}"
                elif report["exit_code"] is None:
                    report["error_kind"] = "environment_error"
                    report["reason"] = "验收执行没有返回退出码。"
                elif self.spec.command is not None:
                    passed = report["exit_code"] == 0
                    report.update(
                        status="passed" if passed else "failed",
                        reason=(
                            "指定验收命令退出码为 0。"
                            if passed
                            else f"指定验收命令退出码为 {report['exit_code']}。"
                        ),
                    )
                else:
                    self._read_test_result(outcome_path, report)
                    if self.spec.test_driver and report["test_evidence_valid"]:
                        self._repository_result(report, phase)
                    if report["test_evidence_valid"] and report["test_ids"]:
                        inventory = tuple(report["test_ids"])
                        if self._test_ids is None:
                            self._test_ids = inventory
                            inventory_path = self.artifact_dir / "test_inventory.json"
                            inventory_path.write_text(
                                json.dumps(inventory, ensure_ascii=False), encoding="utf-8"
                            )
                            self._hashes[inventory_path] = hashlib.sha256(
                                inventory_path.read_bytes()
                            ).digest()
                        elif inventory != self._test_ids:
                            report.update(
                                status="error",
                                test_evidence_valid=False,
                                reason="测试 ID 清单发生变化，不能据此认定新增进展或验收通过。",
                            )
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                report.update(
                    status='error',
                    test_evidence_valid=False,
                    error_kind="result_error" if process_finished else "environment_error",
                    reason=f"无法执行验收：{type(exc).__name__}: {exc}",
                )
            if not self._unchanged():
                report.update(
                    status="error",
                    integrity_ok=False,
                    error_kind="integrity_error",
                    test_evidence_valid=False,
                    reason="验收执行期间冻结文件被修改或删除，结果无效。",
                )
        if self.spec.execution_checks:
            report["execution"] = {"checks": []}
            report["execution_ready"] = False
            remaining = self.timeout - (time.monotonic() - started)
            if (remaining > 0 and report["integrity_ok"] and self._unchanged()
                    and report.get("error_kind") not in {
                        "timeout", "environment_error", "integrity_error", "result_error"
                    }):
                self._run_execution(phase, report, remaining)
            elif report["status"] == "passed":
                report.update(status="error", error_kind="execution_unavailable",
                              reason="Required execution_checks did not complete")
        report["stdout"] = self._summary(stdout_path)
        report["stderr"] = self._summary(stderr_path)
        if report["status"] == "error" and report["error_kind"] is None:
            report["error_kind"] = "test_error" if report["test_evidence_valid"] else "result_error"
        report["tests_ready"] = tests_ready_for_candidate(report)
        if self.spec.test_driver:
            report["test_driver"] = list(self.spec.test_driver)
        if report["status"] == "error" and report["tests_ready"]:
            # 保留 unittest 原始错误条目和计数。这些可运行检查的失败，
            # 应视为候选未通过，不能改称测试未启动或补丁已通过。
            report.update(
                status="failed",
                error_kind=None,
                reason="被测源码执行时抛出异常，未通过固定测试；测试执行证据可用于候选验收。",
            )
        (self.artifact_dir / f"{phase}.result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    def _run_execution(self, phase, report, remaining):
        """与已有测试共用总时限和后端；执行声明不能被测试代码替换。"""
        output = self.artifact_dir / (phase + ".execution.json")
        stdout = self.artifact_dir / (phase + ".execution.stdout.log")
        stderr = self.artifact_dir / (phase + ".execution.stderr.log")
        if output.exists():
            raise ModelRequestNotSent(
                "execution_unknown", "execution_checks output appeared before dispatch; preserve it"
            )
        backend = self.environment
        mapped = backend.host_to_container
        argv = [backend.python_executable, "-B", "-X", "utf8", mapped(self.execution_runner),
                mapped(self.execution_path), mapped(self.repo_root), mapped(output), str(remaining)]
        try:
            outcome = backend.run_process(
                argv, stdout_path=stdout, stderr_path=stderr, timeout=remaining, shell=False,
            )
            report["execution_process"] = outcome
            if outcome.get("timed_out") or outcome.get("error") or outcome.get("exit_code") != 0:
                raise RuntimeError("execution_checks process did not return a complete successful receipt")
            execution = validate_execution_result(output, self.spec.execution_checks,
                                                  repository=self.repo_root, map_path=mapped)
            report["execution"] = execution
            rows = execution["checks"]
            if any(row.get("error_kind") == "timeout" for row in rows):
                raise RuntimeError("execution_checks subprocess timeout; inspect preserved processes")
            report["execution_ready"] = all(
                row["status"] in {"passed", "failed"}
                or (row["status"] == "error" and row.get("error_kind") == "source")
                for row in rows
            )
            if any(row["status"] != "passed" for row in rows):
                report.update(
                    status="failed" if report["execution_ready"] else "error",
                    error_kind=None if report["execution_ready"] else "execution_unavailable",
                    reason="Required execution_checks did not pass; inspect real producer/consumer receipts",
                )
            if not self._unchanged():
                report.update(status="error", integrity_ok=False, execution_ready=False,
                              error_kind="integrity_error", reason="Frozen execution declarations changed")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            report.update(status="error", error_kind="execution_unavailable", execution_ready=False,
                          reason="execution_checks incomplete: " + str(error))
            (self.artifact_dir / (phase + ".execution-error.json")).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise ModelRequestNotSent("execution_unknown", report["reason"]) from error

    # ============================================================
    # 仓库驱动结果的校验
    # ============================================================
    def _repository_result(self, report, phase):
        return results._repository_result(self, report, phase)

    _read_test_result = staticmethod(read_test_result)
