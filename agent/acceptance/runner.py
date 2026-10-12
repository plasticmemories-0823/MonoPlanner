"""运行冻结的验收规则并保存证据；本地执行器不提供安全沙箱。"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from agent.acceptance import results
from agent.acceptance.results import _read_test_result as read_test_result
from agent.acceptance.artifact_checks import load_artifact_checks, run_artifact_checks
from agent.acceptance.contracts import VerificationSpec, tests_ready_for_candidate
from agent.acceptance.drivers import _BOOTSTRAP, _REPOSITORY_BOOTSTRAP
from agent.acceptance.behavior import declaration_only, run_behavior_contract
from agent.execution.process import log_excerpt, shell_arguments, stop_process


class AcceptanceRun:
    """同一对象的基线和各轮运行使用同一份验收标准。"""

    def __init__(self, repo_root: Path, spec: VerificationSpec,
                 artifacts_root: Path | None = None, timeout: int = 120, environment=None,
                 repository_baseline=None):
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"验收工作区不存在：{self.repo_root}")
        if timeout <= 0:
            raise ValueError("验收超时必须大于 0。")
        self.spec = spec
        self.behavior_only = bool(spec.test_code and declaration_only(spec.test_code))
        self.artifact_checks = load_artifact_checks(spec.test_code, spec.required_consumer, spec.required_calls) if spec.test_code else []
        self.version = hashlib.sha256(json.dumps({
            "criteria": spec.criteria, "command": spec.command, "test_code": spec.test_code,
            "required_consumer": spec.required_consumer,
            "required_calls": spec.required_calls,
            "test_driver": spec.test_driver,
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
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
            "repo_root": str(self.repo_root), "criteria": list(spec.criteria),
            "command": spec.command,
            "test_file": self.test_path.name if spec.test_code is not None else None,
            "test_driver": spec.test_driver,
        }
        self.plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        paths = [self.plan_path]
        if spec.test_code is not None:
            self.test_path.write_text(spec.test_code, encoding="utf-8")
            # Source files are UTF-8 even on Python 3.6 under an ASCII locale.
            # Do not put a Unicode program in the process argument vector.
            self.bootstrap_path.write_text(_BOOTSTRAP, encoding="utf-8")
            paths.extend((self.test_path, self.bootstrap_path))
            if spec.test_driver:
                self.repository_runner = self.artifact_dir / "repository_runner.py"
                self.repository_runner.write_text(_REPOSITORY_BOOTSTRAP, encoding="utf-8")
                paths.append(self.repository_runner)
        self._hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in paths}

    def is_intact(self):
        """供任务控制器检查跨阶段的冻结材料。"""
        return self._unchanged()

    def _unchanged(self):
        try:
            return all(hashlib.sha256(path.read_bytes()).digest() == digest
                       for path, digest in self._hashes.items())
        except OSError:
            return False

    _stop_process = staticmethod(stop_process)

    @staticmethod
    def _summary(path: Path):
        # 完整输出已直接写入文件；只把尾部放回 Agent 上下文。
        return log_excerpt(path, notice="[前文已截断；完整内容见日志]\n")

    def run(self, phase: str) -> dict:
        started = time.monotonic()
        if not isinstance(phase, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", phase):
            raise ValueError("验收阶段名称只能包含字母、数字、下划线和连字符。")
        stdout_path = self.artifact_dir / f"{phase}.stdout.log"
        stderr_path = self.artifact_dir / f"{phase}.stderr.log"
        outcome_path = self.artifact_dir / f"{phase}.tests.json"
        report = {
            "status": "error", "exit_code": None, "reason": "验收未执行。",
            "stdout": "", "stderr": "", "tests_run": None,
            "failures": None, "errors": None, "skipped": None,
            "artifact_dir": str(self.artifact_dir), "phase": phase,
            "integrity_ok": True,
            "verification_version": self.version,
            "test_ids": None, "test_results": None, "test_evidence_valid": False,
            "error_kind": None,
            "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
        }
        # 同一 phase 可显式重跑，不能把上次的结果误当成本次结果。
        outcome_path.unlink(missing_ok=True)
        # 日志只由执行它的后端打开；预先清空后即关闭，避免 Windows 上重复持有文件。
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        if self.behavior_only:
            report = run_behavior_contract(self, phase, report, started)
            report["tests_ready"] = tests_ready_for_candidate(report)
            (self.artifact_dir / f"{phase}.result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        if not self._unchanged():
            report["integrity_ok"] = False
            report["error_kind"] = "integrity_error"
            report["reason"] = "冻结的验收文件被修改或删除，已停止验收。"
        else:
            process_finished = False
            try:
                command = self.spec.command
                backend = self.environment if hasattr(self.environment, "run_process") else None
                map_path = backend.host_to_container if backend else str
                if command is None:
                    # -B 只阻止写缓存；独立目录同时避免读取仓库中的旧 .pyc。
                    cache_dir = tempfile.mkdtemp(prefix=f"{phase}.cache-", dir=self.artifact_dir)
                    command = [
                        backend.python_executable if backend else sys.executable,
                        "-B", "-X", "utf8", "-X", f"pycache_prefix={map_path(cache_dir)}",
                        map_path(self.bootstrap_path), map_path(self.test_path),
                        map_path(self.repo_root), map_path(outcome_path),
                    ]
                    if self.spec.test_driver:
                        command[6] = map_path(self.repository_runner)
                        command.append(json.dumps(self.spec.test_driver))
                options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                            | subprocess.CREATE_NO_WINDOW} if os.name == "nt"
                           else {"start_new_session": True})
                if backend:
                    outcome = backend.run_process(
                        command, stdout_path=stdout_path, stderr_path=stderr_path,
                        timeout=self.timeout, shell=self.spec.command is not None,
                    )
                    report["exit_code"] = outcome["exit_code"]
                    timed_out = outcome["timed_out"]
                    backend_error = outcome.get("error")
                else:
                    local_shell = self.spec.command is not None
                    if local_shell:
                        command, local_shell = shell_arguments(command)
                    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                        process = subprocess.Popen(
                            command, shell=local_shell,
                            cwd=str(self.repo_root), stdout=stdout, stderr=stderr, **options,
                        )
                        timed_out, backend_error = False, None
                        try:
                            report["exit_code"] = process.wait(timeout=self.timeout)
                        except subprocess.TimeoutExpired:
                            self._stop_process(process)
                            report["exit_code"] = process.returncode
                            timed_out = True
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
                    report.update(status="passed" if passed else "failed",
                                  reason="指定验收命令退出码为 0。" if passed
                                  else f"指定验收命令退出码为 {report['exit_code']}。")
                else:
                    self._read_test_result(outcome_path, report)
                    if self.spec.test_driver and report["test_evidence_valid"]:
                        self._repository_result(report, phase)
                    if self.artifact_checks and report["test_evidence_valid"]:
                        from agent.execution.local import LocalEnvironment
                        consumer_backend = backend or LocalEnvironment(self.repo_root, artifact_root=self.artifact_dir)
                        remaining = self.timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            raise RuntimeError('验收总时间已用完，尚未运行原始产物检查。')
                        rows = run_artifact_checks(self.artifact_checks, consumer_backend,
                            self.artifact_dir / (phase + '.artifacts'), remaining)
                        report['artifact_results'] = rows
                        report['test_results'] += rows
                        report['test_ids'] = sorted([*report['test_ids'], *(r['id'] for r in rows)])
                        report['tests_run'] += sum(r['status'] != 'not_run' for r in rows)
                        report['failures'] += sum(r['status'] == 'failed' for r in rows)
                        report['errors'] += sum(r['status'] == 'error' for r in rows)
                        if any(r['status'] == 'error' for r in rows):
                            report.update(status='error', reason='原始产物检查的配置或运行环境有误，不能确认验收。')
                        elif any(r['status'] != 'passed' for r in rows) and report['status'] != 'error':
                            report.update(status='failed', reason='原始产物未通过真实编译或执行；需修复源码后按同一标准返工。')
                    if report["test_evidence_valid"] and report["test_ids"]:
                        inventory = tuple(report["test_ids"])
                        if self._test_ids is None:
                            self._test_ids = inventory
                            inventory_path = self.artifact_dir / "test_inventory.json"
                            inventory_path.write_text(json.dumps(inventory, ensure_ascii=False), encoding="utf-8")
                            self._hashes[inventory_path] = hashlib.sha256(inventory_path.read_bytes()).digest()
                        elif inventory != self._test_ids:
                            report.update(status="error", test_evidence_valid=False,
                                          reason="测试 ID 清单发生变化，不能据此认定新增进展或验收通过。")
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                report.update(status='error', test_evidence_valid=False,
                              error_kind="result_error" if process_finished else "environment_error",
                              reason=f"无法执行验收：{type(exc).__name__}: {exc}")
            if not self._unchanged():
                report.update(status="error", integrity_ok=False,
                              error_kind="integrity_error",
                              test_evidence_valid=False,
                              reason="验收执行期间冻结文件被修改或删除，结果无效。")
        report["stdout"] = self._summary(stdout_path)
        report["stderr"] = self._summary(stderr_path)
        if report["status"] == "error" and report["error_kind"] is None:
            report["error_kind"] = "test_error" if report["test_evidence_valid"] else "result_error"
        report["tests_ready"] = tests_ready_for_candidate(report)
        if self.spec.test_driver:
            report["test_driver"] = list(self.spec.test_driver)
        if report["status"] == "error" and report["tests_ready"]:
            # Keep the raw unittest error rows/counts. The candidate failed these
            # runnable checks; it is not a failed test launch or a passing patch.
            report.update(status="failed", error_kind=None,
                          reason="被测源码执行时抛出异常，未通过固定测试；测试执行证据可用于候选验收。")
        (self.artifact_dir / f"{phase}.result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _repository_result(self, report, phase):
        return results._repository_result(self, report, phase)

    _read_test_result = staticmethod(read_test_result)
