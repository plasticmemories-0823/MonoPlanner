"""Acceptance uses its selected execution backend for both command kinds."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent.acceptance.runner import AcceptanceRun
from agent.acceptance.contracts import VerificationSpec


TEST_CODE = """import unittest
class Acceptance(unittest.TestCase):
    def test_answer(self):
        self.assertEqual(6 * 7, 42)
"""


class VerificationBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.backend = Mock()
        self.backend.python_executable = "/container/python"
        self.backend.host_to_container.side_effect = lambda path: (
            "/testbed" if Path(path) == self.repo
            else "/evidence/" + Path(path).relative_to(self.base).as_posix()
        )

    def runner(self, command=None):
        spec = VerificationSpec(("计算结果应为 42",), command=command,
                                test_code=None if command is not None else TEST_CODE)
        return AcceptanceRun(self.repo, spec, self.base / "evidence", environment=self.backend)

    @patch("agent.acceptance.runner.subprocess.Popen")
    def test_custom_command_uses_backend_exit_code_and_no_host_process(self, popen):
        for code in (0, 7):
            with self.subTest(code=code):
                self.backend.run_process.return_value = {"exit_code": code, "timed_out": False}
                result = self.runner(command="python check.py").run("final")
                self.assertEqual(result["status"], "passed" if code == 0 else "failed")
                self.assertEqual(result["exit_code"], code)
                self.assertIsNone(result["tests_run"])
                self.assertEqual(self.backend.run_process.call_args.args, ("python check.py",))
                self.assertTrue(self.backend.run_process.call_args.kwargs["shell"])
        popen.assert_not_called()

    @patch("agent.acceptance.runner.subprocess.Popen")
    def test_generated_test_uses_container_python_mapped_paths_and_structured_result(self, popen):
        runner = self.runner()

        def execute(command, **kwargs):
            self.assertEqual(command[0], "/container/python")
            self.assertNotIn("-c", command)
            self.assertTrue(command[-4].endswith("acceptance_runner.py"))
            self.assertIn("保留每个实际测试", runner.bootstrap_path.read_text(encoding="utf-8"))
            self.assertEqual(command[-2], "/testbed")
            self.assertTrue(command[-3].startswith("/evidence/"))
            self.assertFalse(kwargs["shell"])
            # Renaming succeeds on Windows only if no outer stream holds the file.
            stdout = kwargs["stdout_path"]
            stdout.rename(stdout.with_suffix(".empty"))
            stdout.write_text("CONTAINER_OUTPUT", encoding="utf-8")
            (runner.artifact_dir / "final.tests.json").write_text(json.dumps({
                "status": "passed", "tests_run": 1, "failures": 0, "errors": 0,
                "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
                "test_ids": ["agent_acceptance_tests.Acceptance.test_answer"],
                "test_results": [{"id": "agent_acceptance_tests.Acceptance.test_answer",
                                  "status": "passed", "detail": ""}],
            }), encoding="utf-8")
            return {"exit_code": 0, "timed_out": False}

        self.backend.run_process.side_effect = execute
        result = runner.run("final")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["tests_run"], 1)
        self.assertEqual(result["stdout"], "CONTAINER_OUTPUT")
        popen.assert_not_called()

    def test_backend_error_cannot_be_overridden_by_zero_exit(self):
        self.backend.run_process.return_value = {
            "exit_code": 0, "timed_out": False, "error": "missing process record",
        }
        result = self.runner(command="echo ok").run("final")
        self.assertEqual(result["status"], "error")
        self.assertIn("missing process record", result["reason"])
        self.assertEqual(result["error_kind"], "environment_error")

    def test_backend_startup_error_is_saved_as_execution_evidence(self):
        self.backend.run_process.side_effect = RuntimeError("image not available")
        result = self.runner(command="echo ok").run("final")
        self.assertEqual(result["status"], "error")
        self.assertIn("image not available", result["reason"])
        self.assertEqual(result["error_kind"], "environment_error")
        self.assertTrue((Path(result["artifact_dir"]) / "final.result.json").is_file())

    def test_backend_timeout_keeps_partial_logs_and_reports_uncertain_cleanup(self):
        def execute(command, **kwargs):
            kwargs["stdout_path"].write_text("BEFORE_TIMEOUT", encoding="utf-8")
            return {"exit_code": 124, "timed_out": True, "process_cleanup_confirmed": False}
        self.backend.run_process.side_effect = execute
        result = self.runner(command="sleep 100").run("final")
        self.assertEqual(result["status"], "error")
        self.assertIn("无法确认", result["reason"])
        self.assertEqual(result["error_kind"], "timeout")
        self.assertEqual(result["stdout"], "BEFORE_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
