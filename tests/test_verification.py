from agent.execution.local import LocalEnvironment
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest

from agent.acceptance.runner import AcceptanceRun
from agent.acceptance.contracts import VerificationSpec


def acceptance_code(body="self.assertEqual(2 + 2, 4)", decorator=""):
    return (
        "import unittest\n\nclass AcceptanceTests(unittest.TestCase):\n"
        + (f"    {decorator}\n" if decorator else "")
        + "    def test_behavior(self):\n"
        + "\n".join(f"        {line}" for line in body.splitlines()) + "\n"
    )


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.artifacts = self.root / "runs"

    def runner(self, *, code=None, command=None, timeout=120):
        spec = VerificationSpec(("满足可验证的行为要求",), command=command,
                                test_code=acceptance_code() if code is None and command is None else code)
        return AcceptanceRun(self.repo, spec, self.artifacts, timeout, environment=LocalEnvironment(self.repo, artifact_root=self.artifacts))

    def test_generated_tests_pass_and_save_evidence(self):
        runner = self.runner()
        result = runner.run("baseline")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["tests_run"], 1)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["failures"], 0)
        saved = json.loads((runner.artifact_dir / "baseline.result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        self.assertIn("test_behavior", result["stderr"])
        self.assertFalse(runner.artifact_dir.is_relative_to(self.repo))

    def test_malformed_child_status_returns_error_instead_of_escaping_parser(self):
        outcome = self.root / "child-result.json"
        for status in ([], {}, None, "unknown"):
            with self.subTest(status=status):
                outcome.write_text(json.dumps({"status": status}), encoding="utf-8")
                report = {"status": "passed", "exit_code": 0}
                AcceptanceRun._read_test_result(outcome, report)
                self.assertEqual(report["status"], "error")
                self.assertIn("没有有效的验收测试结果", report["reason"])

    def test_same_frozen_tests_check_before_and_after_repository_change(self):
        module = self.repo / "sample_app.py"
        module.write_text("VALUE = 3\n", encoding="utf-8")
        runner = self.runner(code=acceptance_code("from sample_app import VALUE\nself.assertEqual(VALUE, 4)"))
        before = runner.run("baseline")
        module.write_text("VALUE = 4\n", encoding="utf-8")
        after = runner.run("final")
        self.assertEqual(before["status"], "failed")
        self.assertEqual(before["failures"], 1)
        self.assertEqual(after["status"], "passed")

    def test_same_size_edit_does_not_load_existing_stale_bytecode(self):
        module = self.repo / "cached_app.py"
        module.write_text("VALUE = 3\n", encoding="utf-8")
        os.utime(module, (1700000000, 1700000000))
        cached_path = Path(py_compile.compile(str(module), doraise=True))
        cached_bytes = cached_path.read_bytes()
        module.write_text("VALUE = 4\n", encoding="utf-8")
        os.utime(module, (1700000000, 1700000000))

        runner = self.runner(code=acceptance_code("from cached_app import VALUE\nself.assertEqual(VALUE, 4)"))
        result = runner.run("final")

        self.assertEqual(result["status"], "passed", result["stderr"])
        self.assertEqual(cached_path.read_bytes(), cached_bytes)

    def test_generated_runtime_error_is_distinct_from_assertion_failure(self):
        result = self.runner(code=acceptance_code("import module_that_does_not_exist_for_test")).run("final")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"], 1)

    def test_import_error_is_not_a_successful_empty_test_run(self):
        code = "import module_that_does_not_exist_for_test\n" + acceptance_code()
        result = self.runner(code=code).run("final")
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["tests_run"])

    def test_empty_invalid_or_ambiguous_specs_are_rejected(self):
        invalid_specs = [
            ((), {"command": "echo OK"}),
            (("",), {"command": "echo OK"}),
            (("A",), {}),
            (("A",), {"command": ""}),
            (("A",), {"test_code": "def broken(:"}),
            (("A",), {"command": "echo OK", "test_code": acceptance_code()}),
        ]
        for criteria, kwargs in invalid_specs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VerificationSpec(criteria, **kwargs)

    def test_non_test_module_is_rejected_by_actual_collection(self):
        result = self.runner(code="print('not a test')").run("candidate")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["tests_run"], 0)
        self.assertFalse(result["tests_ready"])

    def test_skips_expected_failures_and_zero_discovered_tests_never_pass(self):
        codes = [acceptance_code(decorator="@unittest.skip('not implemented')"),
                 acceptance_code("self.fail('TODO')", "@unittest.expectedFailure"),
                 acceptance_code() + "\ndef load_tests(loader, tests, pattern):\n    return unittest.TestSuite()\n"]
        for code in codes:
            with self.subTest(code=code):
                result = self.runner(code=code).run("final")
                self.assertEqual(result["status"], "error")

    def test_unittest_import_alias_is_supported(self):
        code = "from unittest import TestCase as Base\nclass Acceptance(Base):\n    def test_ok(self):\n        self.assertTrue(True)\n"
        self.assertEqual(self.runner(code=code).run("final")["status"], "passed")

    def test_user_command_uses_exit_code_without_inventing_test_counts(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code):
                command = subprocess.list2cmdline([sys.executable, "-B", "-c", f"raise SystemExit({exit_code})"])
                result = self.runner(command=command).run("final")
                self.assertEqual(result["status"], "passed" if exit_code == 0 else "failed")
                self.assertEqual(result["exit_code"], exit_code)
                self.assertIsNone(result["tests_run"])

    def test_timeout_is_an_error_and_keeps_partial_output(self):
        code = acceptance_code("import time\nprint('BEFORE_TIMEOUT', flush=True)\ntime.sleep(30)")
        result = self.runner(code=code, timeout=0.5).run("final")
        self.assertEqual(result["status"], "error")
        self.assertIn("超过", result["reason"])
        self.assertIn("BEFORE_TIMEOUT", result["stdout"])

    def test_tampered_test_or_plan_prevents_execution(self):
        for filename in ("test_acceptance.py", "plan.json"):
            with self.subTest(filename=filename):
                runner = self.runner()
                (runner.artifact_dir / filename).write_text("TAMPERED", encoding="utf-8")
                result = runner.run("final")
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["exit_code"])
                self.assertIn("冻结", result["reason"])

    def test_test_file_modified_during_execution_invalidates_success(self):
        code = acceptance_code(
            "from pathlib import Path\n"
            "path = Path(__file__)\n"
            "path.write_text(path.read_text(encoding='utf-8') + '\\n# changed', encoding='utf-8')\n"
            "self.assertTrue(True)"
        )
        result = self.runner(code=code).run("final")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["status"], "error")
        self.assertIn("执行期间", result["reason"])

    def test_logs_are_full_on_disk_but_summaries_are_bounded(self):
        result = self.runner(code=acceptance_code("print('A' * 20000)\nself.assertTrue(True)")).run("final")
        self.assertEqual(result["status"], "passed")
        self.assertLess(len(result["stdout"]), 6100)
        self.assertGreater(Path(result["stdout_log"]).stat().st_size, 20000)
        self.assertEqual(result["stdout"], "[前文已截断；完整内容见日志]\n" +
                         Path(result["stdout_log"]).read_bytes()[-6000:].decode("utf-8", errors="replace"))

    def test_phase_names_cannot_escape_artifact_directory(self):
        runner = self.runner()
        with self.assertRaises(ValueError):
            runner.run("../outside")


if __name__ == "__main__":
    unittest.main()
