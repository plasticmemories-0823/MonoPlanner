from agent.execution.local import LocalEnvironment
"""Repository setup and native result classes survive generated acceptance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from agent.acceptance.contracts import VerificationSpec
from agent.acceptance.runner import AcceptanceRun


class RepositoryAcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "configuration.py").write_text("DEFAULT_ROW = 17\n", encoding="utf8")
        (self.repo / "sample.py").write_text("VALUE = 4\n", encoding="utf8")
        (self.repo / "native_support.py").write_text(
            "import sqlite3, unittest\nfrom configuration import DEFAULT_ROW\n"
            "class NativeCase(unittest.TestCase):\n"
            "    def setUp(self):\n"
            "        self.db = sqlite3.connect(':memory:')\n"
            "        self.addCleanup(self.db.close)\n"
            "        self.db.execute('create table condition(value integer)')\n"
            "        self.db.execute('insert into condition values (?)', (DEFAULT_ROW,))\n"
            "class NativeResult(unittest.TextTestResult):\n"
            "    def addSuccess(self, test):\n"
            "        print('NATIVE_RESULT_HANDLER')\n"
            "        super().addSuccess(test)\n", encoding="utf8")
        (self.repo / "run_selected.py").write_text(
            "import sys, unittest\nfrom native_support import NativeResult\n"
            "suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])\n"
            "result = unittest.TextTestRunner(resultclass=NativeResult).run(suite)\n"
            "sys.exit(0 if result.wasSuccessful() else 1)\n", encoding="utf8")
        self.code = ("from native_support import NativeCase\nimport sample\n"
                     "class GeneratedTests(NativeCase):\n"
                     "    def test_behavior(self):\n"
                     "        self.assertEqual(self.db.execute('select value from condition').fetchone()[0], 17)\n"
                     "        self.assertEqual(sample.VALUE, 4)\n")

    def runner(self, code=None, driver=None):
        spec = VerificationSpec(("public behavior",), test_code=self.code if code is None else code,
                                test_driver=["run_selected.py", "{test_module}"] if driver is None else driver)
        return AcceptanceRun(self.repo, spec, self.root / "runs", environment=LocalEnvironment(self.repo, artifact_root=self.root / "runs"))

    def test_native_database_configuration_and_result_class_are_used(self):
        runner = self.runner()
        result = runner.run("baseline")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["tests_run"], 1)
        self.assertTrue(result["tests_ready"])
        self.assertIn("NATIVE_RESULT_HANDLER", result["stdout"])
        saved = json.loads((runner.artifact_dir / "baseline.tests.json").read_text())
        self.assertEqual(saved["generated_test_ids"], result["test_ids"])
        self.assertFalse(any("unittest" in line for line in self.code.splitlines()))

    def test_same_frozen_driver_checks_original_failure_and_fixed_source(self):
        runner = self.runner()
        (self.repo / "sample.py").write_text("VALUE = 3\n")
        baseline = runner.run("baseline")
        self.assertEqual(baseline["status"], "failed", baseline)
        self.assertTrue(baseline["tests_ready"])
        (self.repo / "sample.py").write_text("VALUE = 4\n")
        self.assertEqual(runner.run("candidate")["status"], "passed")

    def test_repository_success_without_generated_cases_is_rejected(self):
        (self.repo / "run_selected.py").write_text("import sys\nprint('success')\nsys.exit(0)\n")
        result = self.runner().run("candidate")
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["tests_ready"])
        self.assertFalse(result["test_evidence_valid"])

    def test_collection_skip_load_and_assertion_failures_remain_real(self):
        for code in (self.code.replace("self.assertEqual(sample.VALUE, 4)", "self.assertEqual(sample.VALUE, 99)"),
                     self.code.replace("import sample", "import missing_acceptance_module"),
                     self.code.replace("self.assertEqual(sample.VALUE, 4)", "self.skipTest('missing')")):
            with self.subTest(code=code):
                self.assertNotEqual(self.runner(code).run("candidate")["status"], "passed")

    def test_frozen_driver_adapter_cannot_be_modified(self):
        runner = self.runner()
        runner.repository_runner.write_text("raise RuntimeError('modified')\n")
        result = runner.run("candidate")
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["integrity_ok"])

    def repository_suite(self, code):
        (self.repo / "repo_cases.py").write_text(code, encoding="utf8")
        (self.repo / "run_selected.py").write_text(
            "import sys, unittest\nfrom native_support import NativeResult\n"
            "suite = unittest.defaultTestLoader.loadTestsFromNames([sys.argv[1], 'repo_cases'])\n"
            "result = unittest.TextTestRunner(resultclass=NativeResult).run(suite)\n"
            "sys.exit(0 if result.wasSuccessful() else 1)\n", encoding="utf8")

    def test_native_skip_is_separate_and_original_failure_can_freeze(self):
        self.repository_suite(
            "import unittest\nclass NativeChecks(unittest.TestCase):\n"
            "    def test_existing(self):\n        self.assertEqual(2 + 2, 4)\n"
            "    @unittest.skip('native database feature')\n"
            "    def test_conditional(self):\n        self.fail('not applicable')\n")
        (self.repo / "sample.py").write_text("VALUE = 3\n")
        baseline = self.runner().run("baseline")
        self.assertEqual(baseline["status"], "failed", baseline)
        self.assertTrue(baseline["tests_ready"])
        self.assertEqual(baseline["test_groups"]["generated"]["counts"], {"failed": 1})
        self.assertEqual(baseline["test_groups"]["repository"]["counts"], {"passed": 1, "skipped": 1})
        (self.repo / "sample.py").write_text("VALUE = 4\n")
        runner = self.runner()
        runner.repository_baseline = baseline
        result = runner.run("candidate")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["tests_run"], 3)
        self.assertEqual(result["skipped"], 1)
        self.assertTrue(result["repository_baseline_matched"])
        self.assertEqual(result["repository_skips"], baseline["repository_skips"])
        self.assertIn("NATIVE_RESULT_HANDLER", result["stdout"])

    def test_generated_skip_is_still_rejected_with_native_skip(self):
        self.repository_suite("import unittest\n@unittest.skip('native')\nclass Repo(unittest.TestCase):\n"
                              "    def test_conditional(self):\n        pass\n")
        result = self.runner(self.code.replace("self.assertEqual(sample.VALUE, 4)", "self.skipTest('generated')")).run("baseline")
        self.assertNotEqual(result["status"], "passed")
        self.assertFalse(result["tests_ready"])

    def test_repository_real_failure_is_not_hidden_by_generated_success(self):
        self.repository_suite("import unittest\nclass Repo(unittest.TestCase):\n"
                              "    def test_regression(self):\n        self.assertEqual(2, 3)\n")
        result = self.runner().run("baseline")
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["test_groups"]["generated"]["counts"], {"passed": 1})
        self.assertEqual(result["test_groups"]["repository"]["counts"], {"failed": 1})

    def test_new_repository_skip_cannot_replace_a_failing_regression(self):
        self.repository_suite("import unittest, sample\nclass Repo(unittest.TestCase):\n"
                              "    def test_regression(self):\n"
                              "        if sample.VALUE == 4:\n            self.skipTest('hide failure')\n"
                              "        self.assertEqual(2, 3)\n")
        (self.repo / "sample.py").write_text("VALUE = 3\n")
        baseline = self.runner().run("baseline")
        (self.repo / "sample.py").write_text("VALUE = 4\n")
        runner = self.runner()
        runner.repository_baseline = baseline
        result = runner.run("candidate")
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(result["error_kind"], "repository_comparison")
        self.assertFalse(result["tests_ready"])

    def test_repository_skip_requires_original_comparison(self):
        self.repository_suite("import unittest\n@unittest.skip('native')\nclass Repo(unittest.TestCase):\n"
                              "    def test_conditional(self):\n        pass\n")
        result = self.runner().run("candidate")
        self.assertEqual(result["status"], "error", result)
        self.assertFalse(result["tests_ready"])

    def test_repository_inventory_cannot_shrink_after_freeze(self):
        self.repository_suite("import unittest, sample\nclass Repo(unittest.TestCase):\n"
                              "    if sample.VALUE == 3:\n"
                              "        def test_regression(self):\n            self.assertEqual(2, 3)\n")
        (self.repo / "sample.py").write_text("VALUE = 3\n")
        baseline = self.runner().run("baseline")
        (self.repo / "sample.py").write_text("VALUE = 4\n")
        runner = self.runner()
        runner.repository_baseline = baseline
        result = runner.run("candidate")
        self.assertEqual(result["status"], "error", result)
        self.assertIn("清单", result["reason"])

    def test_native_exit_fault_and_setup_error_still_block_freeze(self):
        original = (self.repo / "run_selected.py").read_text()
        for ending in ("sys.exit(2)", "raise RuntimeError('native post-run fault')"):
            with self.subTest(ending=ending):
                (self.repo / "run_selected.py").write_text(original.replace("sys.exit(0 if result.wasSuccessful() else 1)", ending))
                result = self.runner().run("baseline")
                self.assertEqual(result["status"], "error", result)
                self.assertFalse(result["tests_ready"])

    def test_repository_skip_reason_cannot_change_after_freeze(self):
        self.repository_suite("import unittest, sample\nclass Repo(unittest.TestCase):\n"
                              "    def test_conditional(self):\n        self.skipTest(str(sample.VALUE))\n")
        (self.repo / "sample.py").write_text("VALUE = 3\n")
        baseline = self.runner().run("baseline")
        (self.repo / "sample.py").write_text("VALUE = 4\n")
        runner = self.runner()
        runner.repository_baseline = baseline
        self.assertEqual(runner.run("candidate")["status"], "error")

    def test_driver_requires_explicit_generated_target_and_bound_python_entry(self):
        for driver in (["../outside.py", "{test_module}"], ["/outside.py", "{test_file}"],
                       ["run_selected.py", "existing_repository_test"], ["-c", "{test_file}"]):
            with self.subTest(driver=driver), self.assertRaises(ValueError):
                self.runner(driver=driver)

    @unittest.skipUnless(importlib.util.find_spec("pytest"), "No local pytest installation")
    def test_pytest_functions_reuse_repository_configuration(self):
        (self.repo / "conftest.py").write_text(
            "import pytest\n@pytest.fixture\ndef condition():\n    return 17\n", encoding="utf8")
        # pytest only loads conftest along the test path; a repository can load
        # its fixture plugin explicitly for a generated module outside source.
        runner = self.runner("def test_behavior(condition):\n    import sample\n    assert condition == 17\n    assert sample.VALUE == 4\n",
                             ["-m", "pytest", "{test_file}", "-p", "conftest"])
        result = runner.run("candidate")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["tests_run"], 1)
        self.assertTrue(result["tests_ready"])

    @unittest.skipUnless(importlib.util.find_spec("pytest"), "No local pytest installation")
    def test_pytest_fail_skip_and_setup_error_cannot_pass(self):
        for code in ("def test_behavior():\n    assert 2 == 3\n",
                     "import pytest\ndef test_behavior():\n    pytest.skip('missing')\n",
                     "import pytest\n@pytest.fixture\ndef broken():\n    raise RuntimeError('setup')\ndef test_behavior(broken):\n    assert broken == 1\n"):
            with self.subTest(code=code):
                self.assertNotEqual(self.runner(code, ["-m", "pytest", "{test_file}"]).run("candidate")["status"], "passed")
