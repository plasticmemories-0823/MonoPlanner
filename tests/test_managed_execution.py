"""验证真实入口、不可替换的执行回执及通用消费声明的失败边界。"""

import json
from pathlib import Path
import tempfile
import unittest

from agent.acceptance.contracts import VerificationSpec
from agent.acceptance.execution import checked_execution, validate_execution_result
from agent.acceptance.runner import AcceptanceRun
from agent.execution.local import LocalEnvironment
from agent.task.runner import TaskRunner
from agent.model.errors import ModelRequestNotSent
from unittest.mock import patch
import test_task_runner as fixture


def check(**changes):
    return {
        "id": "value", "requirement_quote": "", "artifact_type": "repository_entry",
        "imports": "", "setup": "", "expression": "", "consumer": "command",
        "entry_file": "sample.py", "interpreter": "python", "argv": [],
        "variables": [], "result_variable": "", "cases": [], **changes,
    }


def code_check(**changes):
    return check(artifact_type="generated_code", imports="from sample import value",
                 expression="value()", entry_file="", interpreter="", **{
                     "consumer": "c_expression", "cases": [{"values": [], "expected": 4}],
                     **changes,
                 })


class ManagedExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "sample.py").write_text("def value():\n    return 4\n", encoding="utf-8")
        self.env = LocalEnvironment(self.repo, artifact_root=self.root / "runtime")

    def execute(self, declaration):
        spec = VerificationSpec(("Return the public value",), test_code=fixture.CODE,
                                execution_checks=[declaration])
        return AcceptanceRun(self.repo, spec, environment=self.env,
                             artifacts_root=self.root / "acceptance").run("candidate")

    def test_real_repository_entry_and_source_receipt(self):
        result = self.execute(check())
        self.assertEqual(result["status"], "passed", result)
        row = result["execution"]["checks"][0]
        self.assertTrue(row["producer_called"])
        self.assertEqual(Path(row["producer_source"]), self.repo / "sample.py")
        self.assertEqual(len(row["producer_source_sha256"]), 64)

    def test_passing_unit_tests_cannot_cover_missing_producer(self):
        declaration = code_check()
        declaration.update(imports="from sample import absent", expression="absent()")
        result = self.execute(declaration)
        self.assertEqual(result["status"], "error", result)
        self.assertFalse(result["execution_ready"])
        self.assertFalse(result["tests_ready"])

    def test_external_substitute_is_not_a_repository_producer(self):
        declaration = code_check()
        declaration.update(imports="from json import dumps", expression="dumps(4)")
        result = self.execute(declaration)
        self.assertEqual(result["status"], "error", result)
        self.assertFalse(result["execution"]["checks"][0]["producer_called"])

    def test_source_failure_is_recorded_as_behavior_failure(self):
        (self.repo / "sample.py").write_text("def value():\n    raise ValueError('broken')\n")
        result = self.execute(code_check())
        self.assertEqual(result["status"], "failed", result)
        self.assertTrue(result["execution_ready"])
        self.assertEqual(result["execution"]["checks"][0]["error_kind"], "source")

    def test_same_receipt_is_not_executed_twice(self):
        spec = VerificationSpec(("Return value",), test_code=fixture.CODE, execution_checks=[check()])
        runner = AcceptanceRun(self.repo, spec, environment=self.env,
                               artifacts_root=self.root / "acceptance")
        self.assertEqual(runner.run("candidate")["status"], "passed")
        before = (runner.artifact_dir / "candidate.tests.json").read_bytes()
        with self.assertRaisesRegex(ModelRequestNotSent, "without replay"):
            runner.run("candidate")
        self.assertEqual((runner.artifact_dir / "candidate.tests.json").read_bytes(), before)

    def test_frozen_execution_file_tamper_rejected(self):
        spec = VerificationSpec(("Return value",), test_code=fixture.CODE, execution_checks=[check()])
        runner = AcceptanceRun(self.repo, spec, environment=self.env,
                               artifacts_root=self.root / "acceptance")
        runner.execution_path.write_text("[]")
        result = runner.run("candidate")
        self.assertEqual(result["error_kind"], "integrity_error")
        self.assertFalse(result["integrity_ok"])

    def test_empty_declaration_cannot_enter_task_acceptance(self):
        model = fixture.Model([{
            "kind": "candidate", "note": "change value", "steps": [], "test_code": fixture.CODE,
        }])
        loop = TaskRunner(model, self.env, state_dir=self.root / "state")
        loop.state = {"request": "Return value", "acceptance": "", "verify_command": None}
        with self.assertRaisesRegex(ValueError, "execution_checks"):
            loop.validate_work({"kind": "candidate", "note": "change", "test_code": fixture.CODE})

    def test_requirement_quote_must_be_original(self):
        with self.assertRaisesRegex(ValueError, "requirement_quote"):
            checked_execution([check(requirement_quote="invented requirement")], ("Return value",))

    def test_entry_cannot_be_replaced_in_setup(self):
        with self.assertRaisesRegex(ValueError, "replace imported"):
            checked_execution([code_check(setup="value = 4")], ("Return value",))

    def test_expression_cannot_replace_raw_output(self):
        with self.assertRaisesRegex(ValueError, "directly call"):
            declaration = code_check()
            declaration["expression"] = "value().replace('bad', 'good')"
            checked_execution([declaration], ("Return value",))

    def test_invalid_python_is_a_validation_error(self):
        with self.assertRaisesRegex(ValueError, "execution_checks.*syntax"):
            declaration = code_check()
            declaration["expression"] = "value("
            checked_execution([declaration], ("Return value",))

    def test_numeric_arity_and_zero_cases_rejected(self):
        with self.assertRaisesRegex(ValueError, "actual input"):
            checked_execution([code_check(cases=[])], ("Generate code",))
        with self.assertRaisesRegex(ValueError, "arity"):
            checked_execution([code_check(variables=["x"],
                                     cases=[{"values": [], "expected": 0}])], ("Generate code",))

    def test_fake_pass_without_producer_receipt_rejected(self):
        path = self.root / "result.json"
        path.write_text(json.dumps({"checks": [{"id": "value", "consumer": "command", "status": "passed",
                                                "artifact_type": "repository_entry", "target_runtime": "python"}]}))
        with self.assertRaisesRegex(ValueError, "producer receipt"):
            validate_execution_result(path, [check()])

    def test_fake_numeric_pass_without_compilation_rejected(self):
        path = self.root / "result.json"
        value = code_check()
        path.write_text(json.dumps({"checks": [{"id": "value", "consumer": "c_expression", "status": "passed",
                                                "artifact_type": "generated_code", "target_runtime": "c",
                                                "producer_called": True, "producer_source_sha256": "0" * 64}]}))
        with self.assertRaisesRegex(ValueError, "compilation"):
            validate_execution_result(path, [value])

    def test_unknown_execution_stops_instead_of_returning_retryable_test_failure(self):
        spec = VerificationSpec(("Return value",), test_code=fixture.CODE, execution_checks=[check()])
        runner = AcceptanceRun(self.repo, spec, environment=self.env,
                               artifacts_root=self.root / "acceptance")
        actual = self.env.run_process

        def interrupt(command, **kwargs):
            if any("execution_runner.py" in str(arg) for arg in command):
                return {"exit_code": None, "timed_out": True, "process_cleanup_confirmed": False}
            return actual(command, **kwargs)

        with patch.object(self.env, "run_process", side_effect=interrupt):
            with self.assertRaises(ModelRequestNotSent):
                runner.run("candidate")
        self.assertTrue((runner.artifact_dir / "candidate.execution-error.json").exists())

    def test_repository_script_runs_with_exact_arguments(self):
        (self.repo / "entry.py").write_text("import sys\nprint(sys.argv[1])\n")
        declaration = check(consumer="command", imports="", expression="", entry_file="entry.py",
                            interpreter="python", argv=["literal ; not shell code"])
        result = self.execute(declaration)
        self.assertEqual(result["status"], "passed", result)
        row = result["execution"]["checks"][0]
        self.assertEqual(row["stdout"].strip(), "literal ; not shell code")
        self.assertEqual(row["exit_code"], 0)

    def test_script_failure_blocks_passing_unit_tests(self):
        (self.repo / "entry.py").write_text("raise SystemExit(3)\n")
        result = self.execute(check(consumer="command", imports="", expression="",
                                    entry_file="entry.py", interpreter="python"))
        self.assertEqual(result["status"], "failed", result)
        self.assertTrue(result["tests_ready"])

    def test_new_script_absence_is_a_failed_original_behavior(self):
        result = self.execute(check(consumer="command", imports="", expression="",
                                    entry_file="entry.py", interpreter="python"))
        self.assertEqual(result["status"], "failed", result)
        self.assertTrue(result["tests_ready"])
        self.assertFalse(result["execution"]["checks"][0]["producer_called"])

    def test_command_entry_cannot_escape_repository(self):
        with self.assertRaisesRegex(ValueError, "relative repository"):
            checked_execution([check(consumer="command", imports="", expression="",
                                     entry_file="../outside.py", interpreter="python")], ("run entry",))

    def test_call_only_acceptance_is_rejected(self):
        with self.assertRaises(ValueError):
            checked_execution([code_check(consumer="call", cases=[])],
                              ("Deliver usable code",), required=True)

    def test_zero_exit_without_compilation_receipt_is_rejected(self):
        declaration = check(consumer="command", imports="", expression="",
                            entry_file="entry.py", interpreter="python")
        result = self.root / "incomplete-result.json"
        result.write_text(json.dumps({"checks": [{
            "id": "value", "consumer": "command", "status": "passed",
            "artifact_type": "repository_entry", "target_runtime": "python",
            "producer_called": True, "producer_source_sha256": "0" * 64,
            "exit_code": 0, "argv": ["python", "entry.py"],
        }]}))
        with self.assertRaisesRegex(ValueError, "compilation"):
            validate_execution_result(result, [declaration])

    def test_script_has_successful_compile_receipt_before_running(self):
        (self.repo / "entry.py").write_text("print('ready')\n")
        result = self.execute(check(consumer="command", imports="", expression="",
                                    entry_file="entry.py", interpreter="python"))
        row = result["execution"]["checks"][0]
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(row.get("compile_exit_code"), 0)
        self.assertEqual(row["compiled_source_sha256"], row["producer_source_sha256"])
        self.assertEqual(row["exit_code"], 0)

    def test_compile_failure_blocks_entry_execution(self):
        (self.repo / "entry.py").write_text(
            "from pathlib import Path\nPath('ran').write_text('yes')\ninvalid syntax !\n")
        result = self.execute(check(entry_file="entry.py"))
        row = result["execution"]["checks"][0]
        self.assertEqual(result["status"], "failed", result)
        self.assertNotEqual(row["compile_exit_code"], 0)
        self.assertFalse(row["producer_called"])
        self.assertNotIn("exit_code", row)
        self.assertFalse((self.repo / "ran").exists())

    def test_unchecked_binary_entry_is_rejected(self):
        with self.assertRaises(ValueError):
            checked_execution([check(interpreter="executable")], ("run entry",))

    def test_generated_artifact_cannot_use_repository_entry_consumer(self):
        with self.assertRaisesRegex(ValueError, "producer output"):
            checked_execution([check(artifact_type="generated_code")], ("produce code",))

    def test_runtime_receipt_cannot_switch_artifact_or_compiled_source(self):
        result = self.execute(check())
        evidence = result["execution"]
        output = self.root / "changed-result.json"
        evidence["checks"][0]["compiled_source_sha256"] = "0" * 64
        output.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "command evidence"):
            validate_execution_result(output, [check()])

    def test_complete_looking_receipt_without_real_artifacts_is_rejected(self):
        evidence = self.execute(check())["execution"]
        copied = self.root / 'forged.execution.json'
        copied.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, 'original process logs'):
            validate_execution_result(copied, [check()], repository=self.repo,
                                      map_path=self.env.host_to_container)

    def test_receipt_cannot_reuse_changed_actual_entry_output(self):
        result = self.execute(check())
        root = Path(result['artifact_dir'])
        (root / 'candidate.execution-value' / 'command.stdout').write_text('forged')
        with self.assertRaisesRegex(ValueError, 'original artifact bytes'):
            validate_execution_result(root / 'candidate.execution.json', [check()],
                                      repository=self.repo, map_path=self.env.host_to_container)
