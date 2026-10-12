"""Real original-source trials, explicit review, version comparison and recovery."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from agent.acceptance.revisions import behavior
from agent.task.runner import TaskRunner
from agent.task.budget import StepBudget
from agent.storage.history import LocalHistory
from agent.model.payload import wire_input
from agent.model.tools import WORK_TOOLS
from agent.storage.recovery import code_key
from support import interrupt_after_persist
import test_task_runner as f


def review(messages, name="freeze_acceptance", code=None):
    result = json.loads(messages[-1].content)
    identifier = result.get("trial_id") or result["acceptance_trial_id"]
    args = {"trial_id": identifier, "reason": "Use the recorded original-source execution evidence"}
    if code is not None:
        args["test_code"] = code
    return AIMessage(content="", tool_calls=[{"name": name, "id": name + "-call", "args": args}],
                     usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})


class ScriptModel:
    def __init__(self, replies):
        self.replies, self.calls, self.tools = iter(replies), [], []

    def bind_tools(self, tools):
        self.tools.append(wire_input([], tools)["tools"])
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        item = next(self.replies)
        return item(messages) if callable(item) else item


class AcceptanceReviewTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    candidate = f.TaskRunnerTests.candidate
    shell = f.TaskRunnerTests.shell

    def run_replies(self, replies, **kw):
        self.model = ScriptModel(replies)
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state", **kw)
        return self.engine.run([], "return 4", StepBudget(max_steps=12))

    def state(self):
        store = LocalHistory(self.root / "state")
        try:
            with store.lease():
                return store.load()
        finally:
            store.close()

    def crash(self, event):
        return interrupt_after_persist(event)

    def test_no_source_edit_or_freeze_before_explicit_confirmation(self):
        interrupt, error = self.crash("acceptance_trial_finished")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate())])
        self.assertIn("return 0", (self.repo / "sample.py").read_text())
        self.assertFalse((self.root / "state/frozen-acceptance.json").exists())
        baseline = json.loads(self.model.calls[0][-1].content)["initial_evidence"]
        self.assertIn("source", baseline)
        result = self.run_replies([review])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["planner_calls"], 2)
        self.assertEqual(result["calls_this_invocation"], 1)
        self.assertEqual(len(self.state()["acceptance_trials"]), 1)

    def test_repository_driver_survives_trial_freeze_and_recovery(self):
        (self.repo / "native_support.py").write_text(
            "import unittest\nclass NativeCase(unittest.TestCase):\n    pass\n")
        (self.repo / "runtests.py").write_text(
            "import sys, unittest\n"
            "class RepoChecks(unittest.TestCase):\n"
            "    @unittest.skip('native configuration')\n"
            "    def test_conditional(self):\n        pass\n"
            "suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])\n"
            "suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RepoChecks))\n"
            "result = unittest.TextTestRunner().run(suite)\n"
            "sys.exit(0 if result.wasSuccessful() else 1)\n")
        code = f.CODE.replace("import unittest", "from native_support import NativeCase").replace("unittest.TestCase", "NativeCase")
        interrupt, error = self.crash("acceptance_trial_finished")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate(test_code=code, test_driver=["runtests.py", "{test_module}"]))])
        result = self.run_replies([review])
        self.assertEqual(result["status"], "verified_success", result)
        frozen = json.loads((self.root / "state/frozen-acceptance.json").read_text())
        self.assertEqual(frozen["spec"]["test_driver"], ["runtests.py", "{test_module}"])
        self.assertEqual(result["calls_this_invocation"], 1)
        self.assertEqual(result["verification"]["test_groups"]["generated"]["counts"], {"passed": 1})
        self.assertEqual(result["verification"]["test_groups"]["repository"]["counts"], {"skipped": 1})
        self.assertTrue(result["verification"]["repository_baseline_matched"])

    def test_missing_initialization_can_be_revised_before_freeze(self):
        bad = "import missing_setup_module\n" + f.CODE
        result = self.run_replies([f.reply(self.candidate(test_code=bad)),
            lambda m: review(m, "revise_acceptance", f.CODE), review])
        self.assertEqual(result["status"], "verified_success", result)
        trial = json.loads(self.model.calls[1][-1].content)
        self.assertFalse(trial["baseline"]["tests_ready"])
        self.assertEqual(trial["baseline"]["execution_exception"]["phase"], "load")
        self.assertIn("missing_setup_module", (self.root / "state" / trial["proposal_record"][8:]).read_text())
        self.assertEqual(len(self.state()["acceptance_trials"]), 2)
        self.assertEqual(len(self.state()["acceptance_versions"]), 1)

    def test_model_can_correct_fixture_expectation_before_first_freeze(self):
        bad = f.CODE.replace("value(), 4", "value(), 2")
        result = self.run_replies([f.reply(self.candidate(test_code=bad)),
            lambda m: review(m, "revise_acceptance", f.CODE), review])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(json.loads((self.root / "state/frozen-acceptance.json").read_text())["spec"]["test_code"], f.CODE)

    def test_unrunnable_trial_cannot_be_frozen_or_execute_retained_steps(self):
        bad = "import missing_setup_module\n" + f.CODE
        result = self.run_replies([f.reply(self.candidate(test_code=bad)), review,
            f.reply({"kind": "blocked", "note": "Need a valid test setup"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())
        self.assertFalse((self.root / "state/frozen-acceptance.json").exists())

    def test_skipped_trial_is_not_runnable_acceptance(self):
        code = f.CODE.replace("    def test_value", "    @unittest.skip('invalid')\n    def test_value")
        result = self.run_replies([f.reply(self.candidate(test_code=code)), review,
            f.reply({"kind": "blocked", "note": "Skipped test cannot validate the requirement"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_zero_case_trial_is_not_frozen(self):
        code = f.CODE + "\ndef load_tests(loader, tests, pattern):\n    return unittest.TestSuite()\n"
        result = self.run_replies([f.reply(self.candidate(test_code=code)), review,
            f.reply({"kind": "blocked", "note": "No real cases ran"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertFalse(self.state().get("verification_spec"))
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_fixed_user_command_does_not_need_generated_test_review(self):
        self.model = ScriptModel([f.reply(self.candidate())])
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state")
        result = self.engine.run([], "return 4", StepBudget(max_steps=4),
                                 verify_command=self.shell("from sample import value; assert value() == 4")["command"])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertFalse(self.state().get("acceptance_trials"))

    def test_completed_baseline_result_tampering_stops_before_confirmation(self):
        interrupt, error = self.crash("acceptance_trial_finished")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate())])
        result_ref = self.state()["acceptance_trial"]["baseline_result"]
        (self.root / "state" / result_ref["record"]).write_text('{"status":"passed"}')
        with self.assertRaisesRegex(Exception, "Acceptance record changed"):
            self.run_replies([review])
        self.assertFalse(self.model.calls)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_runnable_behavior_failure_cannot_open_frozen_revision(self):
        result = self.run_replies([f.reply(self.candidate(2)), review,
            lambda m: review(m, "revise_acceptance", f.CODE.replace("value(), 4", "value(), 2")),
            f.reply({"kind": "blocked", "note": "Repair actual source behavior"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertIn("Runnable behavior failed", self.model.calls[-1][-1].content)
        self.assertEqual(len(self.state()["acceptance_versions"]), 1)

    def test_frozen_setup_revision_retains_behavior_and_compares_both_versions(self):
        code = "from sample import value\nif value() == 2:\n    raise RuntimeError('setup broken')\n" + f.CODE
        result = self.run_replies([f.reply(self.candidate(2, test_code=code)), review,
            lambda m: review(m, "revise_acceptance", f.CODE), review,
            f.reply(self.candidate(test_code=""))])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(self.state()["acceptance_versions"]), 2)
        compare = result["verification"]["acceptance_comparison"]
        data = json.loads((self.root / "state" / compare["record"][8:]).read_text())
        self.assertEqual(len(data["versions"]), 2)
        self.assertTrue(all(row["candidate"]["status"] == "passed" for row in data["versions"]))

    def test_new_pass_does_not_erase_old_runnable_failure(self):
        code = "from sample import value\nif value() != 0:\n    raise RuntimeError('old setup conflict')\n" + f.CODE
        result = self.run_replies([f.reply(self.candidate(2, test_code=code)), review,
            lambda m: review(m, "revise_acceptance", f.CODE), review,
            f.reply(self.candidate(test_code="")),
            f.reply({"kind": "blocked", "note": "Old runnable baseline still conflicts"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertEqual(result["rounds"][-1]["verification"]["status"], "failed")
        self.assertTrue(result["verification"]["acceptance_comparison"]["conflicts"])

    def test_frozen_revision_cannot_change_assertions_or_inputs(self):
        code = "from sample import value\nif value() == 2:\n    raise RuntimeError('setup broken')\n" + f.CODE
        result = self.run_replies([f.reply(self.candidate(2, test_code=code)), review,
            lambda m: review(m, "revise_acceptance", f.CODE.replace("value(), 4", "value(), 2")),
            f.reply({"kind": "blocked", "note": "Changing expected behavior is not a setup repair"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertIn("must remain unchanged", self.model.calls[-1][-1].content)

    def test_unknown_trial_execution_is_not_replayed(self):
        interrupt, error = self.crash("acceptance_trial_started")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate())])
        with patch("agent.acceptance.revisions.run_report", side_effect=AssertionError("no replay")):
            result = self.run_replies([])
        self.assertEqual(result["status"], "execution_unknown", result)
        self.assertFalse(self.model.calls)

    def test_completed_freeze_resume_releases_original_edits_once(self):
        interrupt, error = self.crash("acceptance_frozen")
        plan = self.candidate()
        plan["steps"].insert(0, self.shell("from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1 if p.exists() else 1))", "count"))
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(plan), review])
        self.assertFalse((self.repo / "counter").exists())
        result = self.run_replies([])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual((self.repo / "counter").read_text(), "1")
        self.assertEqual(result["calls_this_invocation"], 0)

    def test_changed_trial_record_is_rejected_before_model_or_execution(self):
        interrupt, error = self.crash("acceptance_trial_finished")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate())])
        (self.root / "state" / self.state()["acceptance_trial"]["ref"]["record"]).write_text("{}")
        with self.assertRaisesRegex(Exception, "Acceptance record changed"):
            self.run_replies([])
        self.assertFalse(self.model.calls)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_trial_cannot_modify_retained_original_or_live_workspace(self):
        code = "from pathlib import Path\nPath('sample.py').write_text('def value():\\n    return 4\\n')\n" + f.CODE
        result = self.run_replies([f.reply(self.candidate(test_code=code)), review,
            f.reply({"kind": "blocked", "note": "Trial modified its scratch source"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())
        self.assertIn("return 0", (self.engine.baseline.root / "sample.py").read_text())

    def test_review_tools_are_fixed_and_revisions_have_no_work_steps(self):
        result = self.run_replies([f.reply(self.candidate()), review])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertTrue(all(tools == self.model.tools[0] for tools in self.model.tools))
        for tool in self.model.tools[0]:
            self.assertTrue(tool["function"]["strict"])
        revise = next(t for t in self.model.tools[0] if t["function"]["name"] == "revise_acceptance")
        self.assertEqual(set(revise["function"]["parameters"]["properties"]), {"trial_id", "reason", "test_code", "test_driver"})
        self.assertNotEqual(behavior(f.CODE), behavior(f.CODE.replace("value(), 4", "value(), 2")))

    def test_revision_resource_is_in_restore_fingerprint(self):
        original = code_key()
        from agent.settings import PROJECT_ROOT
        (self.root / "resources").mkdir()
        p = self.root / "resources/acceptance-review.schema.json"
        p.write_bytes((PROJECT_ROOT / "resources/acceptance-review.schema.json").read_bytes() + b"\n")

        class Root:
            def __truediv__(self, name):
                return p if name == "resources/acceptance-review.schema.json" else PROJECT_ROOT / name
        with patch("agent.storage.recovery.PROJECT_ROOT", Root()):
            self.assertNotEqual(code_key(), original)


if __name__ == "__main__":
    unittest.main()
