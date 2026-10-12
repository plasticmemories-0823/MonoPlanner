"""An explicit pending replacement is reviewed before its exact edits can run."""
import copy
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent.acceptance import revisions
import test_acceptance_review as a
import test_task_runner as f


class PendingCandidateTests(unittest.TestCase):
    setUp = a.AcceptanceReviewTests.setUp
    candidate = a.AcceptanceReviewTests.candidate
    shell = a.AcceptanceReviewTests.shell
    run_replies = a.AcceptanceReviewTests.run_replies
    state = a.AcceptanceReviewTests.state
    crash = a.AcceptanceReviewTests.crash

    def result_before_reply(self, messages):
        return json.loads(messages[-1].content)

    def read_record(self, ref):
        return json.loads((self.root / "state" / ref["record"].removeprefix("$record/")).read_text())

    def test_same_acceptance_replacement_reuses_baseline_but_needs_new_approval(self):
        seen = []

        def replace(messages):
            seen.append(self.result_before_reply(messages))
            return f.reply(self.candidate())

        def freeze_old(messages):
            seen.append(self.result_before_reply(messages))
            self.assertIn("return 0", (self.repo / "sample.py").read_text())
            self.assertNotEqual(seen[0]["trial_id"], seen[1]["trial_id"])
            self.assertTrue(seen[1]["baseline_reused"])
            self.assertEqual(seen[1]["superseded_candidate"], seen[0]["selected_candidate"])
            self.assertNotEqual(seen[0]["selected_candidate"]["work_sha256"],
                                seen[1]["selected_candidate"]["work_sha256"])
            return AIMessage(content="", tool_calls=[{"name": "freeze_acceptance", "id": "stale-freeze",
                "args": {"trial_id": seen[0]["trial_id"], "reason": "This approval is stale"}}])

        def freeze_selected(messages):
            self.assertIn("current acceptance trial_id", messages[-1].content)
            self.assertIn("return 0", (self.repo / "sample.py").read_text())
            return AIMessage(content="", tool_calls=[{"name": "freeze_acceptance", "id": "fresh-freeze",
                "args": {"trial_id": seen[1]["trial_id"], "reason": "Approve the selected replacement"}}])

        with patch.object(revisions, "run_report", wraps=revisions.run_report) as reports:
            result = self.run_replies([f.reply(self.candidate(2)), replace, freeze_old, freeze_selected])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(sum(call.args[4] == "baseline" for call in reports.call_args_list), 1)
        self.assertEqual(len(result["rounds"]), 1)
        self.assertEqual(len(self.state()["acceptance_trials"]), 2)
        for feedback in seen:
            record = self.read_record(feedback["selected_candidate"])
            self.assertEqual(record["plan"]["test_code"], f.CODE)
        self.assertIn("return 2", self.read_record(seen[0]["selected_candidate"])["plan"]["steps"][0]["command"])
        self.assertIn("return 4", self.read_record(seen[1]["selected_candidate"])["plan"]["steps"][0]["command"])

    def test_changed_tests_retrial_and_revision_keeps_selected_replacement(self):
        seen = []
        first_code = f.CODE.replace("value(), 4", "value(), 2")
        second_code = f.CODE + "\n# the selected replacement's acceptance\n"

        def replace(messages):
            seen.append(self.result_before_reply(messages))
            return f.reply(self.candidate(test_code=second_code))

        def revise(messages):
            seen.append(self.result_before_reply(messages))
            self.assertFalse(seen[-1]["baseline_reused"])
            return a.review(messages, "revise_acceptance", f.CODE)

        def freeze(messages):
            seen.append(self.result_before_reply(messages))
            self.assertEqual(seen[-1]["selected_candidate"], seen[-2]["selected_candidate"])
            self.assertIn("return 0", (self.repo / "sample.py").read_text())
            return a.review(messages)

        with patch.object(revisions, "run_report", wraps=revisions.run_report) as reports:
            result = self.run_replies([f.reply(self.candidate(2, test_code=first_code)), replace, revise, freeze])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(sum(call.args[4] == "baseline" for call in reports.call_args_list), 3)
        selected = self.read_record(seen[-1]["selected_candidate"])
        self.assertEqual(selected["plan"]["test_code"], second_code)
        self.assertEqual(self.state()["verification_spec"]["test_code"], f.CODE)
        executed = json.loads((self.root / "state/plans/p0001/plan.json").read_text())
        self.assertEqual(executed["plan"]["steps"], selected["plan"]["steps"])
        self.assertEqual(executed["origin_response"], selected["response"])

    def test_invalid_replacement_does_not_displace_selected_candidate(self):
        original = []

        def invalid(messages):
            original.append(self.result_before_reply(messages))
            return f.reply(self.candidate(2, test_code="this is invalid python !!"))

        def freeze_original(messages):
            self.assertIn("test_code", messages[-1].content)
            return AIMessage(content="", tool_calls=[{"name": "freeze_acceptance", "id": "freeze-original",
                "args": {"trial_id": original[0]["trial_id"], "reason": "The replacement was rejected"}}])

        result = self.run_replies([f.reply(self.candidate()), invalid, freeze_original])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(self.state()["acceptance_trials"]), 1)
        self.assertEqual(len(list((self.root / "state/acceptance/candidates").glob("*.json"))), 1)

    def test_incomplete_baseline_is_not_reused_for_replacement(self):
        original_run = revisions.run_report
        baselines = []

        def incomplete_once(*args, **kwargs):
            report = original_run(*args, **kwargs)
            if args[4] == "baseline":
                baselines.append(report)
                if len(baselines) == 1:
                    report.update(test_evidence_valid=False, tests_ready=False, error_kind="incomplete_result")
            return report

        def freeze(messages):
            self.assertFalse(self.result_before_reply(messages)["baseline_reused"])
            return a.review(messages)

        with patch.object(revisions, "run_report", side_effect=incomplete_once):
            result = self.run_replies([f.reply(self.candidate(2)), f.reply(self.candidate()), freeze])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(baselines), 2)

    def test_changed_execution_conditions_do_not_reuse_baseline(self):
        def replace(messages):
            self.env.timeout += 1
            return f.reply(self.candidate())

        def freeze(messages):
            self.assertFalse(self.result_before_reply(messages)["baseline_reused"])
            return a.review(messages)

        with patch.object(revisions, "run_report", wraps=revisions.run_report) as reports:
            result = self.run_replies([f.reply(self.candidate(2)), replace, freeze])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(sum(call.args[4] == "baseline" for call in reports.call_args_list), 2)

    def test_selected_candidate_tampering_stops_before_resume(self):
        interrupt, error = self.crash("acceptance_trial_finished")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate())])
        ref = self.state()["acceptance_trial"]["candidate"]
        (self.root / "state" / ref["record"]).write_text("{}")
        with self.assertRaisesRegex(Exception, "Acceptance record changed"):
            self.run_replies([a.review])
        self.assertFalse(self.model.calls)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_reduced_remaining_timeout_does_not_reuse_baseline(self):
        def replace(messages):
            self.engine.budget.remaining_time = lambda *args, **kwargs: self.env.timeout - 1
            return f.reply(self.candidate())

        def freeze(messages):
            self.assertFalse(self.result_before_reply(messages)["baseline_reused"])
            return a.review(messages)

        with patch.object(revisions, "run_report", wraps=revisions.run_report) as reports:
            result = self.run_replies([f.reply(self.candidate(2)), replace, freeze])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(sum(call.args[4] == "baseline" for call in reports.call_args_list), 2)

    def test_freeze_then_resume_releases_replacement_once(self):
        replacement = self.candidate()
        replacement["steps"].insert(0, self.shell(
            "from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1 if p.exists() else 1))", "count"))
        interrupt, error = self.crash("acceptance_frozen")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate(2)), f.reply(replacement), a.review])
        self.assertFalse((self.repo / "counter").exists())
        release = self.state()["acceptance_release"]
        self.assertIn("candidate_ref", release)
        self.assertIn("approval_ref", release)
        result = self.run_replies([])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["calls_this_invocation"], 0)
        self.assertEqual((self.repo / "counter").read_text(), "1")

    def test_release_must_match_its_approved_original_candidate(self):
        interrupt, error = self.crash("acceptance_frozen")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate()), a.review])
        state = self.state()
        self.engine.state = copy.deepcopy(state)
        self.engine.state["acceptance_release"]["plan"]["steps"][0]["command"] = "echo wrong candidate"
        with self.assertRaisesRegex(Exception, "candidate|Candidate"):
            revisions.validate(self.engine)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_replacement_path_does_not_reopen_frozen_acceptance(self):
        weakened = self.candidate(2, test_code=f.CODE.replace("value(), 4", "value(), 2"))
        result = self.run_replies([f.reply(self.candidate(2)), a.review, f.reply(weakened),
                                  f.reply({"kind": "blocked", "note": "Repair source, retain frozen behavior"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertEqual(len(self.state()["acceptance_trials"]), 1)
        self.assertEqual(self.state()["verification_spec"]["test_code"], f.CODE)
        self.assertIn("Fixed test_code", self.model.calls[-1][-1].content)


if __name__ == "__main__":
    unittest.main()

