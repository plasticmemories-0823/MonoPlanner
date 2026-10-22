"""通过真实任务协议验证边界词典、读取凭据及验收前置条件。"""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, messages_from_dict

from agent.model.payload import wire_input
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from support import interrupt_after_persist
import test_task_runner as fixtures


def lookup_reply(keys=None):
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "get_boundary_hints", "id": "boundary-call",
            "args": {"keys": ["general"] if keys is None else keys},
        }],
        usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    )


class HandbookModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return fixtures.confirmation(messages) or next(self.replies)


class BoundaryHintsTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    candidate = fixtures.TaskRunnerTests.candidate
    shell = fixtures.TaskRunnerTests.shell

    def runner(self, replies, **options):
        self.model = HandbookModel(replies)
        self.engine = TaskRunner(
            self.model, self.env, state_dir=self.root / "state", **options,
        )
        return self.engine

    def run_replies(self, replies, **options):
        runner = self.runner(replies, **options)
        return runner.run([], "return 4", StepBudget(max_steps=12))

    def test_only_requested_cards_enter_model_context_before_real_acceptance(self):
        from agent.acceptance.boundaries import lookup

        result = self.run_replies([
            lookup_reply(["general", "collections"]), fixtures.reply(self.candidate()),
        ])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["planner_calls"], 3)
        returned = json.loads(self.model.calls[1][-1].content)
        self.assertEqual(list(returned["entries"]), ["general", "collections"])
        self.assertNotIn(lookup(["concurrency"])["entries"]["concurrency"]["hints"][0],
                         str(self.model.calls[1]))
        self.assertEqual(self.engine.state["plan_count"], 1)
        self.assertTrue(self.engine.state["boundary_reads"]["general"])

    def test_candidate_without_general_is_rejected_before_trial_or_edits(self):
        result = self.run_replies([
            fixtures.reply(self.candidate()),
            fixtures.reply({"kind": "blocked", "note": "Lookup prerequisite not met"}),
        ])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertIn("get_boundary_hints", str(self.model.calls[1]))
        self.assertFalse(self.engine.state.get("acceptance_trials"))
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_other_category_does_not_satisfy_general_requirement(self):
        result = self.run_replies([
            lookup_reply(["math"]), fixtures.reply(self.candidate()),
            fixtures.reply({"kind": "blocked", "note": "General card still missing"}),
        ])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertFalse(self.engine.state.get("acceptance_trials"))

    def test_index_is_short_english_and_strict_schema_contains_no_card_bodies(self):
        from agent.acceptance.boundaries import lookup
        from agent.model.tools import get_boundary_hints

        tool = wire_input([], [get_boundary_hints])["tools"][0]["function"]
        params = tool["parameters"]
        self.assertEqual(params["required"], ["keys"])
        self.assertFalse(params["additionalProperties"])
        keys = params["properties"]["keys"]["items"]["enum"]
        self.assertEqual(len(keys), 12)
        self.assertEqual(len(keys), len(set(keys)))
        serialized = json.dumps(tool)
        self.assertNotIn("minItems", serialized)
        self.assertNotIn("maxItems", serialized)
        for key, entry in lookup(keys)["entries"].items():
            self.assertIn(key, tool["description"])
            self.assertGreaterEqual(len(entry["hints"]), 4)
            self.assertLessEqual(len(entry["hints"]), 6)
            for hint in entry["hints"]:
                self.assertTrue(hint.isascii())
                self.assertNotIn(hint, serialized)

    def test_invalid_or_untrusted_keys_are_rejected_without_partial_read(self):
        from agent.acceptance.boundaries import Lookup

        for keys in ([], ["unknown"], ["../hidden-tests"], ["general", "general"], "general"):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                Lookup.model_validate({"keys": keys})
        with self.assertRaises(ValueError):
            Lookup.model_validate({"keys": ["general"], "path": "private-answer"})

    def test_fixed_user_command_and_inspection_do_not_require_lookup(self):
        result = self.runner([fixtures.reply(self.candidate(test_code=""))]).run(
            [], "return 4", StepBudget(max_steps=3),
            verify_command=fixtures.python_command("from sample import value; assert value() == 4"),
        )
        self.assertEqual(result["status"], "verified_success", result)
        self.assertNotIn("boundary_reads", self.engine.state)

    def test_saved_lookup_response_recovers_without_new_model_call(self):
        interrupt, interrupted = interrupt_after_persist("model_response_received")
        with interrupt, self.assertRaises(interrupted):
            self.run_replies([lookup_reply()])
        result = self.run_replies([fixtures.reply(self.candidate())])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["planner_calls"], 3)
        self.assertEqual(result["calls_this_invocation"], 2)

    def test_committed_lookup_survives_resume_and_preserves_tool_pairing(self):
        interrupt, interrupted = interrupt_after_persist("boundary_hints_read")
        with interrupt, self.assertRaises(interrupted):
            self.run_replies([lookup_reply()])
        result = self.run_replies([fixtures.reply(self.candidate())])
        self.assertEqual(result["status"], "verified_success", result)
        history = messages_from_dict(self.engine.state["history"])
        calls = [c for m in history for c in getattr(m, "tool_calls", [])
                 if c["name"] == "get_boundary_hints"]
        replies = [m for m in history if getattr(m, "name", None) == "get_boundary_hints"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(replies), 1)
        self.assertEqual(calls[0]["id"], replies[0].tool_call_id)

    def test_altered_lookup_receipt_cannot_authorize_acceptance(self):
        from agent.acceptance.boundaries import require_general
        from agent.model.errors import ModelRequestNotSent

        self.run_replies([lookup_reply()], yield_boundary=True)
        ref = self.engine.state["boundary_reads"]["general"]
        (self.engine.root / ref["record"]).write_text("{}")
        with self.assertRaises(ModelRequestNotSent):
            require_general(self.engine)

    def test_catalog_is_in_source_identity(self):
        from agent.acceptance.boundaries import CATALOG_PATH
        from agent.storage.recovery import code_key

        original = code_key()
        read_bytes = Path.read_bytes

        def changed(path):
            raw = read_bytes(path)
            return raw + b"\n" if path == CATALOG_PATH else raw

        with patch.object(Path, "read_bytes", changed):
            self.assertNotEqual(code_key(), original)

    def test_record_saved_before_checkpoint_is_reused_without_repeating_lookup(self):
        from agent.acceptance import boundaries

        class PowerLoss(BaseException):
            pass

        save = boundaries.save

        def interrupted_save(path, data):
            save(path, data)
            raise PowerLoss()

        with patch.object(boundaries, "save", interrupted_save), self.assertRaises(PowerLoss):
            self.run_replies([lookup_reply()])
        with patch.object(boundaries, "lookup", side_effect=AssertionError("must reuse record")):
            result = self.run_replies([fixtures.reply(self.candidate())])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["planner_calls"], 3)

    def test_storage_failure_stops_without_acceptance_or_more_requests(self):
        with patch("agent.acceptance.boundaries.save", side_effect=OSError("disk full")):
            result = self.run_replies([lookup_reply(), fixtures.reply(self.candidate())])
        self.assertEqual(result["status"], "runtime_error", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertNotIn("boundary_reads", self.engine.state)
        self.assertFalse(self.engine.state.get("acceptance_trials"))
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_catalog_change_stops_lookup_instead_of_returning_empty_hints(self):
        from agent.acceptance import boundaries

        with patch.object(boundaries, "load_catalog", return_value=({}, "changed")):
            result = self.run_replies([lookup_reply(), fixtures.reply(self.candidate())])
        self.assertEqual(result["status"], "state_conflict", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertNotIn("boundary_reads", self.engine.state)

    def test_invalid_and_mixed_calls_are_paired_rejections_not_partial_lookups(self):
        mixed = lookup_reply()
        mixed.tool_calls.extend(fixtures.reply(self.candidate()).tool_calls)
        result = self.run_replies([
            lookup_reply(["private-answer"]), mixed,
            fixtures.reply({"kind": "blocked", "note": "Do not run malformed requests"}),
        ])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertNotIn("boundary_reads", self.engine.state)
        self.assertFalse(self.engine.state.get("acceptance_trials"))
        history = messages_from_dict(self.engine.state["history"])
        requested = sum(len(getattr(m, "tool_calls", [])) for m in history)
        returned = sum(m.type == "tool" for m in history)
        self.assertEqual(requested, returned)

    def test_lookup_round_uses_the_existing_model_budget(self):
        runner = self.runner([lookup_reply(), fixtures.reply(self.candidate())])
        result = runner.run([], "return 4", StepBudget(max_steps=1))
        self.assertEqual(result["status"], "budget_exhausted", result)
        self.assertEqual(result["planner_calls"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 20)
        self.assertFalse(self.engine.state.get("acceptance_trials"))

    def test_tool_definition_and_sent_prefix_stay_fixed_during_lookup_and_acceptance(self):
        self.run_replies([lookup_reply(), fixtures.reply(self.candidate())])
        requests = [wire_input(messages, self.model.tools) for messages in self.model.calls]
        for previous, current in zip(requests, requests[1:]):
            self.assertEqual(previous["tools"], current["tools"])
            self.assertEqual(previous["messages"], current["messages"][:len(previous["messages"])])

    def test_receipt_is_not_a_correctness_certificate(self):
        result = self.run_replies([
            lookup_reply(), fixtures.reply(self.candidate(2)),
            fixtures.reply({"kind": "blocked", "note": "Real test still fails"}),
        ])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertEqual(result["rounds"][0]["verification"]["status"], "failed")

    def test_repeated_query_returns_full_selected_cards_for_context_recovery(self):
        from agent.acceptance.boundaries import lookup

        first = lookup(["general", "math"])
        first["entries"]["general"]["hints"][0] = "changed caller copy"
        second = lookup(["general", "math"])
        self.assertNotIn("changed caller copy", str(second))
        self.assertEqual(len(second["entries"]["general"]["hints"]), 5)
