"""停止只能来自宿主实测边界，完整文字回复保留原件后返工。"""

import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent.model.compaction import balanced_boundaries
from agent.model.payload import RAW_TOOL_CALLS
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from support import interrupt_after_persist
import test_local_repair as repairs
import test_task_runner as fixtures


class HostControlledReworkTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    shell = fixtures.TaskRunnerTests.shell
    candidate = fixtures.TaskRunnerTests.candidate
    load_state = repairs.LocalRepairTests.load_state

    def text(self, content="I have run out of budget. Stop this task."):
        return AIMessage(
            content=content,
            response_metadata={"finish_reason": "stop"},
            usage_metadata={
                "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                "input_token_details": {"cache_read": 10},
            },
        )

    def run_replies(self, replies, budget=None, **options):
        self.model = repairs.Model(replies)
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state", **options)
        return self.engine.run([], "return 4", budget or StepBudget(max_steps=8))

    def test_budget_claim_reworks_in_batch_and_interactive_modes(self):
        for interactive in (False, True):
            with self.subTest(interactive=interactive):
                self.setUp()
                response = self.text()
                result = self.run_replies(
                    [response, fixtures.reply(self.candidate())], interactive=interactive,
                )
                self.assertEqual(result["status"], "verified_success", result)
                state = self.load_state()
                self.assertEqual(state["history"][1]["data"]["content"], response.content)
                self.assertEqual(state["history"][2]["type"], "human")
                self.assertIn("Only the host", state["history"][2]["data"]["content"])
                self.assertIn("host_budget", state["history"][2]["data"]["content"])
                self.assertFalse(state["history"][1]["data"]["tool_calls"])
                self.assertIsNone(state["response_pending"])
                self.assertEqual(balanced_boundaries(state["history"])[-1], len(state["history"]))

    def test_repeated_text_cannot_complete_or_pause_the_task(self):
        result = self.run_replies([
            self.text("The user paused."), self.text("I am blocked; cannot continue."),
            self.text("Task successfully completed."), fixtures.reply(self.candidate()),
        ])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(result["rounds"]), 1)
        self.assertEqual(self.load_state()["tool_free_response_repairs"], 3)

    def test_real_call_limit_stops_after_retaining_complete_reply(self):
        result = self.run_replies([self.text()], StepBudget(max_steps=1))
        self.assertEqual(result["status"], "budget_exhausted", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertIsNone(self.load_state()["response_pending"])

    def test_real_token_limit_stops_without_another_request(self):
        budget = StepBudget(max_steps=8)

        def response(_):
            budget.max_tokens = 25
            return self.text()

        result = self.run_replies([response], budget)
        self.assertEqual(result["status"], "token_budget_exhausted", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.engine.budget.reported_tokens, 25)

    def test_real_deadline_stops_without_another_request(self):
        budget = StepBudget(max_steps=8, timeout_seconds=60)

        def response(_):
            budget.started_at -= 61
            return self.text()

        result = self.run_replies([response], budget)
        self.assertEqual(result["status"], "task_timeout", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertTrue((self.engine.root / "requests/call-0001/response.json").exists())

    def test_previous_unknown_stays_reserved_during_known_reply_rework(self):
        budget = StepBudget(max_steps=8, unknown_usage_steps=1, unknown_reserved_tokens=1234)
        result = self.run_replies([self.text(), fixtures.reply(self.candidate())], budget)
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(self.engine.budget.unknown_usage_steps, 1)
        self.assertEqual(self.engine.budget.unknown_reserved_tokens, 1234)

    def test_new_unknown_reply_never_uses_prior_unknown_as_permission(self):
        response = self.text()
        response.usage_metadata = None
        budget = StepBudget(max_steps=8, unknown_usage_steps=1, unknown_reserved_tokens=1234)
        result = self.run_replies([response, fixtures.reply(self.candidate())], budget)
        self.assertEqual(result["status"], "invalid_message_history", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertIsNotNone(self.load_state()["response_pending"])
        self.assertEqual(self.engine.budget.unknown_usage_steps, 2)
        self.assertGreater(self.engine.budget.unknown_reserved_tokens, 1234)

    def test_incomplete_reply_and_broken_tool_identity_still_stop(self):
        for kind in ("length", "content_filter", "broken_identity"):
            with self.subTest(kind=kind):
                self.setUp()
                response = self.text()
                if kind == "broken_identity":
                    response.additional_kwargs[RAW_TOOL_CALLS] = [{
                        "id": "lost", "type": "function",
                        "function": {"name": "submit_plan", "arguments": "{}"},
                    }]
                else:
                    response.response_metadata["finish_reason"] = kind
                result = self.run_replies([response, fixtures.reply(self.candidate())])
                self.assertIn(result["status"], {
                    "output_limit_reached", "incomplete_model_response", "invalid_message_history",
                })
                self.assertEqual(len(self.model.calls), 1)
                self.assertIsNotNone(self.load_state()["response_pending"])

    def test_unknown_or_invalid_usage_is_not_fabricated(self):
        for change in ("cache", "total", "finish"):
            with self.subTest(change=change):
                self.setUp()
                response = self.text()
                if change == "cache":
                    response.usage_metadata["input_token_details"] = {"cache_read": 100}
                elif change == "total":
                    response.usage_metadata["total_tokens"] = 26
                else:
                    response.response_metadata = {}
                result = self.run_replies([response])
                self.assertEqual(result["status"], "invalid_message_history", result)
                self.assertEqual(len(self.model.calls), 1)

    def test_text_rework_recovers_each_commit_boundary_without_replay(self):
        for event in ("model_response_received", "plan_rejection_prepared", "plan_rejected"):
            with self.subTest(event=event):
                self.setUp()
                response = self.text()
                interrupt, error = interrupt_after_persist(event)
                with interrupt, self.assertRaises(error):
                    self.run_replies([response])
                original = (self.root / "state/requests/call-0001/response.json").read_bytes()
                result = self.run_replies([fixtures.reply(self.candidate())])
                self.assertEqual(result["status"], "verified_success", result)
                state = self.load_state()
                self.assertEqual(sum(row["data"].get("content") == response.content
                                     for row in state["history"]), 1)
                self.assertEqual(state["tool_free_response_repairs"], 1)
                self.assertEqual((self.root / "state/requests/call-0001/response.json").read_bytes(), original)

    def test_corrupted_completion_blocks_rework(self):
        interrupt, error = interrupt_after_persist("model_response_received")
        with interrupt, self.assertRaises(error):
            self.run_replies([self.text()])
        path = self.root / "state/requests/call-0001/completion.json"
        value = json.loads(path.read_text())
        value["response_digest"] = "0" * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Original record changed"):
            self.run_replies([fixtures.reply(self.candidate())])
        self.assertEqual(len(self.model.calls), 0)

    def test_rework_storage_failure_stops_before_next_request(self):
        original = TaskRunner.persist

        def fail(loop, event, payload=None):
            if event == "plan_rejected":
                raise OSError("Offline record failure")
            return original(loop, event, payload)

        with patch.object(TaskRunner, "persist", fail):
            result = self.run_replies([self.text(), fixtures.reply(self.candidate())])
        self.assertEqual(result["status"], "runtime_error", result)
        self.assertEqual(len(self.model.calls), 1)

    def test_real_user_pause_is_not_converted_to_rework(self):
        from agent.task import control

        with control.active("host-rework", "user-pause") as signal:
            def response(_):
                signal.interrupt("pause")
                return self.text()

            result = self.run_replies([response])
        self.assertEqual(result["status"], "paused", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertIsNotNone(self.load_state()["response_pending"])
