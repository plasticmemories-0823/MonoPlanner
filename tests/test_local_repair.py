"""局部修正保留真实验收和持久化的执行边界。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, messages_from_dict

from agent.model.client import create_model
from agent.model.errors import ModelRequestNotSent
from agent.model.payload import RAW_TOOL_CALLS, wire_input
from agent.model.tools import WORK_TOOLS
from agent.storage.history import LocalHistory
from agent.storage.recovery import code_key
from agent.task import repair
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from support import interrupt_after_persist, boundary_aware_reply
import test_task_runner as f


def correction(messages, edits, identifier=None):
    notice = json.loads(messages[-1].content.split("] ", 1)[1].split("\n", 1)[0])
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "repair_work",
                "id": "repair-call",
                "args": {"draft_id": identifier or notice["draft_id"], "edits": edits},
            }
        ],
    )


def edit(path, before, after, operation="set"):
    return {"path": path, "operation": operation, "before": before, "after": after}


class Model:
    def __init__(self, replies):
        self.replies, self.calls, self.tools = iter(replies), [], []

    def bind_tools(self, tools):
        self.tools.append(wire_input([], tools)["tools"])
        return self

    def invoke(self, messages):
        self.calls.append(messages)

        def produce():
            review = f.confirmation(messages)
            if review is not None:
                return review
            reply = next(self.replies)
            return reply(messages) if callable(reply) else reply

        return boundary_aware_reply(self, messages, produce)


class LocalRepairTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    candidate = f.TaskRunnerTests.candidate
    shell = f.TaskRunnerTests.shell

    def run_replies(self, replies, **options):
        self.model = Model(replies)
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state", **options)
        return self.engine.run([], "return 4", StepBudget(max_steps=8))

    def load_state(self):
        store = LocalHistory(self.root / "state")
        try:
            with store.lease():
                return store.load()
        finally:
            store.close()

    def crash_on(self, event_name):
        return interrupt_after_persist(event_name)

    def test_extra_field_repair_executes_original_code_and_keeps_full_draft_local(self):
        plan = self.candidate()
        plan["steps"][0]["query_placeholder"] = ""
        plan["steps"][0]["command"] += " # UNCHANGED_WORK_NOT_IN_CORRECTION"
        result = self.run_replies(
            [
                f.reply(plan),
                lambda m: correction(m, [edit("/steps/0/query_placeholder", '""', "", "remove")]),
            ]
        )
        self.assertEqual(result["status"], "verified_success", result)
        notice = self.model.calls[1][-1].content
        self.assertNotIn("UNCHANGED_WORK_NOT_IN_CORRECTION", notice)
        self.assertNotIn(f.CODE, notice)
        draft = json.loads(next((self.root / "state/drafts").glob("*.json")).read_text())
        self.assertEqual(draft["value"], plan)
        accepted = json.loads((self.root / "state/plans/p0001/plan.json").read_text())
        self.assertEqual(accepted["plan"]["steps"][0]["command"], plan["steps"][0]["command"])
        self.assertEqual(accepted["plan"]["test_code"], f.CODE)
        self.assertEqual(
            accepted["origin_response"]["data"]["tool_calls"][0]["name"], "repair_work"
        )
        self.assertEqual(accepted["response"]["data"]["tool_calls"][0]["name"], "freeze_acceptance")
        state = self.load_state()
        self.assertEqual(state["history"][-1]["data"]["tool_call_id"], "confirm-call")
        self.assertEqual(state["history"][-1]["data"]["name"], "freeze_acceptance")
        self.assertEqual(self.model.tools[0], self.model.tools[1])
        self.assertNotIn("repair_draft", state)

    def repaired_inspection(self):
        plan = {"kind": "inspect", "note": "Inspect the public source", "steps": [
            {"id": "read", "op": "read", "path": "sample.py", "expected_exit": []},
        ]}
        saved = {}

        def correct(messages):
            notice = json.loads(messages[-1].content.split("] ", 1)[1].split("\n", 1)[0])
            saved["draft_id"] = notice["draft_id"]
            return correction(messages, [edit("/steps/0/expected_exit", "[]", "[0]")])

        return plan, saved, correct

    def test_consumed_inspection_draft_stays_consumed_after_resume(self):
        plan, saved, correct = self.repaired_inspection()
        result = self.run_replies([f.reply(plan), correct], pause_after_steps=1)
        self.assertEqual(result["status"], "paused", result)
        state = self.load_state()
        self.assertNotIn("repair_draft", state)
        self.assertTrue((self.root / "state/drafts" / (saved["draft_id"] + ".json")).is_file())
        result = self.run_replies([f.reply(self.candidate())])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(self.load_state()["plan_count"], 2)
        self.assertEqual(len(list((self.root / "state/plans/p0001").glob("*/result.json"))), 1)

    def test_corrected_candidate_draft_is_consumed_when_trial_is_committed(self):
        plan = self.candidate()
        plan["extra"] = 1
        interrupt, error = self.crash_on("acceptance_trial_finished")
        with interrupt, self.assertRaises(error):
            self.run_replies([
                f.reply(plan),
                lambda m: correction(m, [edit("/extra", "1", "", "remove")]),
            ])
        state = self.load_state()
        self.assertTrue(state.get("acceptance_trial"))
        self.assertNotIn("repair_draft", state)
        self.assertEqual(len(list((self.root / "state/drafts").glob("*.json"))), 1)
        result = self.run_replies([])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["calls_this_invocation"], 1)

    def test_protocol_error_after_consumed_draft_cannot_resurrect_old_work(self):
        plan, saved, correct = self.repaired_inspection()
        multiple = AIMessage(content="", tool_calls=[
            {"id": "lookup", "name": "get_boundary_hints", "args": {"keys": ["general"]}},
            {"id": "new-work", "name": "submit_plan", "args": plan},
        ])

        def stale_repair(messages):
            for message in messages[-2:]:
                self.assertIn("exactly one valid work tool call", message.content)
                self.assertNotIn("allowed_paths", message.content)
                self.assertNotIn(saved["draft_id"], message.content)
            return AIMessage(content="", tool_calls=[{
                "id": "stale", "name": "repair_work", "args": {
                    "draft_id": saved["draft_id"],
                    "edits": [edit("/steps/0/expected_exit", "[]", "[0]")],
                },
            }])

        def recover(messages):
            self.assertIn("requires the current draft_id", messages[-1].content)
            self.assertNotIn("allowed_paths", messages[-1].content)
            return f.reply(self.candidate())

        result = self.run_replies([f.reply(plan), correct, multiple, stale_repair, recover])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(self.load_state()["plan_count"], 2)
        self.assertNotIn("repair_draft", self.load_state())

    def test_protocol_error_does_not_borrow_fields_from_unconsumed_draft(self):
        plan = self.candidate()
        plan["extra"] = 1
        saved = {}

        def invalid_pair(messages):
            notice = json.loads(messages[-1].content.split("] ", 1)[1].split("\n", 1)[0])
            saved["draft_id"] = notice["draft_id"]
            return AIMessage(content="", tool_calls=[
                {"id": "one", "name": "get_boundary_hints", "args": {"keys": ["general"]}},
                {"id": "two", "name": "submit_plan", "args": plan},
            ])

        def correct(messages):
            self.assertIn("exactly one valid work tool call", messages[-1].content)
            self.assertNotIn("allowed_paths", messages[-1].content)
            return AIMessage(content="", tool_calls=[{
                "id": "actual-repair", "name": "repair_work", "args": {
                    "draft_id": saved["draft_id"], "edits": [edit("/extra", "1", "", "remove")],
                },
            }])

        result = self.run_replies([f.reply(plan), invalid_pair, correct])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(self.load_state()["plan_count"], 1)

    def test_wrong_draft_id_does_not_advertise_another_drafts_fields(self):
        plan = self.candidate()
        plan["extra"] = 1
        saved = {}

        def wrong(messages):
            reply = correction(messages, [edit("/extra", "1", "", "remove")])
            saved.update(deepcopy(reply.tool_calls[0]["args"]))
            reply.tool_calls[0]["args"]["draft_id"] = "unrelated-draft"
            return reply

        def correct(messages):
            self.assertIn("requires the current draft_id", messages[-1].content)
            self.assertNotIn("allowed_paths", messages[-1].content)
            return AIMessage(content="", tool_calls=[{
                "id": "current-repair", "name": "repair_work", "args": saved,
            }])

        result = self.run_replies([f.reply(plan), wrong, correct])
        self.assertEqual(result["status"], "verified_success", result)

    def test_duplicate_inspection_still_stops_without_reexecution(self):
        plan = {"kind": "inspect", "note": "Read the same source", "steps": [
            {"id": "read", "op": "read", "path": "sample.py"},
        ]}
        result = self.run_replies([f.reply(plan), f.reply(plan)])
        self.assertEqual(result["status"], "stalled", result)
        self.assertEqual(self.load_state()["plan_count"], 1)
        self.assertEqual(len(list((self.root / "state/plans/p0001").glob("*/result.json"))), 1)

    def test_missing_field_is_explicitly_added(self):
        plan = self.candidate()
        del plan["note"]
        result = self.run_replies(
            [
                f.reply(plan),
                lambda m: correction(m, [edit("/note", "", json.dumps("落实已给出的修改，中文"))]),
            ]
        )
        self.assertEqual(result["status"], "verified_success", result)

    def draft_read_then_correction(self):
        plan = self.candidate()
        plan["extra"] = 1
        draft = {}

        def read_draft(messages):
            notice = json.loads(messages[-1].content.split("] ", 1)[1].split("\n", 1)[0])
            draft["id"] = notice["draft_id"]
            return f.reply(
                {
                    "kind": "inspect",
                    "note": "Read the rejected original before correcting it",
                    "steps": [
                        {
                            "id": "draft",
                            "op": "read",
                            "path": notice["original_record"],
                            "offset": 0,
                            "limit": 200,
                        }
                    ],
                }
            )

        def correct_draft(messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "repair_work",
                        "id": "after-read",
                        "args": {
                            "draft_id": draft["id"],
                            "edits": [edit("/extra", "1", "", "remove")],
                        },
                    }
                ],
            )

        return plan, draft, read_draft, correct_draft

    def test_reading_rejected_draft_retains_repair_target(self):
        plan, draft, read_draft, correct_draft = self.draft_read_then_correction()
        result = self.run_replies([f.reply(plan), read_draft, correct_draft])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["planner_calls"], 5)
        read_result = json.loads(self.model.calls[3][-1].content)["steps"][0]
        self.assertEqual(read_result["status"], "succeeded")
        self.assertIn('"extra": 1', read_result["data"]["text"])
        accepted = json.loads((self.root / "state/plans/p0002/plan.json").read_text())
        expected = f.WorkRequest.model_validate({k: v for k, v in plan.items() if k != "extra"})
        self.assertEqual(accepted["plan"]["steps"], expected.model_dump()["steps"])
        self.assertEqual(accepted["plan"]["test_code"], plan["test_code"])
        state = self.load_state()
        self.assertEqual(accepted["origin_response"]["data"]["tool_calls"][0]["id"], "after-read")
        self.assertEqual(state["history"][-1]["data"]["tool_call_id"], "confirm-call")
        self.assertNotIn("repair_draft", state)

    def test_resume_after_reading_draft_does_not_repeat_read_or_request(self):
        plan, draft, read_draft, correct_draft = self.draft_read_then_correction()
        result = self.run_replies([f.reply(plan), read_draft], pause_after_steps=1)
        self.assertEqual(result["status"], "paused", result)
        self.assertEqual(self.load_state().get("repair_draft", {}).get("id"), draft["id"])
        result = self.run_replies([correct_draft])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["calls_this_invocation"], 2)
        self.assertEqual(result["planner_calls"], 5)
        self.assertEqual(
            len(list((self.root / "state/plans/p0001").glob("p0001-*/result.json"))), 1
        )

    def test_invalid_repair_keeps_original_error_and_then_can_correct(self):
        plan = self.candidate()
        plan["steps"][0]["query_placeholder"] = ""
        result = self.run_replies(
            [
                f.reply(plan),
                lambda m: correction(
                    m, [edit("/steps/0/query_placeholder", '"wrong"', "", "remove")]
                ),
                lambda m: correction(m, [edit("/steps/0/query_placeholder", '""', "", "remove")]),
            ]
        )
        self.assertEqual(result["status"], "verified_success", result)
        self.assertIn("Original value does not match", self.model.calls[3][-1].content)
        self.assertIn("query_placeholder", self.model.calls[3][-1].content)
        self.assertIn("original_error", self.model.calls[3][-1].content)
        self.assertEqual(len(list((self.root / "state/drafts").glob("*.json"))), 1)

    def test_modified_value_is_fully_revalidated_before_execution(self):
        plan = self.candidate()
        plan["steps"][0]["op"] = "unknown"
        result = self.run_replies(
            [
                f.reply(plan),
                lambda m: correction(m, [edit("/steps/0/op", '"unknown"', '"still_unknown"')]),
                lambda m: correction(m, [edit("/steps/0/op", '"still_unknown"', '"shell"')]),
            ]
        )
        self.assertEqual(result["status"], "verified_success", result)
        drafts = list((self.root / "state/drafts").glob("*.json"))
        self.assertEqual(len(drafts), 2)
        ids = [
            json.loads(m[-1].content.split("] ", 1)[1].split("\n", 1)[0])["draft_id"]
            for m in self.model.calls[1:]
            if m[-1].content.startswith("[Plan rejected")
        ]
        self.assertNotEqual(*ids)

    def test_real_assertion_failure_is_not_repaired_away(self):
        plan = self.candidate(value=2)
        plan["steps"][0]["extra"] = ""
        result = self.run_replies(
            [
                f.reply(plan),
                lambda m: correction(m, [edit("/steps/0/extra", '""', "", "remove")]),
                f.reply({"kind": "blocked", "note": "actual failure"}),
            ]
        )
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertEqual(result["rounds"][0]["verification"]["status"], "failed")
        self.assertIn("AssertionError", str(self.model.calls[4]))
        self.assertEqual(
            json.loads(self.model.calls[4][-1].content)["outcome"], "verification_failed"
        )

    def test_frozen_tests_cannot_be_replaced_by_field_correction(self):
        weak = self.candidate(test_code=f.CODE.replace("value(), 4", "value(), 2"))
        replacements = lambda m: correction(
            m,
            [
                edit(
                    "/test_code",
                    json.dumps(weak["test_code"]),
                    json.dumps(weak["test_code"] + "# still weaker\n"),
                )
            ],
        )
        result = self.run_replies([f.reply(self.candidate(2)), f.reply(weak)] + [replacements] * 6)
        self.assertEqual(result["status"], "budget_exhausted", result)
        self.assertEqual(
            json.loads((self.root / "state/frozen-acceptance.json").read_text())["spec"][
                "test_code"
            ],
            f.CODE,
        )
        self.assertEqual(len(result["rounds"]), 1)

    def test_c_output_request_uses_the_same_generated_acceptance_path(self):
        self.model = Model([f.reply(self.candidate())])
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state")
        result = self.engine.run(
            [], "Generate C code for the supplied function", StepBudget(max_steps=8)
        )
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(result["rounds"]), 1)
        self.assertEqual(self.engine.state["verification_spec"]["test_code"], f.CODE)
        self.assertEqual(len(self.model.calls), 3)

    def test_schema_and_independent_acceptance_errors_arrive_together(self):
        plan = self.candidate(
            note="x" * 1601,
            test_code=f.CODE.replace("value(), 4", "value(), value()"),
            test_driver=["../unsafe.py", "{test_file}"],
        )
        self.model = Model(
            [f.reply(plan), f.reply({"kind": "blocked", "note": "Inspect the reported issues"})]
        )
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state")
        result = self.engine.run(
            [], "Fix the requested function", StepBudget(max_steps=4)
        )
        self.assertEqual(result["status"], "task_blocked", result)
        notice = json.loads(self.model.calls[2][-1].content.split("] ", 1)[1].split("\n", 1)[0])
        self.assertEqual(notice["allowed_paths"], ["/note", "/test_code", "/test_driver"])
        for text in ("1600", "test_driver", "独立预期"):
            self.assertIn(text, "\n".join(notice["issues"]))
        self.assertNotIn(plan["steps"][0]["command"], self.model.calls[2][-1].content)
        self.assertEqual(result["rounds"], [])
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_all_reported_fields_can_be_corrected_in_one_local_call(self):
        wrong = f.CODE.replace("value(), 4", "value(), value()")
        plan = self.candidate(
            note="x" * 1601, test_code=wrong, test_driver=["../unsafe.py", "{test_file}"]
        )
        result = self.run_replies(
            [
                f.reply(plan),
                lambda m: correction(
                    m,
                    [
                        edit("/note", json.dumps(plan["note"]), '"Fix the specified value"'),
                        edit("/test_code", json.dumps(wrong), json.dumps(f.CODE)),
                        edit("/test_driver", json.dumps(plan["test_driver"]), "[]"),
                    ],
                ),
            ]
        )
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(self.model.calls), 4)

    def test_syntax_error_skips_dependent_ast_checks_but_keeps_other_errors(self):
        plan = self.candidate(
            note="x" * 1601, test_code="def broken(", test_driver=["../unsafe.py"]
        )
        self.model = Model(
            [f.reply(plan), f.reply({"kind": "blocked", "note": "Inspect syntax failure"})]
        )
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state")
        result = self.engine.run(
            [], "Fix the requested function", StepBudget(max_steps=4)
        )
        self.assertEqual(result["status"], "task_blocked", result)
        notice = self.model.calls[2][-1].content
        self.assertIn("1600", notice)
        self.assertIn("test_driver", notice)
        self.assertIn("语法错误", notice)

    def test_repeated_format_errors_return_diagnostics_without_executing_work(self):
        broken = f.CODE.replace("\n", "\\n")
        plan = self.candidate(test_code=broken)
        original_source = (self.repo / "sample.py").read_bytes()

        def correct(messages, edits):
            self.assertEqual((self.repo / "sample.py").read_bytes(), original_source)
            self.assertEqual(self.engine.state.get("rounds", []), [])
            return correction(messages, edits)

        result = self.run_replies([
            f.reply(plan),
            lambda m: correct(m, [edit("/test_code", broken, f.CODE)]),
            lambda m: correct(m, [edit("/test_code", broken + "x", f.CODE, "edit_text")]),
            lambda m: correct(m, [edit("/test_code", json.dumps(broken), json.dumps(f.CODE))]),
        ])
        self.assertEqual(result["status"], "verified_success", result)
        notices = [json.loads(self.model.calls[i][-1].content.split("] ", 1)[1].split("\n", 1)[0])
                   for i in (2, 3, 4)]
        self.assertIn("category=python_syntax", notices[0]["error"])
        self.assertIn("source line 1, column", notices[0]["error"])
        self.assertIn("/edits/0/before", notices[1]["error"])
        self.assertIn("field JSON", notices[1]["error"])
        self.assertIn("match_count=0", notices[2]["error"])
        self.assertEqual(len({notice["draft_id"] for notice in notices}), 1)
        self.assertTrue(all(notice["allowed_paths"] == ["/test_code"] for notice in notices))
        accepted = json.loads((self.root / "state/plans/p0001/plan.json").read_text())
        self.assertEqual(accepted["plan"]["test_code"], f.CODE)
        self.assertEqual(len(accepted["plan"]["steps"]), len(plan["steps"]))
        for actual, original in zip(accepted["plan"]["steps"], plan["steps"]):
            self.assertEqual({key: actual[key] for key in original}, original)

    def test_review_parameters_and_code_are_checked_together_without_execution(self):
        from agent.acceptance.revisions import Review

        self.engine = TaskRunner(None, self.env)
        self.engine.state = {"request": "Fix the requested function", "acceptance": None}
        with self.assertRaises(repair.ValidationErrors) as caught:
            self.engine.validate_work(
                {
                    "decision": "revise",
                    "trial_id": "unknown",
                    "reason": "x" * 1601,
                    "test_code": f.CODE.replace("value(), 4", "value(), value()"),
                },
                Review,
            )
        for text in ("1600", "独立预期"):
            self.assertIn(text, str(caught.exception))
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_malformed_arguments_can_be_corrected_without_replanning(self):
        arguments = json.dumps(self.candidate()) + "}"
        malformed = AIMessage(
            content="",
            invalid_tool_calls=[
                {"name": "submit_plan", "id": "bad-json", "args": arguments, "error": "Extra data"}
            ],
        )
        result = self.run_replies(
            [
                malformed,
                lambda m: correction(
                    m, [edit("$raw", arguments[-30:], arguments[-30:-1], "edit_text")]
                ),
            ]
        )
        self.assertEqual(result["status"], "verified_success", result)
        self.assertNotIn(f.CODE, self.model.calls[2][-1].content)

    def test_rejected_draft_resume_does_not_repeat_completed_read(self):
        inspect = {
            "kind": "inspect",
            "note": "needed location",
            "steps": [{"id": "r", "op": "read", "path": "sample.py"}],
        }
        plan = self.candidate()
        plan["extra"] = "unnecessary"
        interrupt, error = self.crash_on("plan_rejected")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(inspect), f.reply(plan)])
        result = self.run_replies(
            [lambda m: correction(m, [edit("/extra", '"unnecessary"', "", "remove")])]
        )
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["calls_this_invocation"], 2)
        self.assertEqual(
            len(list((self.root / "state/plans/p0001").glob("p0001-*/result.json"))), 1
        )

    def test_received_correction_resume_does_not_repeat_request(self):
        plan = self.candidate()
        plan["extra"] = 1
        interrupt, error = self.crash_on("plan_rejected")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(plan)])
        interrupt, error = self.crash_on("model_response_received")
        with interrupt, self.assertRaises(error):
            self.run_replies([lambda m: correction(m, [edit("/extra", "1", "", "remove")])])
        result = self.run_replies([])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["calls_this_invocation"], 1)
        self.assertEqual(result["planner_calls"], 4)

    def test_completed_correction_step_resume_does_not_repeat_write(self):
        plan = self.candidate()
        plan["extra"] = 1
        counter = self.shell(
            "from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1 if p.exists() else 1))",
            "counter",
        )
        plan["steps"].insert(0, counter)
        result = self.run_replies(
            [f.reply(plan), lambda m: correction(m, [edit("/extra", "1", "", "remove")])],
            pause_after_steps=1,
        )
        self.assertEqual(result["status"], "paused", result)
        result = self.run_replies([])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual((self.repo / "counter").read_text(), "1")
        self.assertEqual(result["calls_this_invocation"], 0)

    def test_tampered_draft_stops_instead_of_correcting_or_writing(self):
        plan = self.candidate()
        plan["extra"] = 1
        interrupt, error = self.crash_on("plan_rejected")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(plan)])
        file = next((self.root / "state/drafts").glob("*.json"))
        file.write_text("{}")
        result = self.run_replies([lambda m: correction(m, [edit("/extra", "1", "", "remove")])])
        self.assertEqual(result["status"], "state_conflict", result)
        self.assertEqual(self.model.calls, [])
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_real_sdk_requests_keep_both_strict_tools_fixed_and_raw_repair_arguments(self):
        requests = []
        original = self.candidate()
        original["extra"] = 1

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                name, arguments = "get_boundary_hints", json.dumps({"keys": ["general"]})
            elif len(requests) == 2:
                name, arguments = "submit_plan", json.dumps(original, ensure_ascii=False, indent=2)
            elif len(requests) == 3:
                notice = json.loads(
                    body["messages"][-1]["content"].split("] ", 1)[1].split("\n", 1)[0]
                )
                name, arguments = "repair_work", json.dumps(
                    {"draft_id": notice["draft_id"], "edits": [edit("/extra", "1", "", "remove")]},
                    ensure_ascii=False,
                    indent=2,
                )
            else:
                notice = json.loads(body["messages"][-1]["content"])
                name, arguments = "freeze_acceptance", json.dumps(
                    {
                        "trial_id": notice["trial_id"],
                        "reason": "Inspect and retain the baseline behavior",
                    }
                )
            return httpx.Response(
                200,
                json={
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "offline reasoning",
                                "tool_calls": [
                                    {
                                        "id": "call-" + name,
                                        "type": "function",
                                        "function": {"name": name, "arguments": arguments},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with patch(
                "agent.model.network.model_http_clients", return_value={"http_client": client}
            ):
                model = create_model(
                    {"model_name": "deepseek-flash", "api_key": "offline-placeholder"}
                )
            engine = TaskRunner(model, self.env, state_dir=self.root / "state")
            result = engine.run([], "return 4", StepBudget(max_steps=8))
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(requests), 4)
        from agent.model.tools import CODING_TOOLS

        expected = wire_input([], CODING_TOOLS)["tools"]
        for body in requests:
            self.assertEqual(body["tools"], expected)
            self.assertTrue(all(t["function"]["strict"] for t in body["tools"]))
            self.assertNotIn("response_format", body)
        self.assertEqual(
            requests[1]["messages"][: len(requests[0]["messages"])], requests[0]["messages"]
        )
        self.assertNotIn(original["steps"][0]["command"], requests[2]["messages"][-1]["content"])
        accepted = engine.state["history"][-2:]
        wire = wire_input(messages_from_dict(accepted), WORK_TOOLS)
        self.assertEqual(
            wire["messages"][0]["tool_calls"][0]["function"]["arguments"],
            accepted[0]["data"]["additional_kwargs"][RAW_TOOL_CALLS][0]["function"]["arguments"],
        )
        self.assertEqual(wire["messages"][1]["tool_call_id"], "call-freeze_acceptance")


class CorrectionSafetyTests(unittest.TestCase):
    def test_json_diagnostic_identifies_inner_field_and_multiline_position(self):
        for field in ("before", "after"):
            item = edit("/note", json.dumps("old"), json.dumps("new"))
            item[field] = '[\n  invalid\n]'
            original = {"note": "old"}
            with self.subTest(field=field), self.assertRaises(ValueError) as caught:
                repair.apply(original, [repair.Edit(**item)], ["/note"])
            message = str(caught.exception)
            self.assertIn("category=correction_json", message)
            self.assertIn("/edits/0/" + field, message)
            self.assertIn("target=/note", message)
            self.assertIn("line 2, column 3", message)
            self.assertIn("field JSON", message)
            self.assertIn("JSON literals", message)
            self.assertEqual(original, {"note": "old"})

    def test_strict_json_rejections_keep_the_parameter_context(self):
        for invalid in ('{"same": 1, "same": 2}', 'NaN'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError) as caught:
                repair.apply({"note": "old"}, [repair.Edit(**edit(
                    "/note", json.dumps("old"), invalid))], ["/note"])
            self.assertIn("/edits/0/after", str(caught.exception))
            self.assertIn("category=correction_json", str(caught.exception))
            self.assertNotIn("line ", str(caught.exception))

    def test_text_diagnostic_counts_matches_without_guessing_a_missing_location(self):
        original = {"test_code": "same\nsame"}
        for before, count in (("same", 2), ("missing", 0)):
            with self.subTest(before=before), self.assertRaises(ValueError) as caught:
                repair.apply(original, [repair.Edit(**edit(
                    "/test_code", before, "different", "edit_text"))], ["/test_code"])
            message = str(caught.exception)
            self.assertIn("category=correction_text_match", message)
            self.assertIn("/edits/0/before", message)
            self.assertIn("match_count=" + str(count), message)
            self.assertIn("target_length=9", message)
            self.assertEqual(original, {"test_code": "same\nsame"})

    def test_exact_text_preserves_unicode_quotes_slashes_and_newlines(self):
        value = {"test_code": 'print("中文\\n")\r\npath = "C:\\\\files"\n# imports 错误\n'}
        changed = repair.apply(
            value,
            [repair.Edit(**edit("/test_code", "imports 错误", "imports 正确", "edit_text"))],
            ["/test_code"],
        )
        self.assertEqual(
            changed["test_code"], value["test_code"].replace("imports 错误", "imports 正确")
        )
        self.assertEqual(value["test_code"].count("错误"), 1)

    def test_wrong_id_outside_scope_and_ambiguous_text_do_not_guess(self):
        value = {"test_code": "same same", "steps": [{"command": "protected"}]}
        for item in [
            edit("/steps/0/command", '"protected"', '"changed"'),
            edit("/test_code", "same", "different", "edit_text"),
            edit("/test_code", '"incorrect"', '"different"'),
        ]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                repair.apply(value, [repair.Edit(**item)], ["/test_code"])
        owner = SimpleNamespace(state={"repair_draft": {"id": "current"}})
        with self.assertRaisesRegex(ValueError, "current draft_id"):
            repair.merge(owner, {"draft_id": "stale", "edits": [edit("/note", '"x"', '"y"')]})
        self.assertEqual(value["steps"][0]["command"], "protected")

    def test_failed_multi_edit_is_atomic(self):
        value = {"steps": [{"query_placeholder": "", "limit": 25}]}
        original = deepcopy(value)
        items = [
            edit("/steps/0/query_placeholder", '""', "", "remove"),
            edit("/steps/0/limit", "1", "10"),
        ]
        with self.assertRaisesRegex(ValueError, "Original value"):
            repair.apply(value, [repair.Edit(**item) for item in items], ["/steps/0"])
        self.assertEqual(value, original)

    def test_bad_pointers_return_mechanical_errors(self):
        for path in (
            "note",
            "/steps/8/op",
            "/steps/foo/op",
            "/steps/0/command/nested",
            "/steps/~3",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                repair.apply(
                    {"steps": [{"command": "text"}]},
                    [repair.Edit(**edit(path, '"text"', '"changed"'))],
                    [path],
                )

    def test_raw_json_changes_are_exact_and_still_must_parse(self):
        for after in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x":1}}'):
            with self.subTest(after=after), self.assertRaises(ValueError):
                repair.apply(
                    '{"x":1}}',
                    [repair.Edit(**edit("$raw", '{"x":1}}', after, "edit_text"))],
                    ["$raw"],
                )
        with self.assertRaisesRegex(ValueError, "requires edit_text"):
            repair.apply('{"x":1}}', [repair.Edit(**edit("$raw", '"anything"', '{}'))], ["$raw"])

    def test_missing_or_changed_draft_is_not_rebased(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            owner = SimpleNamespace(
                root=Path(folder), state={"source": "version"}, rejected_input={"note": "invalid"}
            )
            repair.reject(owner, ValueError("Step not valid"))
            ref = owner.state["repair_draft"]
            record = owner.root / ref["record"]
            record.write_text("{}")
            with self.assertRaises(ModelRequestNotSent):
                repair.reject(owner, ValueError("Step not valid"))
            record.unlink()
            with self.assertRaises(ModelRequestNotSent):
                repair.read(owner, ref)

    def test_repair_resource_is_archived_and_bound_to_recovery_fingerprint(self):
        from eval.swebench_runner import archive_source
        from agent.settings import PROJECT_ROOT
        import tempfile

        key = code_key()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "resources").mkdir()
            schema = PROJECT_ROOT / "resources/repair.schema.json"
            (root / "resources/repair.schema.json").write_bytes(schema.read_bytes() + b"\n")

            class Overlay:
                def __truediv__(self, name):
                    return (
                        root / name
                        if name == "resources/repair.schema.json"
                        else PROJECT_ROOT / name
                    )

            with patch("agent.storage.recovery.PROJECT_ROOT", Overlay()):
                self.assertNotEqual(code_key(), key)
            archive_source(root)
            hashes = json.loads((root / "source-sha256.json").read_text())
            self.assertEqual(
                hashes["resources/repair.schema.json"],
                hashlib.sha256(schema.read_bytes()).hexdigest(),
            )
            self.assertIn("agent/task/repair.py", hashes)


if __name__ == "__main__":
    unittest.main()
