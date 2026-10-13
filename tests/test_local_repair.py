"""Local corrections retain real acceptance and persistent execution boundaries."""
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
from support import interrupt_after_persist
import test_task_runner as f


def correction(messages, edits, identifier=None):
    notice = json.loads(messages[-1].content.split("] ", 1)[1].split("\n", 1)[0])
    return AIMessage(content="", tool_calls=[{"name": "correct_plan_arguments", "id": "repair-call",
        "args": {"draft_id": identifier or notice["draft_id"], "edits": edits}}])


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
        review = f.confirmation(messages)
        if review is not None:
            return review
        reply = next(self.replies)
        return reply(messages) if callable(reply) else reply


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
        result = self.run_replies([f.reply(plan), lambda m: correction(m, [
            edit("/steps/0/query_placeholder", '""', "", "remove")])])
        self.assertEqual(result["status"], "verified_success", result)
        notice = self.model.calls[1][-1].content
        self.assertNotIn("UNCHANGED_WORK_NOT_IN_CORRECTION", notice)
        self.assertNotIn(f.CODE, notice)
        draft = json.loads(next((self.root / "state/drafts").glob("*.json")).read_text())
        self.assertEqual(draft["value"], plan)
        accepted = json.loads((self.root / "state/plans/p0001/plan.json").read_text())
        self.assertEqual(accepted["plan"]["steps"][0]["command"], plan["steps"][0]["command"])
        self.assertEqual(accepted["plan"]["test_code"], f.CODE)
        self.assertEqual(accepted["origin_response"]["data"]["tool_calls"][0]["name"], "correct_plan_arguments")
        self.assertEqual(accepted["response"]["data"]["tool_calls"][0]["name"], "freeze_acceptance")
        state = self.load_state()
        self.assertEqual(state["history"][-1]["data"]["tool_call_id"], "confirm-call")
        self.assertEqual(state["history"][-1]["data"]["name"], "freeze_acceptance")
        self.assertEqual(self.model.tools[0], self.model.tools[1])
        self.assertNotIn("repair_draft", state)

    def test_missing_field_is_explicitly_added(self):
        plan = self.candidate()
        del plan["note"]
        result = self.run_replies([f.reply(plan), lambda m: correction(m, [edit(
            "/note", "", json.dumps("落实已给出的修改，中文"))])])
        self.assertEqual(result["status"], "verified_success", result)



    def draft_read_then_correction(self):
        plan = self.candidate()
        plan["extra"] = 1
        draft = {}

        def read_draft(messages):
            notice = json.loads(messages[-1].content.split("] ", 1)[1].split("\n", 1)[0])
            draft["id"] = notice["draft_id"]
            return f.reply({"kind": "inspect", "note": "Read the rejected original before correcting it",
                "steps": [{"id": "draft", "op": "read", "path": notice["original_record"],
                           "offset": 0, "limit": 200}]})

        def correct_draft(messages):
            return AIMessage(content="", tool_calls=[{"name": "correct_plan_arguments", "id": "after-read",
                "args": {"draft_id": draft["id"], "edits": [edit("/extra", "1", "", "remove")]}}])

        return plan, draft, read_draft, correct_draft

    def test_reading_rejected_draft_retains_repair_target(self):
        plan, draft, read_draft, correct_draft = self.draft_read_then_correction()
        result = self.run_replies([f.reply(plan), read_draft, correct_draft])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["planner_calls"], 4)
        read_result = json.loads(self.model.calls[2][-1].content)["steps"][0]
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
        self.assertEqual(result["planner_calls"], 4)
        self.assertEqual(len(list((self.root / "state/plans/p0001").glob("p0001-*/result.json"))), 1)

    def test_invalid_repair_keeps_original_error_and_then_can_correct(self):
        plan = self.candidate()
        plan["steps"][0]["query_placeholder"] = ""
        result = self.run_replies([f.reply(plan), lambda m: correction(m, [edit(
            "/steps/0/query_placeholder", '"wrong"', "", "remove")]),
            lambda m: correction(m, [edit("/steps/0/query_placeholder", '""', "", "remove")])])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertIn("Original value does not match", self.model.calls[2][-1].content)
        self.assertIn("query_placeholder", self.model.calls[2][-1].content)
        self.assertIn("original_error", self.model.calls[2][-1].content)
        self.assertEqual(len(list((self.root / "state/drafts").glob("*.json"))), 1)

    def test_modified_value_is_fully_revalidated_before_execution(self):
        plan = self.candidate()
        plan["steps"][0]["op"] = "unknown"
        result = self.run_replies([f.reply(plan), lambda m: correction(m, [edit(
            "/steps/0/op", '"unknown"', '"still_unknown"')]),
            lambda m: correction(m, [edit("/steps/0/op", '"still_unknown"', '"shell"')])])
        self.assertEqual(result["status"], "verified_success", result)
        drafts = list((self.root / "state/drafts").glob("*.json"))
        self.assertEqual(len(drafts), 2)
        ids = [json.loads(m[-1].content.split("] ", 1)[1].split("\n", 1)[0])["draft_id"]
               for m in self.model.calls[1:] if m[-1].content.startswith("[Plan rejected" )]
        self.assertNotEqual(*ids)

    def test_real_assertion_failure_is_not_repaired_away(self):
        plan = self.candidate(value=2)
        plan["steps"][0]["extra"] = ""
        result = self.run_replies([f.reply(plan), lambda m: correction(m, [edit(
            "/steps/0/extra", '""', "", "remove")]), f.reply({"kind": "blocked", "note": "actual failure"})])
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertEqual(result["rounds"][0]["verification"]["status"], "failed")
        self.assertIn("AssertionError", str(self.model.calls[3]))
        self.assertEqual(json.loads(self.model.calls[3][-1].content)["outcome"], "verification_failed")

    def test_frozen_tests_cannot_be_replaced_by_field_correction(self):
        weak = self.candidate(test_code=f.CODE.replace("value(), 4", "value(), 2"))
        replacements = lambda m: correction(m, [edit("/test_code", json.dumps(weak["test_code"]),
            json.dumps(weak["test_code"] + "# still weaker\n"))])
        result = self.run_replies([f.reply(self.candidate(2)), f.reply(weak)] + [replacements] * 6)
        self.assertEqual(result["status"], "budget_exhausted", result)
        self.assertEqual(json.loads((self.root / "state/frozen-acceptance.json").read_text())["spec"]["test_code"], f.CODE)
        self.assertEqual(len(result["rounds"]), 1)

    def test_schema_and_independent_acceptance_errors_arrive_together(self):
        plan = self.candidate(note="x" * 1601, test_code=f.CODE.replace("value(), 4", "value(), value()"),
                              test_driver=["../unsafe.py", "{test_file}"])
        self.model = Model([f.reply(plan), f.reply({"kind": "blocked", "note": "Inspect the reported issues"})])
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state")
        result = self.engine.run([], "ccode should generate working C code", StepBudget(max_steps=4))
        self.assertEqual(result["status"], "task_blocked", result)
        notice = json.loads(self.model.calls[1][-1].content.split("] ", 1)[1].split("\n", 1)[0])
        self.assertEqual(notice["allowed_paths"], ["/note", "/test_code", "/test_driver"])
        for text in ("1600", "test_driver", "独立预期", "ARTIFACT_CHECKS"):
            self.assertIn(text, "\n".join(notice["issues"]))
        self.assertNotIn(plan["steps"][0]["command"], self.model.calls[1][-1].content)
        self.assertEqual(result["rounds"], [])
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_all_reported_fields_can_be_corrected_in_one_local_call(self):
        wrong = f.CODE.replace("value(), 4", "value(), value()")
        plan = self.candidate(note="x" * 1601, test_code=wrong, test_driver=["../unsafe.py", "{test_file}"])
        result = self.run_replies([f.reply(plan), lambda m: correction(m, [
            edit("/note", json.dumps(plan["note"]), '"Fix the specified value"'),
            edit("/test_code", json.dumps(wrong), json.dumps(f.CODE)),
            edit("/test_driver", json.dumps(plan["test_driver"]), "[]")])])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(self.model.calls), 3)

    def test_syntax_error_skips_dependent_ast_checks_but_keeps_other_errors(self):
        plan = self.candidate(note="x" * 1601, test_code="def broken(", test_driver=["../unsafe.py"])
        self.model = Model([f.reply(plan), f.reply({"kind": "blocked", "note": "Inspect syntax failure"})])
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / "state")
        result = self.engine.run([], "ccode should generate working C code", StepBudget(max_steps=4))
        self.assertEqual(result["status"], "task_blocked", result)
        notice = self.model.calls[1][-1].content
        self.assertIn("1600", notice)
        self.assertIn("test_driver", notice)
        self.assertIn("语法错误", notice)
        self.assertNotIn("缺少顶格 ARTIFACT_CHECKS", notice)

    def test_review_parameters_and_code_are_checked_together_without_execution(self):
        from agent.acceptance.revisions import Review
        self.engine = TaskRunner(None, self.env)
        self.engine.state = {"request": "ccode should generate working C code", "acceptance": None}
        with self.assertRaises(repair.ValidationErrors) as caught:
            self.engine.validate_work({"decision": "revise", "trial_id": "unknown", "reason": "x" * 1601,
                "test_code": f.CODE.replace("value(), 4", "value(), value()")}, Review)
        for text in ("1600", "独立预期", "ARTIFACT_CHECKS"):
            self.assertIn(text, str(caught.exception))
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_malformed_arguments_can_be_corrected_without_replanning(self):
        arguments = json.dumps(self.candidate()) + "}"
        malformed = AIMessage(content="", invalid_tool_calls=[{
            "name": "submit_plan", "id": "bad-json", "args": arguments, "error": "Extra data"}])
        result = self.run_replies([malformed, lambda m: correction(m, [edit(
            "$raw", arguments[-30:], arguments[-30:-1], "edit_text")])])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertNotIn(f.CODE, self.model.calls[1][-1].content)

    def test_rejected_draft_resume_does_not_repeat_completed_read(self):
        inspect = {"kind": "inspect", "note": "needed location", "steps": [
            {"id": "r", "op": "read", "path": "sample.py"}]}
        plan = self.candidate()
        plan["extra"] = "unnecessary"
        interrupt, error = self.crash_on("plan_rejected")
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(inspect), f.reply(plan)])
        result = self.run_replies([lambda m: correction(m, [edit("/extra", '"unnecessary"', "", "remove")])])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["calls_this_invocation"], 2)
        self.assertEqual(len(list((self.root / "state/plans/p0001").glob("p0001-*/result.json"))), 1)

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
        self.assertEqual(result["planner_calls"], 3)

    def test_completed_correction_step_resume_does_not_repeat_write(self):
        plan = self.candidate()
        plan["extra"] = 1
        counter = self.shell("from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1 if p.exists() else 1))", "counter")
        plan["steps"].insert(0, counter)
        result = self.run_replies([f.reply(plan), lambda m: correction(m, [edit("/extra", "1", "", "remove")])], pause_after_steps=1)
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
                name, arguments = "submit_plan", json.dumps(original, ensure_ascii=False, indent=2)
            elif len(requests) == 2:
                notice = json.loads(body["messages"][-1]["content"].split("] ", 1)[1].split("\n", 1)[0])
                name, arguments = "correct_plan_arguments", json.dumps({"draft_id": notice["draft_id"],
                    "edits": [edit("/extra", "1", "", "remove")]}, ensure_ascii=False, indent=2)
            else:
                notice = json.loads(body["messages"][-1]["content"])
                name, arguments = "freeze_acceptance", json.dumps({"trial_id": notice["trial_id"],
                    "reason": "Inspect and retain the baseline behavior"})
            return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "created": 1,
                "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": "", "reasoning_content": "offline reasoning",
                    "tool_calls": [{"id": "call-" + name, "type": "function", "function": {
                        "name": name, "arguments": arguments}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with patch("agent.model.network.model_http_clients", return_value={"http_client": client}):
                model = create_model({"model_name": "deepseek-flash", "api_key": "offline-placeholder"})
            engine = TaskRunner(model, self.env, state_dir=self.root / "state")
            result = engine.run([], "return 4", StepBudget(max_steps=8))
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(requests), 3)
        from agent.model.tools import CODING_TOOLS
        expected = wire_input([], CODING_TOOLS)["tools"]
        for body in requests:
            self.assertEqual(body["tools"], expected)
            self.assertTrue(all(t["function"]["strict"] for t in body["tools"]))
            self.assertNotIn("response_format", body)
        self.assertEqual(requests[1]["messages"][:len(requests[0]["messages"])], requests[0]["messages"])
        self.assertNotIn(original["steps"][0]["command"], requests[1]["messages"][-1]["content"])
        accepted = engine.state["history"][-2:]
        wire = wire_input(messages_from_dict(accepted), WORK_TOOLS)
        self.assertEqual(wire["messages"][0]["tool_calls"][0]["function"]["arguments"],
            accepted[0]["data"]["additional_kwargs"][RAW_TOOL_CALLS][0]["function"]["arguments"])
        self.assertEqual(wire["messages"][1]["tool_call_id"], "call-freeze_acceptance")


class CorrectionSafetyTests(unittest.TestCase):
    def test_exact_text_preserves_unicode_quotes_slashes_and_newlines(self):
        value = {"test_code": 'print("中文\\n")\r\npath = "C:\\\\files"\n# imports 错误\n'}
        changed = repair.apply(value, [repair.Edit(**edit("/test_code", "imports 错误", "imports 正确", "edit_text"))], ["/test_code"])
        self.assertEqual(changed["test_code"], value["test_code"].replace("imports 错误", "imports 正确"))
        self.assertEqual(value["test_code"].count("错误"), 1)

    def test_wrong_id_outside_scope_and_ambiguous_text_do_not_guess(self):
        value = {"test_code": "same same", "steps": [{"command": "protected"}]}
        for item in [edit("/steps/0/command", '"protected"', '"changed"'),
                     edit("/test_code", "same", "different", "edit_text"),
                     edit("/test_code", '"incorrect"', '"different"')]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                repair.apply(value, [repair.Edit(**item)], ["/test_code"])
        owner = SimpleNamespace(state={"repair_draft": {"id": "current"}})
        with self.assertRaisesRegex(ValueError, "current draft_id"):
            repair.merge(owner, {"draft_id": "stale", "edits": [edit("/note", '"x"', '"y"')]})
        self.assertEqual(value["steps"][0]["command"], "protected")

    def test_failed_multi_edit_is_atomic(self):
        value = {"steps": [{"query_placeholder": "", "limit": 25}]}
        original = deepcopy(value)
        items = [edit("/steps/0/query_placeholder", '""', "", "remove"),
                 edit("/steps/0/limit", "1", "10")]
        with self.assertRaisesRegex(ValueError, "Original value"):
            repair.apply(value, [repair.Edit(**item) for item in items], ["/steps/0"])
        self.assertEqual(value, original)

    def test_bad_pointers_return_mechanical_errors(self):
        for path in ("note", "/steps/8/op", "/steps/foo/op", "/steps/0/command/nested", "/steps/~3"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                repair.apply({"steps": [{"command": "text"}]}, [repair.Edit(**edit(path, '"text"', '"changed"'))], [path])

    def test_raw_json_changes_are_exact_and_still_must_parse(self):
        for after in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x":1}}'):
            with self.subTest(after=after), self.assertRaises(ValueError):
                repair.apply('{"x":1}}', [repair.Edit(**edit("$raw", '{"x":1}}', after, "edit_text"))], ["$raw"])
        with self.assertRaisesRegex(ValueError, "requires edit_text"):
            repair.apply('{"x":1}}', [repair.Edit(**edit("$raw", '"anything"', '{}'))], ["$raw"])

    def test_missing_or_changed_draft_is_not_rebased(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            owner = SimpleNamespace(root=Path(folder), state={"source": "version"}, rejected_input={"note": "invalid"})
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
                    return root / name if name == "resources/repair.schema.json" else PROJECT_ROOT / name
            with patch("agent.storage.recovery.PROJECT_ROOT", Overlay()):
                self.assertNotEqual(code_key(), key)
            archive_source(root)
            hashes = json.loads((root / "source-sha256.json").read_text())
            self.assertEqual(hashes["resources/repair.schema.json"], hashlib.sha256(schema.read_bytes()).hexdigest())
            self.assertIn("agent/task/repair.py", hashes)


if __name__ == "__main__":
    unittest.main()
