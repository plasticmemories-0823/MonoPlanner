"""计划去重忽略说明和局部编号，同时保留真实操作、依赖与版本边界。"""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import pytest

from agent.task.budget import StepBudget
from agent.task.contracts import WorkRequest
from agent.task.runner import TaskRunner
import test_task_runner as fixtures


@pytest.fixture
def receiver(tmp_path):
    """直接检查接收边界；真实执行与恢复由下面的流程测试覆盖。"""
    runner = TaskRunner(None, None)
    runner.root = tmp_path
    runner.evidence = SimpleNamespace(current="source-a")
    runner.state = {
        "seen_plans": [],
        "plan_count": 0,
        "requirements": {"digest": "requirements-a"},
    }
    runner.persist = Mock()
    return runner


def inspection(**step_changes):
    return {
        "kind": "inspect",
        "note": "Read the relevant source",
        "steps": [{"id": "read", "op": "read", "path": "sample.py", **step_changes}],
    }


def accept(runner, value):
    plan = WorkRequest.model_validate(value)
    runner.accept_plan(plan, {"submitted": value})
    return plan


@pytest.mark.parametrize("change", [
    {"note": "Check the same source again"},
    {"steps": [{"id": "renamed", "op": "read", "path": "sample.py"}]},
    {
        "note": "The step has a new label",
        "steps": [{"id": "renamed", "op": "read", "path": "sample.py"}],
    },
])
def test_description_or_step_rename_cannot_replay_same_plan(receiver, change):
    first = inspection()
    accept(receiver, first)
    before = deepcopy(receiver.state)
    with pytest.raises(ValueError, match="Identical plan"):
        accept(receiver, {**first, **change})
    assert receiver.state == before
    assert not (receiver.root / "plans/p0002").exists()


def test_same_echo_command_cannot_replay_with_a_new_id_and_note(receiver):
    first = inspection(id="keep1", op="shell", command="echo ok")
    accept(receiver, first)
    changed = deepcopy(first)
    changed["note"] = "Retain the same result"
    changed["steps"][0]["id"] = "keep2"
    with pytest.raises(ValueError, match="Identical plan"):
        accept(receiver, changed)
    assert receiver.state["plan_count"] == 1


def related_steps():
    return {
        "kind": "inspect",
        "note": "Locate two symbols, then read the first result",
        "steps": [
            {"id": "first", "op": "search", "query": "alpha"},
            {"id": "second", "op": "search", "query": "beta"},
            {"id": "read", "op": "read", "from_step": "first"},
        ],
    }


def renamed_related_steps():
    plan = related_steps()
    plan["note"] = "Read the same dependencies with new labels"
    plan["steps"][0]["id"] = "second"
    plan["steps"][1]["id"] = "first"
    plan["steps"][2].update(id="renamed_read", from_step="second")
    return plan


def test_consistently_renamed_dependency_graph_is_duplicate(receiver):
    accept(receiver, related_steps())
    with pytest.raises(ValueError, match="Identical plan"):
        accept(receiver, renamed_related_steps())
    assert receiver.state["plan_count"] == 1


def test_changing_dependency_target_is_not_a_duplicate(receiver):
    accept(receiver, related_steps())
    changed = renamed_related_steps()
    changed["steps"][2]["from_step"] = "first"
    accept(receiver, changed)
    assert receiver.state["plan_count"] == 2


def test_changing_execution_order_is_not_a_duplicate(receiver):
    first = {
        "kind": "inspect",
        "note": "Run two diagnostics",
        "steps": [
            {"id": "a", "op": "shell", "command": "echo alpha"},
            {"id": "b", "op": "shell", "command": "echo beta"},
        ],
    }
    accept(receiver, first)
    changed = deepcopy(first)
    changed["steps"].reverse()
    accept(receiver, changed)
    assert receiver.state["plan_count"] == 2


@pytest.mark.parametrize("before,after", [
    ({"path": "sample.py"}, {"path": "other.py"}),
    ({"offset": 0}, {"offset": 10}),
    ({"limit": 20}, {"limit": 40}),
    ({"cursor": "page-one"}, {"cursor": "page-two"}),
    ({"query": ""}, {"query": "reread: verify the original lines"}),
    (
        {"op": "search", "query": "alpha beta"},
        {"op": "search", "query": "alpha  beta"},
    ),
    (
        {"op": "shell", "command": "echo first"},
        {"op": "shell", "command": "echo second"},
    ),
    (
        {"op": "shell", "command": "echo ok", "expected_exit": [0]},
        {"op": "shell", "command": "echo ok", "expected_exit": [0, 1]},
    ),
])
def test_changed_operation_parameters_are_not_duplicates(receiver, before, after):
    accept(receiver, inspection(**before))
    accept(receiver, inspection(**after))
    assert receiver.state["plan_count"] == 2


@pytest.mark.parametrize("boundary", ["source", "requirements"])
def test_same_operation_can_be_replanned_after_version_change(receiver, boundary):
    plan = inspection()
    accept(receiver, plan)
    if boundary == "source":
        receiver.evidence.current = "source-b"
    else:
        receiver.state["requirements"]["digest"] = "requirements-b"
    accept(receiver, plan)
    assert receiver.state["plan_count"] == 2


def test_identity_normalization_preserves_submitted_plan_and_saved_record(receiver):
    original = renamed_related_steps()
    expected = WorkRequest.model_validate(original).model_dump()
    plan = accept(receiver, original)
    saved = json.loads((receiver.root / "plans/p0001/plan.json").read_text())
    assert plan.model_dump() == expected
    assert saved["plan"] == expected
    assert saved["response"] == {"submitted": original}


class PlanDeduplicationFlowTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    shell = fixtures.TaskRunnerTests.shell
    candidate = fixtures.TaskRunnerTests.candidate
    loop = fixtures.TaskRunnerTests.loop

    def repeated_inspection(self):
        first = {
            "kind": "inspect",
            "note": "Read the source once",
            "steps": [{"id": "keep1", "op": "read", "path": "sample.py"}],
        }
        renamed = deepcopy(first)
        renamed["note"] = "Same read with a different label"
        renamed["steps"][0]["id"] = "keep2"
        return first, renamed

    def assert_only_first_inspection_executed(self, first):
        requests = [
            json.loads(path.read_text())
            for path in self.agent.root.glob("plans/*/*/request.json")
        ]
        matching = [
            row for row in requests
            if row["op"] == "read" and row["path"] == first["steps"][0]["path"]
        ]
        self.assertEqual([row["id"] for row in matching], ["keep1"])
        self.assertEqual(self.agent.state["plan_count"], 2)
        events = [
            json.loads(line)
            for line in (self.agent.root / "events.jsonl").read_text().splitlines()
        ]
        self.assertEqual(sum(row["kind"] == "plan_rejected" for row in events), 1)

    def test_renamed_inspection_is_rejected_before_execution_and_rework_continues(self):
        first, renamed = self.repeated_inspection()
        result = self.loop([first, renamed, self.candidate()]).run(
            [], "return 4", StepBudget(max_steps=8)
        )
        self.assertEqual(result["status"], "verified_success", result)
        self.assert_only_first_inspection_executed(first)

    def test_renamed_inspection_is_still_rejected_after_persisted_restart(self):
        first, renamed = self.repeated_inspection()
        state_dir = self.root / "persistent-state"
        paused = self.loop([first], state_dir=state_dir, pause_after_steps=1).run(
            [], "return 4", StepBudget(max_steps=8)
        )
        self.assertEqual(paused["status"], "paused", paused)
        original = state_dir / "plans/p0001/p0001-keep1/request.json"
        original_bytes = original.read_bytes()
        result = self.loop(
            [renamed, self.candidate()], state_dir=state_dir, recovery_id="resume-dedup-test"
        ).run([], "return 4", StepBudget(max_steps=8))
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assert_only_first_inspection_executed(first)
