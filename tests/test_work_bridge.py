"""独立工作适配不经过聊天执行器，暂停不等待正在进行的文件操作。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

from agent.application import work_bridge as bridge
from agent.execution.local import LocalEnvironment
from agent.task import control
from agent.storage.requests import read_request
from support import fixture_delivery


class RecordedModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        if self.calls:
            from agent.model.errors import ModelRequestNotSent
            raise ModelRequestNotSent('task_cancelled', 'Offline work fixture explicitly stopped')
        self.calls.append(messages)
        return AIMessage(content="工作原始回复，不直接发布到主聊天", tool_calls=[{
            "id": "finished", "name": "submit_plan", "args": {
                "kind": "blocked", "note": "离线测试停止", "steps": [],
                "test_code": "", "regression_command": "", "test_driver": [],
            },
        }], additional_kwargs={"reasoning_content": "原始工作思考"})


class OfflineEnvironment(LocalEnvironment):
    def __init__(self, repo, image, task_root, **kwargs):
        super().__init__(repo, artifact_root=Path(task_root).parent.parent / "execution")
        self.runtime_facts = {"test": "offline"}
        self.image = image

    def start(self):
        return self

    def close(self):
        pass


def test_work_handoff_preserves_user_originals_and_never_emits_worker_reply(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("原项目材料", encoding="utf-8")
    root = tmp_path / "state"
    model = RecordedModel()
    events = []
    job = {
        "task_id": "a" * 32, "conversation_id": "conversation", "repo": str(repo),
        "budget": {"max_steps": 1}, "request": "保持原始用户要求", "public_conversation": messages_to_dict([
            HumanMessage(content="此前明确约束", id="original")]),
        "execution_binding": {"image": "sha256:" + "b" * 64, "runtime": {}},
    }
    with patch.object(bridge, "require_binding", side_effect=lambda repo, value: value):
        result = bridge.run_work(job, lambda *event: events.append(event), lambda: [],
                                 model=model, environment_factory=OfflineEnvironment, root=root)
    assert result["status"] == "budget_exhausted"
    assert len(model.calls) == 1
    initial = json.loads(model.calls[0][1].content)
    assert initial["request"] == job["request"]
    assert initial["public_conversation"] == job["public_conversation"]
    assert "Chatbot" not in model.calls[0][0].content
    assert all("工作原始回复" not in str(event) for event in events)
    original = read_request(root / "tasks" / job["task_id"] / "requests/call-0001/request.json")
    assert original["messages"] == messages_to_dict(model.calls[0])


def test_pause_records_uncertain_write_and_kills_only_owned_task(tmp_path):
    task = "c" * 32
    owner = {"runner": SimpleNamespace(state={"running_step": "s1"}), "pause": None}
    key = (str(tmp_path.resolve()), task)
    bridge._owners[key] = owner
    try:
        with control.active(tmp_path, task) as signal, patch(
            "agent.storage.resources.cancel_owned_containers",
            return_value=[{"container": "owned", "stopped": True}],
        ) as terminate:
            report = bridge.pause_work(task, "pause-id", root=tmp_path)
            assert signal.kind == "pause"
            assert owner["interrupted_operation"] is True
            assert report["interrupted_operation"] is True
            terminate.assert_called_once_with("task:" + task, root=tmp_path.resolve())
            owner["runner"].state["running_step"] = None
            again = bridge.pause_work(task, "second-pause", root=tmp_path)
            assert again["interrupted_operation"] is True
            assert owner["pause"] == "pause-id"
    finally:
        bridge._owners.pop(key)


def test_unknown_interruption_blocks_before_container_or_model(tmp_path):
    task = "d" * 32
    directory = tmp_path / "tasks" / task
    directory.mkdir(parents=True)
    (directory / "hard-interruption.json").write_text("{}")
    with patch.object(bridge, "TaskDocker") as environment:
        result = bridge.run_work({"task_id": task}, lambda *args: None, lambda: [], root=tmp_path)
    assert result["status"] == "execution_unknown"
    environment.assert_not_called()
    assert (str(tmp_path.resolve()), task) not in bridge._owners


def test_progress_facts_include_committed_actions_but_exclude_private_content():
    state = {"pending": {"plan": {
        "kind": "inspect", "note": "确认解析入口", "test_code": "PRIVATE_TEST",
        "steps": [{"id": "s1", "op": "shell", "path": "parser.py", "code": "PRIVATE_CODE"}],
    }}, "history": [{"reasoning_content": "PRIVATE_THOUGHT"}]}
    accepted = bridge.public_fact(state, "plan_accepted", {"id": "p0001"})
    assert accepted["note"] == "确认解析入口"
    row = {"id": "s1", "op": "shell", "status": "completed", "record": "plans/p0001/s1.json",
           "data": {"exit_code": 0, "stdout": "UNBOUNDED_OUTPUT"}}
    finished = bridge.public_fact(state, "step_finished", row)
    assert finished["steps"][0]["path"] == "parser.py"
    assert finished["steps"][0]["exit_code"] == 0
    assert "PRIVATE" not in str(accepted) + str(finished)
    assert "UNBOUNDED" not in str(finished)


def test_preparation_interruption_does_not_become_an_unknown_model_request(tmp_path):
    from agent.execution.preparation import PreparationCancelled

    task = "e" * 32
    with patch.object(bridge, "prepare_binding", side_effect=PreparationCancelled("stopped")), patch(
        "agent.application.work_bridge.create_model"
    ) as create:
        result = bridge.run_work(
            {"task_id": task, "repo": str(tmp_path)}, lambda *args: None,
            lambda: [], root=tmp_path,
        )
    assert result["status"] == "paused"
    assert result["applied_inputs"] == []
    assert (tmp_path / "tasks" / task / "preparation-interrupted.json").is_file()
    create.assert_not_called()


def test_already_paused_job_never_starts_environment_preparation(tmp_path):
    with patch.object(bridge, "prepare_binding", side_effect=AssertionError("must not start")):
        result = bridge.run_work(
            {"task_id": "2" * 32}, lambda *args: None,
            lambda: [{"kind": "pause", "id": "before-dispatch"}], root=tmp_path,
        )
    assert result["status"] == "paused"
    assert result["applied_inputs"] == ["before-dispatch"]


def test_unconfirmed_supplement_waits_without_restore_or_browser(tmp_path):
    from agent.model.errors import ModelInterrupted

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("fixture")
    commands = []

    class SupplementModel(RecordedModel):
        def invoke(self, messages):
            if not self.calls:
                self.calls.append(messages)
                commands.append({"id": "supplement-1", "kind": "message", "text": "保持公开接口"})
                control.current().interrupt("message", "supplement-1")
                raise ModelInterrupted("message")
            assert control.current().kind is None
            return super().invoke(messages)

    model = SupplementModel()
    job = {"task_id": "f" * 32, "repo": str(repo), "request": "原要求",
           "execution_binding": {"image": "sha256:" + "b" * 64, "runtime": {}}}
    with patch.object(bridge, "require_binding", side_effect=lambda repo, value: value), patch(
        "agent.storage.recovery.restore", side_effect=AssertionError("unexpected full restore")
    ):
        result = bridge.run_work(
            job, lambda *args: None, lambda: commands, model=model,
            environment_factory=OfflineEnvironment, root=tmp_path / "state",
        )
    assert result["status"] == "awaiting_user"
    assert len(model.calls) == 1
    assert result["applied_inputs"] == ["supplement-1"]
    assert result["applied_inputs"] == ["supplement-1"]


def test_bridge_keeps_trial_freeze_edit_and_real_rework_order(tmp_path):
    from test_task_runner import CODE, Model, EXECUTION_CHECK
    from support import python_command

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def value():\n    return 0\n")
    plans = []
    for value in (2, 4):
        command = python_command(
            "from pathlib import Path; Path('sample.py').write_text('def value():\\n    return %d\\n')"
            % value
        )
        plans.append({"kind": "candidate", "note": "实现指定值", "test_code": CODE, "execution_checks": [dict(EXECUTION_CHECK)],
                      "steps": [{"id": "edit", "op": "shell", "command": command}]})
    model = Model(plans)
    events = []
    task = "1" * 32
    job = {"task_id": task, "repo": str(repo), "request": "value must return 4",
           "confirmed_requirements": fixture_delivery("value must return 4"),
           "execution_binding": {"image": "sha256:" + "b" * 64, "runtime": {}}}
    with patch.object(bridge, "require_binding", side_effect=lambda repo, value: value):
        result = bridge.run_work(
            job, lambda kind, fact: events.append(fact), lambda: [], model=model,
            environment_factory=OfflineEnvironment, root=tmp_path / "state",
        )
    assert result["status"] == "verified_success"
    assert [row["verification"]["status"] for row in result["rounds"]] == ["failed", "passed"]
    assert len(model.calls) == 4
    ordered = [event.get("event") for event in events]
    assert ordered.index("acceptance_trial_finished") < ordered.index("acceptance_frozen")
    assert ordered.index("acceptance_frozen") < ordered.index("step_started")
    directory = tmp_path / "state" / "tasks" / task
    frozen = json.loads((directory / "frozen-acceptance.json").read_text())
    assert frozen["spec"]["test_code"] == CODE
    for number, plan in enumerate(plans, 1):
        stored = json.loads((directory / "plans" / f"p{number:04}" / "plan.json").read_text())
        assert stored["plan"]["steps"][0]["command"] == plan["steps"][0]["command"]


def test_explicit_budget_stop_retains_container_receipt_without_terminal_cleanup(tmp_path):
    from test_task_runner import CODE, Model, EXECUTION_CHECK
    from support import python_command

    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'sample.py').write_text('def value():\n    return 0\n')
    command = python_command(
        "from pathlib import Path; Path('sample.py').write_text('def value():\\n    return 2\\n')"
    )
    model = Model([{
        'kind': 'candidate', 'note': '保存失败后的断点', 'test_code': CODE, 'execution_checks': [dict(EXECUTION_CHECK)],
        'steps': [{'id': 'edit', 'op': 'shell', 'command': command}],
    }])
    receipt = {'container': 'retained-container', 'stopped': True}

    class PausableEnvironment(OfflineEnvironment):
        def output_limit_stop_receipt(self):
            return receipt

    task, events = '3' * 32, []
    job = {'task_id': task, 'repo': str(repo), 'request': 'value must return 4',
           'budget': {'max_steps': 3},
           'confirmed_requirements': fixture_delivery('value must return 4'),
           'execution_binding': {'image': 'sha256:' + 'b' * 64, 'runtime': {}}}
    with patch.object(bridge, 'require_binding', side_effect=lambda repo, value: value), patch(
        'agent.storage.resources.release_task_resources',
    ) as cleanup:
        result = bridge.run_work(
            job, lambda kind, fact: events.append(fact), lambda: [], model=model,
            environment_factory=PausableEnvironment, root=tmp_path / 'state',
        )
    assert result['status'] == 'budget_exhausted'
    assert result.get('acceptance_limit') is None
    assert json.loads((tmp_path / 'state/tasks' / task / 'pause-container.json').read_text()) == receipt
    assert events[-1]['status'] == result['status']
    cleanup.assert_not_called()


def test_real_stop_cannot_continue_after_return():
    runner = object.__new__(bridge.WorkRunner)
    runner.state = {'status': 'paused'}
    runner.yield_boundary = False
    assert not runner.continue_after_return()
