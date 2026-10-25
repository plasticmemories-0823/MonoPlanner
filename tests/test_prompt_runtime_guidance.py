"""按已确认交付类型提供模型说明，不改变约束或已发送的系统前缀。"""

from copy import deepcopy
import json

import pytest

from agent.execution.local import LocalEnvironment
from agent.task.budget import StepBudget
from agent.task.requirements import checked, confirmed, model_view
from agent.task.runner import TaskRunner
from test_task_runner import Model


REQUEST = "Inspect the confirmed delivery."


def target(kind="python_api", runtime="python", identity="output"):
    value = {
        "id": identity,
        "artifact_type": kind,
        "target_runtime": runtime,
        "entry": "sample:value",
        "behavior": REQUEST,
        "input_scope": "All inputs in the public contract",
        "quotes": [{"source_id": "task_request", "text": REQUEST}],
    }
    if kind == "file_artifact":
        value["entry"] = "output.txt"
    elif kind == "repository_entry":
        value["entry"] = "run.cmd" if runtime == "wine-cmd" else "sample.py"
    elif kind == "generated_code":
        value.update(imports="from sample import value", expression="value()")
    return value


def packet(targets, parent=None):
    return confirmed(
        REQUEST,
        targets,
        confirmation={"id": "confirmed", "text": "Use this delivery"},
        parent=parent,
    )


@pytest.mark.parametrize("kind,runtime,needed,unrelated", [
    ("python_api", "python", "api_entry", "c_expression"),
    ("file_artifact", "python", "read_artifact", "api_entry"),
    ("generated_code", "c", "c_expression", "wine-cmd"),
    ("repository_entry", "python", "python_module", "wine-cmd"),
    ("repository_entry", "node", "entry_file", "python_module"),
    ("repository_entry", "shell", "entry_file", "python_tests"),
    ("repository_entry", "wine-cmd", "wine-cmd", "c_expression"),
])
def test_view_selects_applicable_help_without_changing_confirmed_contract(
    kind, runtime, needed, unrelated,
):
    original = packet([target(kind, runtime)])
    before = deepcopy(original)
    view = model_view(original)
    help_text = "\n".join(view["execution_guidance"].values())
    assert needed in help_text
    assert unrelated not in help_text
    assert view["targets"] == original["targets"]
    assert original == before
    assert checked(original)["digest"] == view["digest"]


def test_guidance_uses_confirmed_types_not_words_in_the_task_or_entry():
    item = target()
    item.update(entry="wine.generated_c:emit", behavior="Inspect Wine and C examples")
    help_text = "\n".join(model_view(packet([item]))["execution_guidance"].values())
    assert "python_tests" in help_text
    assert "wine-cmd" not in help_text
    assert "c_expression" not in help_text


def test_multiple_targets_share_help_without_losing_their_bindings():
    targets = [target(identity="one"), target(identity="two"), target("generated_code", "c", "c")]
    view = model_view(packet(targets))
    assert len(view["execution_guidance"]) == 2
    assert [item["id"] for item in view["targets"]] == ["one", "two", "c"]


def test_missing_or_unsupported_targets_never_invent_a_supported_adapter():
    assert model_view(None) is None
    view = model_view(packet([target("repository_entry", "unsupported-runtime")]))
    help_text = "\n".join(view["execution_guidance"].values())
    assert "capability gap" in help_text
    assert "c_expression" not in help_text
    assert "python_tests" not in help_text


def inspect_plan():
    return {
        "kind": "inspect",
        "note": "Read the real entry before deciding what to change",
        "steps": [{"id": "read", "op": "read", "path": "sample.py"}],
    }


def test_real_task_request_has_one_lookup_policy_and_applicable_runtime_help(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def value(): return 0\n")
    model = Model([inspect_plan()])
    runner = TaskRunner(
        model, LocalEnvironment(repo, artifact_root=tmp_path / "runtime"),
        state_dir=tmp_path / "state", confirmed_requirements=packet([target()]),
    )
    runner.run([], REQUEST, StepBudget(max_steps=1))
    messages = model.calls[0]
    assert messages[0].content.count("BM25") == 1
    assert "Wine CMD" not in messages[0].content
    initial = json.loads(messages[1].content)
    help_text = "\n".join(initial["confirmed_requirements"]["execution_guidance"].values())
    assert "python_tests" in help_text
    assert "imports/setup/expression" in help_text
    assert "test_ids" in help_text
    assert runner.state["rules"] == messages[0].content


def test_confirmed_runtime_update_appends_help_without_rewriting_system_prefix(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def value(): return 0\n")
    original = packet([target()])
    revised = packet([target("file_artifact")], parent=original)
    controls = []
    model = Model([inspect_plan(), inspect_plan()])
    runner = TaskRunner(
        model, LocalEnvironment(repo, artifact_root=tmp_path / "runtime"),
        state_dir=tmp_path / "state", confirmed_requirements=original,
        control_source=lambda: controls,
    )
    persist = runner.persist

    def observe(event, payload=None):
        persist(event, payload)
        if event == "plan_finished" and not controls:
            controls.append({
                "id": "new-target", "kind": "message", "text": "Use the file delivery",
                "confirmed_requirements": revised,
            })

    runner.persist = observe
    runner.run([], REQUEST, StepBudget(max_steps=2))
    assert len(model.calls) == 2
    assert model.calls[1][0].content == model.calls[0][0].content
    updates = [
        json.loads(message.content) for message in model.calls[1]
        if message.type == "human" and message.content.startswith('{"confirmed_requirements"')
    ]
    help_text = "\n".join(updates[-1]["confirmed_requirements"]["execution_guidance"].values())
    assert "read_artifact" in help_text
    assert "api_entry" not in help_text
    assert runner.state["requirements"]["digest"] == revised["digest"]
