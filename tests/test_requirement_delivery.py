"""确认交付物的通用边界：模型声明、原文、运行时和中途变更。"""

from copy import deepcopy
from types import SimpleNamespace
import json
import uuid

import pytest

from agent.task.requirements import confirmed, check_execution, guard, replace_requirements
from agent.application.requirements import prepare, accept
from agent.application.store import ApplicationStore


@pytest.mark.parametrize("field,value", [
    ("imports", "No external imports；use the standard library"),
    ("setup", "value = ("),
    ("expression", "emit("),
])
def test_invalid_invocation_syntax_is_a_field_validation_error(field, value):
    from agent.task.requirements import DeliveryTarget

    candidate = target()
    candidate[field] = value
    with pytest.raises(ValueError, match=field + r".*line 1.*valid Python"):
        DeliveryTarget.model_validate(candidate)


def target(kind="generated_code"):
    value = {
        "id": "output", "artifact_type": kind, "target_runtime": "c",
        "imports": "from emitter import emit", "entry": "emitter:emit", "expression": "emit(value)", "setup": "value = 0",
        "variables": ["value"], "required_values": [[0], [2]], "argv": [],
        "behavior": "The generated result must run correctly", "input_scope": "Include zero and nonzero inputs",
        "quotes": [{"source_id": "task_request", "text": "Generate runnable code"}],
    }
    if kind == "repository_entry":
        value.update(target_runtime="python", entry="app.py", imports="", expression="", setup="",
                     variables=[], required_values=[])
    return value


def packet(value=None):
    return confirmed("Generate runnable code", [value or target()],
                     confirmation={"id": "human-confirmed", "text": "Confirmed this delivery"})


def execution():
    return {
        "id": "actual_output", "requirement_id": "output", "requirement_quote": "",
        "artifact_type": "generated_code", "consumer": "c_expression",
        "imports": "from emitter import emit", "setup": "value = 0", "expression": "emit(value)",
        "variables": ["value"], "result_variable": "", "entry_file": "", "interpreter": "", "argv": [],
        "cases": [{"values": [0], "expected": 1}, {"values": [2], "expected": 3}],
    }


def loop(tmp_path, value=None):
    return SimpleNamespace(root=tmp_path, state={"requirements": value or packet(),
                                               "rounds": [{"plan": "old"}], "budget": {"tokens": 100}},
                           persist=lambda *_: None)


def test_generated_target_cannot_be_changed_to_its_python_producer(tmp_path):
    candidate = execution()
    candidate.update(artifact_type="repository_entry", consumer="command", imports="", setup="",
                     expression="", entry_file="emitter.py", interpreter="python", variables=[], cases=[])
    with pytest.raises(ValueError, match="artifact or runtime"):
        check_execution(loop(tmp_path), [candidate])


@pytest.mark.parametrize("field,value", [
    ("expression", "emit(1)"), ("setup", "value = 1"),
    ("imports", "from replacement import emit"), ("variables", []),
    ("cases", [{"values": [2], "expected": 3}]),
])
def test_entry_input_scope_and_required_cases_cannot_be_substituted(tmp_path, field, value):
    candidate = execution()
    candidate[field] = value
    with pytest.raises(ValueError):
        check_execution(loop(tmp_path), [candidate])


def test_correct_binding_allows_execution_but_does_not_claim_behavior_pass(tmp_path):
    owner = loop(tmp_path)
    assert check_execution(owner, [execution()]) == owner.state["requirements"]["digest"]
    assert "verification" not in owner.state


def test_missing_confirmation_is_inspection_only(tmp_path):
    owner = loop(tmp_path)
    owner.state["requirements"] = None
    with pytest.raises(ValueError, match="inspection only"):
        check_execution(owner, [execution()])


def test_fixed_user_command_keeps_requirement_version_binding(tmp_path):
    owner = loop(tmp_path)
    owner.state['verify_command'] = 'fixed user command'
    with pytest.raises(ValueError, match='earlier requirement'):
        guard(owner, {'requirement_digest': 'old', 'execution_checks': []})
    owner.state['requirements'] = None
    with pytest.raises(ValueError, match='inspection only'):
        guard(owner)


def test_unsupported_confirmed_runtime_reports_gap_instead_of_substitution(tmp_path):
    value = target()
    value['target_runtime'] = 'unsupported-runtime'
    with pytest.raises(ValueError, match='capability gap'):
        check_execution(loop(tmp_path, packet(value)), [execution()])


def test_original_quote_must_be_exact():
    changed = target()
    changed["quotes"][0]["text"] = "Use a different language"
    with pytest.raises(ValueError, match="exact original"):
        packet(changed)


def test_new_revision_invalidates_frozen_acceptance_without_resetting_budget(tmp_path):
    owner = loop(tmp_path)
    old = owner.state["requirements"]
    spec = {"requirement_digest": old["digest"], "execution_checks": [execution()]}
    owner.state.update(verification_spec=spec, acceptance_versions=["old"], active_acceptance={"id": "old"})
    updated = confirmed("Generate runnable code\nNew requirement", [target()], sources=old["sources"],
                        confirmation={"id": "new-human", "text": "Confirm change"}, parent=old)
    replace_requirements(owner, updated)
    assert owner.state["rounds"] == [{"plan": "old"}]
    assert owner.state["budget"] == {"tokens": 100}
    assert list((tmp_path / "requirements/history").glob("*.json"))
    with pytest.raises(ValueError, match="earlier requirement"):
        guard(owner, spec)


def test_unconfirmed_update_keeps_parent_but_cannot_release_old_acceptance(tmp_path):
    owner = loop(tmp_path)
    old = owner.state["requirements"]
    replace_requirements(owner, None)
    assert owner.state["requirements"] == old
    with pytest.raises(ValueError, match="awaiting confirmation"):
        guard(owner)
    updated = confirmed("Generate runnable code", [target()], sources=old["sources"],
                        confirmation={"id": "new-human", "text": "Confirmed"}, parent=old)
    replace_requirements(owner, updated)
    guard(owner)


def test_proposal_needs_explicit_human_confirmation_control(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = ApplicationStore(tmp_path / "state")
    conversation = store.create(repo=str(project))["thread_id"]
    user = {"id": str(uuid.uuid4()), "kind": "message", "text": "Generate runnable code"}
    store.submit(conversation, user)
    value = target()
    value["quotes"][0]["source_id"] = user["id"]
    proposal = prepare(store, conversation, user, project_path=".", create=False,
                       request_message_ids=[user["id"]], targets=[value])
    ordinary = {"id": str(uuid.uuid4()), "kind": "message", "text": "Do not start"}
    store.submit(conversation, ordinary)
    with pytest.raises(ValueError, match="human confirmation"):
        accept(store, conversation, ordinary, proposal["proposal_id"], ".", False)
    command = {"id": str(uuid.uuid4()), "kind": "confirm_work", "proposal_id": proposal["proposal_id"]}
    store.submit(conversation, command)
    contract, request = accept(store, conversation, command, proposal["proposal_id"], ".", False)
    assert request == user["text"]
    assert contract["sources"][0]["text"] == user["text"]
    assert contract["confirmation"]["id"] == command["id"]


def test_changed_proposal_retains_originals_and_rejects_a_newer_unconfirmed_update(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(project))['thread_id']
    original = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Generate runnable code'}
    store.submit(conversation, original)
    delivery = target()
    delivery['quotes'][0]['source_id'] = original['id']
    proposal = prepare(store, conversation, original, **{
        'project_path': '.', 'create': False, 'request_message_ids': [original['id']], 'targets': [delivery],
    })
    approval = {'id': str(uuid.uuid4()), 'kind': 'confirm_work', 'proposal_id': proposal['proposal_id']}
    store.submit(conversation, approval)
    first, request = accept(store, conversation, approval, proposal['proposal_id'], '.', False)
    work = store.create_work(conversation, approval['id'],
                             work_options={'confirmed_requirements': first, 'original_request': request})
    store.work_state(work['task_id'], 'paused')
    update = {'id': str(uuid.uuid4()), 'kind': 'supplement', 'text': 'Also preserve the full original domain'}
    store.submit(conversation, update)
    revised = prepare(store, conversation, update, **{
        'project_path': '.', 'create': False, 'request_message_ids': [original['id']], 'targets': [delivery],
    })
    # 即使模型漏选补充消息，宿主也把完整新原文加入方案。
    confirm = {'id': str(uuid.uuid4()), 'kind': 'confirm_work', 'proposal_id': revised['proposal_id']}
    store.submit(conversation, confirm)
    second, _ = accept(store, conversation, confirm, revised['proposal_id'], '.', False)
    assert second['revision'] == 2 and [row['id'] for row in second['sources']] == [original['id'], update['id']]
    later = {'id': str(uuid.uuid4()), 'kind': 'supplement', 'text': 'Change the additional behavior too'}
    store.submit(conversation, later)
    with pytest.raises(ValueError, match='newer original requirement'):
        accept(store, conversation, confirm, revised['proposal_id'], '.', False)
    assert store.work(work['task_id'])['confirmed_requirements'] == first


def test_only_explicit_new_human_quote_can_withdraw_a_confirmed_target(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(project))['thread_id']
    original = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Generate runnable code'}
    store.submit(conversation, original)
    first, second = target(), target()
    first['quotes'][0]['source_id'] = original['id']
    second['quotes'][0]['source_id'] = original['id']
    second['id'] = 'secondary'
    contract = confirmed(original['text'], [first, second],
                         sources=[{'id': original['id'], 'role': 'user', 'text': original['text']}],
                         confirmation={'id': 'human', 'text': 'Confirm both deliveries'})
    work = store.create_work(conversation, original['id'], work_options={'confirmed_requirements': contract})
    store.work_state(work['task_id'], 'paused')
    change = {'id': str(uuid.uuid4()), 'kind': 'supplement', 'text': 'Cancel the secondary deliverable; retain the first'}
    store.submit(conversation, change)
    options = dict(project_path='.', create=False, request_message_ids=[change['id']], targets=[first])
    with pytest.raises(ValueError, match='newly quoted user'):
        prepare(store, conversation, change, **options, withdrawals=[{
            'target_id': 'secondary', 'source_id': original['id'], 'quote': original['text'],
        }])
    proposal = prepare(store, conversation, change, **options, withdrawals=[{
        'target_id': 'secondary', 'source_id': change['id'], 'quote': change['text'],
    }])
    control = {'id': str(uuid.uuid4()), 'kind': 'confirm_work', 'proposal_id': proposal['proposal_id']}
    store.submit(conversation, control)
    packet, _ = accept(store, conversation, control, proposal['proposal_id'], '.', False)
    assert [item['id'] for item in packet['targets']] == ['output']
    assert len(packet['sources']) == 2
    notice = next(m['content'] for m in store.snapshot(conversation)['messages']
                  if m['source'] == 'work_proposal')
    assert '取消交付物：secondary' in notice and change['text'] in notice


def test_execution_feedback_reports_each_confirmed_input_field(tmp_path):
    from agent.task.repair import ValidationErrors

    candidate = execution()
    candidate.update(expression="emit(2)", variables=[],
                     cases=[{"values": [], "expected": 3}])
    with pytest.raises(ValidationErrors) as caught:
        check_execution(loop(tmp_path), [candidate])
    assert set(caught.value.paths) == {
        "/execution_checks/0/expression", "/execution_checks/0/variables",
        "/execution_checks/0/cases",
    }
    assert 'expected="emit(value)"' in str(caught.value)
    assert 'actual="emit(2)"' in str(caught.value)


def test_execution_feedback_collects_mismatches_across_targets(tmp_path):
    from agent.task.repair import ValidationErrors

    first, second = target(), target()
    second["id"] = "second"
    owner = loop(tmp_path, confirmed(
        "Generate runnable code", [first, second],
        confirmation={"id": "both", "text": "Confirm both"},
    ))
    checks = [execution(), execution()]
    checks[0]["expression"] = "emit(2)"
    checks[1].update(id="second_check", requirement_id="second", setup="value = 2")
    with pytest.raises(ValidationErrors) as caught:
        check_execution(owner, checks)
    assert set(caught.value.paths) == {
        "/execution_checks/0/expression", "/execution_checks/1/setup",
    }


def test_acceptance_feedback_repairs_independent_errors_atomically(tmp_path):
    from agent.task.runner import TaskRunner
    from agent.task import repair

    valid_code = (
        "import unittest\nfrom emitter import emit\n"
        "class Checks(unittest.TestCase):\n"
        " def test_output(self): self.assertEqual(emit(0), 1)\n"
    )
    wrong_code = valid_code.replace("emit(0), 1", "emit(0), emit(0)")
    declaration = execution()
    declaration.update(expression="emit(2)", variables=[],
                       cases=[{"values": [], "expected": 3}])
    value = {
        "kind": "candidate", "note": "Implement the confirmed behavior",
        "steps": [{"id": "edit", "op": "shell", "command": "retained-work-command"}],
        "test_code": wrong_code, "execution_checks": [declaration],
    }
    owner = TaskRunner.__new__(TaskRunner)
    owner.root = tmp_path
    owner.state = {"request": "Generate runnable code", "acceptance": None,
                   "requirements": packet(), "source": "retained-source"}
    owner.rejected_input = deepcopy(value)
    original_requirements = deepcopy(owner.state["requirements"])
    with pytest.raises(repair.ValidationErrors) as caught:
        owner.validate_work(value)
    notice = json.loads(repair.reject(owner, caught.value).split("] ", 1)[1].split("\n", 1)[0])
    assert set(notice["allowed_paths"]) == {
        "/test_code", "/execution_checks/0/expression",
        "/execution_checks/0/variables", "/execution_checks/0/cases",
    }
    assert sum("/execution_checks/0/expression:" in issue for issue in notice["issues"]) == 1
    changes = [
        ("/test_code", wrong_code, valid_code),
        ("/execution_checks/0/expression", "emit(2)", "emit(value)"),
        ("/execution_checks/0/variables", [], ["value"]),
        ("/execution_checks/0/cases", declaration["cases"], execution()["cases"]),
    ]
    corrected = repair.merge(owner, {
        "draft_id": notice["draft_id"],
        "edits": [{"path": path, "operation": "set", "before": json.dumps(before),
                   "after": json.dumps(after)} for path, before, after in changes],
    })
    owner.validate_work(corrected)
    assert corrected["steps"] == value["steps"]
    assert owner.state["requirements"] == original_requirements
    assert owner.state["source"] == "retained-source"
    with pytest.raises(ValueError, match="outside reported fields"):
        repair.merge(owner, {"draft_id": notice["draft_id"], "edits": [{
            "path": "/steps/0/command", "operation": "set",
            "before": json.dumps("retained-work-command"), "after": json.dumps("different-work"),
        }]})
    corrected["execution_checks"][0]["expression"] = "emit(2)"
    with pytest.raises(repair.ValidationErrors):
        owner.validate_work(corrected)


def test_unconfirmed_requirements_are_not_a_repairable_acceptance_field(tmp_path):
    from agent.task.runner import TaskRunner
    from agent.task import repair

    owner = TaskRunner.__new__(TaskRunner)
    owner.root = tmp_path
    owner.state = {"request": "Generate runnable code", "acceptance": None,
                   "requirements": None}
    value = {"kind": "candidate", "note": "Keep original requirement",
             "test_code": "import unittest\nclass Checks(unittest.TestCase):\n"
                          " def test_value(self): self.assertEqual(1, 1)\n",
             "execution_checks": [execution()]}
    with pytest.raises(repair.ValidationErrors) as caught:
        owner.validate_work(value)
    assert "/execution_checks" not in caught.value.paths
    assert "inspection only" in str(caught.value)
