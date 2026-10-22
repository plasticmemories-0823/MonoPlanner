"""在真实离线调度边界注入需求变更，保留已有工作并拒绝旧验收放行。"""

from copy import deepcopy
import json

import pytest

from agent.execution.local import LocalEnvironment
from agent.task.budget import StepBudget
from agent.task.requirements import confirmed, validate_state
from agent.task.runner import TaskRunner
from support import fixture_delivery, python_command
from test_task_runner import CODE, EXECUTION_CHECK, Model


def candidate(value=4):
    return {
        'kind': 'candidate', 'note': 'Implement the confirmed public behavior',
        'test_code': CODE.replace('value(), 4', f'value(), {value}'),
        'execution_checks': [deepcopy(EXECUTION_CHECK)],
        'steps': [{'id': 'edit', 'op': 'shell', 'command': python_command(
            f"from pathlib import Path; Path('sample.py').write_text('def value():\\n    return {value}\\n')"
        )}],
    }


@pytest.mark.parametrize('boundary', [
    'acceptance_trial_started', 'acceptance_trial_finished', 'acceptance_frozen',
    'plan_accepted', 'step_finished', 'final_acceptance_finished',
])
def test_late_human_requirement_invalidates_old_acceptance_at_each_boundary(tmp_path, monkeypatch, boundary):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'sample.py').write_text('def value():\n    return 0\n')
    contract = fixture_delivery('Return the public value 4')
    controls, triggered = [], []
    model = Model([candidate()])
    runner = TaskRunner(model, LocalEnvironment(repo, artifact_root=tmp_path / 'runtime'),
                        state_dir=tmp_path / 'task', confirmed_requirements=contract,
                        control_source=lambda: controls)
    persist = runner.persist

    def observe(event, payload=None):
        persist(event, payload)
        if event == boundary and not triggered:
            triggered.append(True)
            controls.append({'id': 'original-change', 'kind': 'message', 'text': 'The public value must now be 5'})

    monkeypatch.setattr(runner, 'persist', observe)
    result = runner.run([], 'Return the public value 4', StepBudget(max_steps=12))
    assert triggered and result['status'] == 'awaiting_user', result
    assert runner.state['requirements_pending'] is True
    assert runner.state.get('verification_spec') is None
    assert runner.state['applied_inputs'] == ['original-change']
    assert runner.state['budget']['used_steps'] == len(model.calls)
    assert len(result['rounds']) == (1 if boundary == 'final_acceptance_finished' else 0)
    assert (repo / 'sample.py').read_text().endswith(
        'return 4\n' if boundary in {'step_finished', 'final_acceptance_finished'} else 'return 0\n'
    )
    validate_state(runner)


def test_confirmed_revision_retrials_after_real_work_without_resetting_budget(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'sample.py').write_text('def value():\n    return 0\n')
    request = 'Return the public value 4'
    original = fixture_delivery(request)
    controls = []
    env = LocalEnvironment(repo, artifact_root=tmp_path / 'runtime')
    first = TaskRunner(Model([candidate()]), env, state_dir=tmp_path / 'task',
                       confirmed_requirements=original, control_source=lambda: controls,
                       pause_after_steps=1)
    initial = first.run([], request, StepBudget(max_steps=16))
    assert initial['status'] == 'paused'
    first_budget = dict(first.state['budget'])
    targets = deepcopy(original['targets'])
    change = {'id': 'new-request', 'role': 'user', 'text': 'The public value must now be 5'}
    targets[0]['behavior'] = change['text']
    targets[0]['quotes'].append({'source_id': change['id'], 'text': change['text']})
    revised = confirmed(request + '\n\n' + change['text'], targets,
                        sources=[*original['sources'], change],
                        confirmation={'id': 'new-confirmation', 'text': 'Confirm the revised public behavior'},
                        parent=original)
    controls.append({'id': 'confirmed-update', 'kind': 'message', 'text': change['text'],
                     'confirmed_requirements': revised})
    second = TaskRunner(Model([candidate(5)]), env, state_dir=tmp_path / 'task',
                        confirmed_requirements=original, control_source=lambda: controls)
    final = second.run([], request, StepBudget(max_steps=16))
    assert final['status'] == 'verified_success', final
    assert second.state['requirements']['revision'] == 2
    assert second.state['budget']['used_steps'] > first_budget['used_steps']
    assert (repo / 'sample.py').read_text().endswith('return 5\n')
    events = [json.loads(row)['kind'] for row in (second.root / 'events.jsonl').read_text().splitlines()]
    assert events.count('acceptance_trial_started') == 2
    assert events.count('step_finished') == 2
    assert second.state['retired_requirements']
    validate_state(second)
