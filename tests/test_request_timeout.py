"""模型超时配置必须贯穿主请求、归档包装与同步摘要，完全离线。"""

import pytest

from agent.model import client
from agent.model.errors import ModelRequestNotSent
from eval.recording import RecordingModel


@pytest.mark.parametrize('configured,remaining,expected', [
    (90, None, 90), (180, 240, 180), (30, 240, 30), (180, 12, 12),
])
def test_policy_and_task_deadline(configured, remaining, expected):
    model = client.ConfiguredModel(object(), client.ModelPolicy(api_timeout=configured))
    assert client.request_timeout(model, remaining) == expected


def test_explicit_request_override_and_wrapper_are_preserved():
    model = client.ConfiguredModel(object(), client.ModelPolicy(api_timeout=180), timeout=25)
    wrapper = object.__new__(RecordingModel)
    wrapper.model = model
    assert client.request_timeout(wrapper, 100) == 25
    assert client.request_timeout(object()) == 90


@pytest.mark.parametrize('remaining', [0, -0.1])
def test_exhausted_task_is_not_dispatched(remaining):
    with pytest.raises(ModelRequestNotSent, match='time'):
        client.request_timeout(client.ConfiguredModel(object()), remaining)


@pytest.mark.parametrize('task_seconds', [None, 60])
def test_actual_work_request_obeys_configured_timeout(tmp_path, task_seconds):
    from agent.execution.local import LocalEnvironment
    from agent.task.budget import StepBudget
    from agent.task.runner import TaskRunner
    from test_task_runner import Model

    class TimedModel(Model):
        policy = client.ModelPolicy(api_timeout=180)

        def for_purpose(self, purpose, timeout=None):
            calls.append((purpose, timeout))
            return self

    calls = []
    repo = tmp_path / 'project'
    repo.mkdir()
    model = TimedModel([{'kind': 'blocked', 'note': 'Offline timeout verification.'}])
    result = TaskRunner(
        model, LocalEnvironment(repo, artifact_root=tmp_path / 'runtime'),
        state_dir=tmp_path / 'state',
    ).run([], 'Inspect only.', StepBudget(timeout_seconds=task_seconds))
    assert result['status'] == 'task_cancelled'
    assert len(model.calls) == 1
    if task_seconds is None:
        assert calls and all(purpose in {'plan', 'plan_repair'} and timeout == 180 for purpose, timeout in calls)
    else:
        assert calls and all(purpose in {'plan', 'plan_repair'} and 0 < timeout <= task_seconds
                             for purpose, timeout in calls)
