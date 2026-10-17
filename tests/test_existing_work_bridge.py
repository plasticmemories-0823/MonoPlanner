"""已有任务直接测试当前执行服务；不保留旧聊天调度来驱动进程。"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.application.work_bridge import _existing
from tests.fixtures.models import FixtureDocker


def prepared(tmp_path, code, *, timeout=5, usage_file=''):
    repo, task = tmp_path / 'repo', tmp_path / 'task'
    repo.mkdir()
    task.mkdir()
    (repo / 'task.py').write_text(code, encoding='utf-8')
    environment = FixtureDocker(repo, 'sha256:' + 'a' * 64, task)
    job = {'task_id': 'a' * 32, 'repo': str(repo), 'arguments': {
        'entry': 'task.py', 'args': [], 'timeout': timeout, 'usage_file': usage_file,
    }}
    return job, environment, task, {'pause': None}


def invoke(fixture):
    job, environment, task, owner = fixture
    return _existing(job, environment, task, owner, lambda *event: None)


def test_existing_failure_is_recorded_once_without_coding_or_replay(tmp_path):
    fixture = prepared(tmp_path, "print('retained failure')\nraise SystemExit(3)\n")
    job, environment, task, _ = fixture
    source = (Path(job['repo']) / 'task.py').read_bytes()
    with patch.object(environment, 'run_process', wraps=environment.run_process) as run:
        result = invoke(fixture)
        repeated = invoke(fixture)
    assert result == repeated and run.call_count == 1
    assert result['status'] == 'existing_complete'
    assert result['existing_result']['exit_code'] == 3
    assert 'retained failure' in result['existing_result']['stdout']
    assert (Path(job['repo']) / 'task.py').read_bytes() == source
    assert not (task / 'frozen-acceptance.json').exists()
    assert not (task / 'plans').exists()


def test_existing_timeout_preserves_partial_output_and_does_not_replay(tmp_path):
    fixture = prepared(tmp_path, "import time\nprint('BEFORE_TIMEOUT', flush=True)\ntime.sleep(5)\n", timeout=1)
    with patch.object(fixture[1], 'run_process', wraps=fixture[1].run_process) as run:
        result = invoke(fixture)
        repeated = invoke(fixture)
    assert result == repeated and run.call_count == 1
    assert result['existing_result']['timed_out'] is True
    assert 'BEFORE_TIMEOUT' in result['existing_result']['stdout']


@pytest.mark.parametrize('rewritten', [False, True])
def test_child_usage_requires_a_new_task_report_and_is_not_parent_model_usage(tmp_path, rewritten):
    report = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
              'cached_input_tokens': 90}
    code = ('from pathlib import Path\nPath("usage.json").write_text(' + repr(json.dumps(report)) + ')\n'
            if rewritten else "print('keep original report')\n")
    fixture = prepared(tmp_path, code, usage_file='usage.json')
    (Path(fixture[0]['repo']) / 'usage.json').write_text(json.dumps({'old': True}))
    child = invoke(fixture)['existing_result']['child_usage']
    assert child['status'] == ('reported_by_task' if rewritten else 'unknown')
    if rewritten:
        assert child['total_tokens'] == 120 and child['cached_input_tokens'] == 90
    else:
        assert 'total_tokens' not in child
    assert 'model_usage' not in json.loads((fixture[2] / 'existing/state.json').read_text())


def test_interrupted_existing_process_record_blocks_reexecution(tmp_path):
    fixture = prepared(tmp_path, "print('one attempt')\n")
    with patch.object(fixture[1], 'run_process', side_effect=OSError('unknown execution')) as run:
        with pytest.raises(OSError, match='unknown execution'):
            invoke(fixture)
        resumed = invoke(fixture)
    assert resumed['status'] == 'execution_unknown' and run.call_count == 1
    saved = json.loads((fixture[2] / 'existing/state.json').read_text())
    assert saved['status'] == 'running'
