"""Lazy preparation preserves the existing chat and execution handoff."""
import json
import os
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
import pytest
from starlette.testclient import TestClient as RawClient

from agent.execution.preparation import PreparationCancelled
from agent.execution.projects import ProjectEnvironments, prepare_binding, require_binding
from agent.storage.history import LocalHistory
from tests.http_client import TestClient
from tests import test_web_graph as graph_fixture


@pytest.fixture
def conversation():
    fixture = graph_fixture.ConversationTests()
    fixture.setUp()
    (fixture.root/'runs/project-environments.json').unlink()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def binding():
    return {'backend': 'docker', 'image': 'sha256:' + 'c' * 64,
            'runtime': {'python': '/usr/local/bin/python', 'conda_env': ''}}


def checkpoint(fixture, task):
    store = LocalHistory(fixture.root/'runs/tasks'/task)
    try:
        with store.lease():
            return store.load()
    finally:
        store.close()


def test_plain_chat_with_no_environment_never_prepares_or_starts_docker(conversation):
    model = graph_fixture.ScriptedModel([graph_fixture.text_reply('你好')])
    with patch('agent.execution.preparation.ensure_prepared', side_effect=AssertionError('chat must stay lazy')), \
            patch.object(graph_fixture.FixtureDocker, 'start', side_effect=AssertionError('no Docker')):
        result = conversation.invoke(model)
    assert result['status'] == 'chat_complete'
    assert len(model.calls) == 1


def test_automatic_preparation_precedes_work_and_preserves_handoff(conversation):
    prepared = []

    def prepare(repo, *, progress, cancelled):
        assert not cancelled()
        assert not list((conversation.root/'runs/tasks').glob('*/baseline/*'))
        prepared.append(repo)
        progress({'status': 'preparing', 'stage': 'installing', 'message': '正在准备项目环境'})
        return binding()

    model = graph_fixture.ScriptedModel([graph_fixture.start_reply(),
        graph_fixture.investigation(['first.py']), graph_fixture.text_reply('已读取')])
    with patch('agent.execution.preparation.ensure_prepared', side_effect=prepare):
        result = conversation.invoke(model, '请直接读取项目')
    assert result['status'] != 'runtime_error'
    assert len(prepared) == 1
    assert len(model.calls) == 3
    saved = checkpoint(conversation, result['task_id'])
    assert saved['execution_binding'] == binding()
    assert saved['baseline']
    assert saved['environment_preparation']['stage'] == 'installing'
    assert sum(m.type == 'ai' and '开始准备工作资料' in m.content for m in result['messages']) == 1


def test_pause_during_preparation_preserves_work_and_resume_does_not_reask_model(conversation):
    calls = []

    def prepare(repo, *, progress, cancelled):
        calls.append(repo)
        if len(calls) == 1:
            conversation.store.enqueue(conversation.thread, 'pause-env', 'pause')
            assert cancelled()
            raise PreparationCancelled('环境准备等待已中断')
        return binding()

    model = graph_fixture.ScriptedModel([graph_fixture.start_reply(),
        graph_fixture.investigation(['first.py']), graph_fixture.text_reply('已读取')])
    with patch('agent.execution.preparation.ensure_prepared', side_effect=prepare):
        first = conversation.invoke(model, '请直接读取项目')
        assert first['status'] == 'paused'
        assert len(model.calls) == 1
        saved = checkpoint(conversation, first['task_id'])
        assert saved['coding_preparation_pending'] and not saved.get('baseline')
        conversation.store.enqueue(conversation.thread, 'resume-env', 'resume')
        with patch('agent.server.graph.model_factory', return_value=model):
            result = conversation.compiled.invoke({'messages': []}, conversation.config)
    assert result['task_id'] == first['task_id']
    assert result['status'] != 'runtime_error'
    assert len(model.calls) == 3
    assert len(list((conversation.root/'runs/tasks'/first['task_id']).glob('plans/*/plan.json'))) == 1


def test_existing_entry_pause_during_preparation_does_not_start_or_consume_call(conversation):
    reply = AIMessage(content='运行现有脚本。', tool_calls=[{'name': 'run_project_task', 'id': 'run-once',
        'args': {'entry': 'first.py', 'args': [], 'timeout': 10, 'usage_file': ''}}],
        usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2})
    model = graph_fixture.ScriptedModel([reply, graph_fixture.text_reply('脚本已运行')])

    def cancelled(repo, **kwargs):
        conversation.store.enqueue(conversation.thread, 'pause-env', 'pause')
        raise PreparationCancelled('环境准备等待已中断')

    with patch('agent.execution.preparation.ensure_prepared', side_effect=cancelled), \
            patch.object(graph_fixture.FixtureDocker, 'run_process', side_effect=AssertionError('not started')):
        result = conversation.invoke(model, '直接运行现有脚本')
    assert result['status'] == 'paused'
    saved = checkpoint(conversation, result['task_id'])
    assert saved['assistant_pending']['cursor'] == 0
    assert not saved['assistant_running'] and not saved.get('assistant_process')
    assert not saved.get('existing_runs')
    assert len(model.calls) == 1
    executions = []

    def execute(environment, command, *, stdout_path, stderr_path, timeout=None, shell=False, on_started=None):
        executions.append(command)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text('done\n')
        stderr_path.write_text('')
        if on_started:
            on_started('f' * 64)
        return {'exit_code': 0, 'timed_out': False}

    conversation.store.enqueue(conversation.thread, 'resume-env', 'resume')
    with patch('agent.execution.preparation.ensure_prepared', return_value=binding()), \
            patch.object(graph_fixture.FixtureDocker, 'run_process', execute), \
            patch('agent.server.graph.model_factory', return_value=model):
        resumed = conversation.compiled.invoke({'messages': []}, conversation.config)
    assert resumed['status'] == 'chat_complete'
    assert resumed['task_id'] == result['task_id']
    assert len(executions) == 1 and len(model.calls) == 2
    saved = checkpoint(conversation, result['task_id'])
    assert saved['assistant_pending'] is None
    assert len(saved['existing_runs']) == 1


def test_manual_or_frozen_binding_cannot_be_replaced_by_automatic_defaults(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path/'state'))
    path = ProjectEnvironments().path
    path.parent.mkdir(parents=True)
    original = binding()
    path.write_text(json.dumps({str(repo): original}))
    with patch('agent.execution.preparation.ensure_prepared', side_effect=AssertionError('manual is explicit')):
        frozen = prepare_binding(str(repo))
    ProjectEnvironments().automatic(str(repo))
    assert require_binding(str(repo), frozen) == original


def test_environment_routes_are_authenticated_read_only_on_get_and_strict_on_actions(tmp_path, monkeypatch):
    from agent.execution import preparation
    from agent.server.app import app
    repo = tmp_path/'repo'; repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path/'state'))
    service = SimpleNamespace(status=Mock(return_value={'status': 'needed', 'mode': 'auto'}),
        start=Mock(return_value={'status': 'preparing', 'mode': 'auto'}),
        cancel=Mock(return_value={'status': 'cancelled', 'mode': 'auto'}))
    monkeypatch.setattr(preparation, 'jobs', lambda: service)
    base = 'http://127.0.0.1:' + os.getenv('MONOPLANER_BACKEND_PORT', '2024')
    with RawClient(app, base_url=base) as raw:
        assert raw.get('/workspace/environment', params={'repo': str(repo)}).status_code == 401
        assert raw.post('/workspace/environment', json={'repo': str(repo), 'action': 'prepare'}).status_code == 401
    service.status.assert_not_called()
    service.start.assert_not_called()
    with TestClient(app) as client:
        response = client.get('/workspace/environment', params={'repo': str(repo)})
        assert response.status_code == 200 and response.json()['status'] == 'needed'
        service.start.assert_not_called()
        assert client.post('/workspace/environment', json={'repo': str(repo), 'action': 'prepare'},
            headers={'Origin': 'https://foreign.example'}).status_code == 403
        service.start.assert_not_called()
        for value in ({'repo': str(repo)}, {'repo': str(repo), 'action': 'shell'},
                      {'repo': str(repo), 'action': 'prepare', 'command': 'untrusted'}):
            assert client.post('/workspace/environment', json=value).status_code == 400
        assert client.post('/workspace/environment', json={'repo': str(repo), 'action': 'prepare'}).status_code == 200
        assert client.post('/workspace/environment', json={'repo': str(repo), 'action': 'cancel'}).status_code == 200
    service.start.assert_called_once_with(str(repo))
    service.cancel.assert_called_once_with(str(repo))


def test_environment_route_constructors_and_filesystem_work_stay_off_async_loop(tmp_path, monkeypatch):
    from agent.execution import preparation
    from agent.server import app as server_app
    repo = tmp_path/'repo'; repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path/'state'))
    observed = []

    def off_loop(operation):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            observed.append(operation)
        else:
            pytest.fail(operation + ' must run in a worker thread, not the ASGI event loop')

    def status(path):
        off_loop('status')
        assert path == str(repo)
        return {'mode': 'auto', 'status': 'idle'}

    def start(path):
        off_loop('start')
        assert path == str(repo)
        return {'mode': 'auto', 'status': 'preparing'}

    def cancel(path):
        off_loop('cancel')
        assert path == str(repo)
        return {'mode': 'auto', 'status': 'cancelled'}

    def service():
        off_loop('jobs_constructor')
        return SimpleNamespace(status=status, start=start, cancel=cancel)

    class GuardedProjectEnvironments(ProjectEnvironments):
        def __init__(self, *args, **kwargs):
            off_loop('project_preferences_constructor')
            super().__init__(*args, **kwargs)

        def automatic(self, path):
            off_loop('save_automatic')
            return super().automatic(path)

    monkeypatch.setattr(preparation, 'jobs', service)
    monkeypatch.setattr(server_app, 'ProjectEnvironments', GuardedProjectEnvironments)
    with TestClient(server_app.app) as client:
        assert client.get('/workspace/environment', params={'repo': str(repo)}).status_code == 200
        for action in ('prepare', 'cancel', 'automatic'):
            response = client.post('/workspace/environment', json={'repo': str(repo), 'action': action})
            assert response.status_code == 200
    assert observed.count('jobs_constructor') == 4
    assert observed.count('status') == 2
    assert observed.count('start') == observed.count('cancel') == 1
    assert observed.count('project_preferences_constructor') == observed.count('save_automatic') == 1
