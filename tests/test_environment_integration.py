"""按需准备环境时保留现有聊天和执行交接。"""

import json
import os
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from starlette.testclient import TestClient as RawClient

from agent.execution.projects import ProjectEnvironments, prepare_binding, require_binding
from tests.http_client import TestClient


def binding():
    return {
        'backend': 'docker',
        'image': 'sha256:' + 'c' * 64,
        'runtime': {'python': '/usr/local/bin/python', 'conda_env': ''},
    }


def test_manual_or_frozen_binding_cannot_be_replaced_by_automatic_defaults(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path / 'state'))
    path = ProjectEnvironments().path
    path.parent.mkdir(parents=True)
    original = binding()
    path.write_text(json.dumps({str(repo): original}))
    with patch(
        'agent.execution.preparation.ensure_prepared',
        side_effect=AssertionError('manual is explicit'),
    ):
        frozen = prepare_binding(str(repo))
    ProjectEnvironments().automatic(str(repo))
    assert require_binding(str(repo), frozen) == original


def test_environment_routes_are_authenticated_read_only_on_get_and_strict_on_actions(
    tmp_path, monkeypatch
):
    from agent.execution import preparation
    from agent.server.app import app

    repo = tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path / 'state'))
    service = SimpleNamespace(
        status=Mock(return_value={'status': 'needed', 'mode': 'auto'}),
        start=Mock(return_value={'status': 'preparing', 'mode': 'auto'}),
        retry=Mock(return_value={'status': 'preparing', 'mode': 'auto'}),
        cancel=Mock(return_value={'status': 'cancelled', 'mode': 'auto'}),
    )
    monkeypatch.setattr(preparation, 'jobs', lambda: service)
    base = 'http://127.0.0.1:' + os.getenv('MONOPLANER_BACKEND_PORT', '2024')
    with RawClient(app, base_url=base) as raw:
        assert raw.get('/workspace/environment', params={'repo': str(repo)}).status_code == 401
        assert (
            raw.post(
                '/workspace/environment', json={'repo': str(repo), 'action': 'prepare'}
            ).status_code
            == 401
        )
    service.status.assert_not_called()
    service.start.assert_not_called()
    service.retry.assert_not_called()
    with TestClient(app) as client:
        response = client.get('/workspace/environment', params={'repo': str(repo)})
        assert response.status_code == 200 and response.json()['status'] == 'needed'
        service.start.assert_not_called()
        assert (
            client.post(
                '/workspace/environment',
                json={'repo': str(repo), 'action': 'prepare'},
                headers={'Origin': 'https://foreign.example'},
            ).status_code
            == 403
        )
        service.start.assert_not_called()
        for value in (
            {'repo': str(repo)},
            {'repo': str(repo), 'action': 'shell'},
            {'repo': str(repo), 'action': 'prepare', 'command': 'untrusted'},
        ):
            assert client.post('/workspace/environment', json=value).status_code == 400
        assert (
            client.post(
                '/workspace/environment', json={'repo': str(repo), 'action': 'prepare'}
            ).status_code
            == 200
        )
        assert (
            client.post(
                '/workspace/environment', json={'repo': str(repo), 'action': 'cancel'}
            ).status_code
            == 200
        )
    service.start.assert_not_called()
    service.retry.assert_called_once_with(str(repo))
    service.cancel.assert_called_once_with(str(repo))


def test_environment_route_constructors_and_filesystem_work_stay_off_async_loop(
    tmp_path, monkeypatch
):
    from agent.execution import preparation
    from agent.server import app as server_app

    repo = tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path / 'state'))
    observed = []

    def off_loop(operation):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            observed.append(operation)
        else:
            pytest.fail(operation + ' must run in a worker thread, not the ASGI event loop')

    def status(path, *, verify=False):
        off_loop('status')
        if verify:
            off_loop('verify_resources')
        assert path == str(repo)
        return {'mode': 'auto', 'status': 'idle'}

    def retry(path):
        off_loop('retry')
        assert path == str(repo)
        return {'mode': 'auto', 'status': 'preparing'}

    def cancel(path):
        off_loop('cancel')
        assert path == str(repo)
        return {'mode': 'auto', 'status': 'cancelled'}

    def service():
        off_loop('jobs_constructor')
        return SimpleNamespace(status=status, retry=retry, cancel=cancel)

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
        assert client.get('/workspace/environment', params={
            'repo': str(repo), 'verify': '1'}).status_code == 200
        for action in ('prepare', 'cancel', 'automatic'):
            response = client.post(
                '/workspace/environment', json={'repo': str(repo), 'action': action}
            )
            assert response.status_code == 200
    assert observed.count('jobs_constructor') == 5
    assert observed.count('status') == 3
    assert observed.count('verify_resources') == 1
    assert observed.count('retry') == observed.count('cancel') == 1
    assert (
        observed.count('project_preferences_constructor') == observed.count('save_automatic') == 1
    )
