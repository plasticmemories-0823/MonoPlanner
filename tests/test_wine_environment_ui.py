"""Wine 安装来自用户环境操作，普通查询、模型工具及失败恢复不能代为安装。"""

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.execution.projects import ProjectEnvironments, EnvironmentRequired
from tests.test_environment_jobs import job_fixture, settled


def test_wine_install_is_explicit_shared_and_persistent(job_fixture):
    f = job_fixture
    assert not ProjectEnvironments(f.root).get(f.repos[0]).get('wine_cmd')
    first = f.jobs.install_wine(f.repos[0])
    assert f.entered.wait(2)
    second = f.jobs.install_wine(f.repos[0])
    assert second['id'] == first['id']
    assert first['wine_install_authorized'] is True
    assert ProjectEnvironments(f.root).get(f.repos[0])['wine_cmd'] is True
    assert f.calls == [str(f.repos[0])]
    assert f.jobs.status(f.repos[0])['wine_cmd'] is True
    f.release.set()
    settled(f.jobs)
    assert f.jobs.status(f.repos[0])['status'] == 'ready'


def test_install_does_not_replace_an_active_plain_job_or_other_project(job_fixture):
    f = job_fixture
    f.jobs.start(f.repos[0])
    assert f.entered.wait(2)
    for repo in f.repos[:2]:
        with pytest.raises(EnvironmentRequired):
            f.jobs.install_wine(repo)
        assert not ProjectEnvironments(f.root).get(repo).get('wine_cmd')


def test_failed_install_is_readable_without_automatic_retry(job_fixture):
    f = job_fixture
    f.behavior.update(block=False, error=EnvironmentRequired('下载失败'))
    f.jobs.install_wine(f.repos[0])
    settled(f.jobs)
    for _ in range(2):
        assert f.jobs.status(f.repos[0])['status'] == 'failed'
    f.jobs.start(f.repos[0])
    assert len(f.calls) == 1
    f.behavior['error'] = None
    f.jobs.install_wine(f.repos[0])
    settled(f.jobs)
    assert len(f.calls) == 2


def test_wine_install_can_be_cancelled_without_becoming_ready(job_fixture):
    f = job_fixture
    f.jobs.install_wine(f.repos[0])
    assert f.entered.wait(2)
    f.jobs.cancel(f.repos[0])
    settled(f.jobs)
    value = f.jobs.status(f.repos[0])
    assert value['status'] == 'cancelled' and value['wine_cmd'] is True
    assert not value.get('environment')
    assert len(f.calls) == 1


def test_wine_option_changes_only_its_environment_fingerprint(tmp_path):
    from agent.execution.managed import ManagedEnvironments
    repo = tmp_path / 'repo'
    repo.mkdir()
    root = tmp_path / 'state'
    manager = ManagedEnvironments(root)
    before = manager.status(repo)['fingerprint']
    with patch('subprocess.Popen', side_effect=AssertionError('settings may not execute')):
        ProjectEnvironments(root).enable_wine(repo)
    after = manager.status(repo)['fingerprint']
    assert after != before
    assert manager.status(repo)['state'] == 'needs_preparation'
    ProjectEnvironments(root).automatic(repo)
    assert manager.status(repo)['fingerprint'] == before


def test_manual_image_is_not_silently_replaced(job_fixture, monkeypatch):
    f = job_fixture
    monkeypatch.setattr(ProjectEnvironments, 'get', lambda *_: f.binding)
    with pytest.raises(EnvironmentRequired, match='手动'):
        f.jobs.install_wine(f.repos[0])
    assert not f.calls


def test_closed_manager_cannot_enable_or_install_wine(job_fixture):
    f = job_fixture
    f.jobs.close()
    with pytest.raises(EnvironmentRequired):
        f.jobs.install_wine(f.repos[0])
    assert not ProjectEnvironments(f.root).get(f.repos[0]).get('wine_cmd')


def test_only_authenticated_explicit_button_request_can_install(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from starlette.testclient import TestClient as RawClient
    from tests.http_client import TestClient
    from agent.execution import preparation
    from agent.server.app import app
    import os

    repo = tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path / 'state'))
    jobs = SimpleNamespace(install_wine=Mock(return_value={'status': 'preparing', 'wine_cmd': True}),
                           status=Mock(return_value={'status': 'idle'}))
    monkeypatch.setattr(preparation, 'jobs', lambda: jobs)
    body = {'repo': str(repo), 'action': 'install_wine'}
    with RawClient(app, base_url='http://127.0.0.1:' + os.getenv('MONOPLANER_BACKEND_PORT', '2024')) as client:
        assert client.post('/workspace/environment', json=body).status_code == 401
    with TestClient(app) as client:
        assert client.get('/workspace/environment', params={'repo': str(repo)}).status_code == 200
        assert client.post('/workspace/environment', json=body,
                           headers={'Origin': 'https://foreign.example'}).status_code == 403
        assert client.post('/workspace/environment', json={**body, 'command': 'anything'}).status_code == 400
        jobs.install_wine.assert_not_called()
        assert client.post('/workspace/environment', json=body).json()['wine_cmd'] is True
    jobs.install_wine.assert_called_once_with(str(repo))


def test_wine_setup_uses_trusted_files_and_requires_real_probe_receipt(tmp_path):
    from agent.execution.wine import copy_installer, recipe_key, verify_receipt
    from agent.execution.managed_plan import PreparationError
    from agent.settings import PROJECT_ROOT
    destination = tmp_path / 'installer'
    copy_installer(destination, recipe_key())
    for name in ('install-wine.py', 'compact-wine-prefix.py', 'wine'):
        assert (destination / name).read_bytes() == (PROJECT_ROOT / 'resources/environments' / name).read_bytes()
    with pytest.raises(PreparationError):
        copy_installer(tmp_path / 'other', 'changed')
    valid = {'version': 'wine-8.0 (Debian 8.0~repack-4)',
             'cmd_exit_code': 0, 'cmd_probe': 'MONOPLANNER_WINE_READY'}
    verify_receipt(valid)
    for invalid in (None, {}, {**valid, 'cmd_exit_code': 1}, {**valid, 'cmd_probe': 'different'}):
        with pytest.raises(PreparationError):
            verify_receipt(invalid)


def test_capabilities_reads_ready_automatic_wine_without_installing(tmp_path, monkeypatch):
    import json
    from agent.acceptance.capabilities import describe
    from agent.execution.managed import ManagedEnvironments
    from agent.execution import projects
    repo = tmp_path / 'repo'
    repo.mkdir()
    root = tmp_path / 'state'
    image = 'sha256:' + 'b' * 64
    monkeypatch.setattr(ManagedEnvironments, 'status', lambda *_: {'state': 'ready', 'image': image})
    monkeypatch.setattr(ManagedEnvironments, 'prepare', lambda *_: pytest.fail('read triggered installation'))
    monkeypatch.setattr(projects, 'docker_query', lambda args: json.dumps([
        {'Id': image, 'Os': 'linux', 'Architecture': 'amd64',
         'Config': {'Labels': {'org.monoplanner.wine-cmd': '1'}}}]))
    value = describe(repo=repo, root=root)['wine_cmd']
    assert value['available'] is True
    assert value['configuration']['installer'] == 'user_button_only'


def test_work_preparation_cannot_install_missing_wine_without_the_button(tmp_path):
    from agent.execution.managed import ManagedEnvironments
    from agent.execution.managed_plan import ProjectPlan, PreparationError
    from agent.execution.wine import recipe_key
    manager = ManagedEnvironments(tmp_path / 'state')
    work = tmp_path / 'attempt'
    work.mkdir()
    plan = ProjectPlan(root=tmp_path / 'repo', recipe={'wine_cmd': recipe_key()})
    with patch.object(manager, '_setup', side_effect=AssertionError('not authorized')):
        with pytest.raises(PreparationError, match='点击安装 Wine'):
            manager._empty_environment(plan, 'sha256:' + 'a' * 64, work, lambda *_: None, None)


def test_installed_wine_is_shared_but_project_install_keeps_restricted_capabilities(tmp_path):
    import json
    from agent.execution.managed_plan import ProjectPlan
    from agent.execution.wine import recipe_key
    from tests.test_managed_environment import FakeDockerManager

    class WineDocker(FakeDockerManager):
        def _run(self, arguments, **kwargs):
            result = super()._run(arguments, **kwargs)
            if arguments[0] == 'cp' and arguments[1].endswith(':/monoplanner-result.json'):
                path = Path(arguments[2])
                value = json.loads(path.read_text('utf-8'))
                value['wine_cmd'] = {'version': 'wine-8.0 (Debian 8.0~repack-4)',
                                    'cmd_exit_code': 0, 'cmd_probe': 'MONOPLANNER_WINE_READY'}
                path.write_text(json.dumps(value), 'utf-8')
            return result

    manager = WineDocker(tmp_path / 'state')
    plan = ProjectPlan(root=tmp_path / 'repo', materials={}, recipe={'wine_cmd': recipe_key()})
    work = tmp_path / 'attempt'
    work.mkdir()
    binding = manager._empty_environment(plan, 'sha256:' + 'a' * 64, work,
                                        lambda *_: None, None, allow_wine_install=True)
    assert binding['managed']['wine_cmd'] is True
    assert len([args for args in manager.calls if args[0] == 'start']) == 1
    installer = next(args for args in manager.calls if args[0] == 'create' and '--name' in args)
    assert '--cap-drop' in installer and '--cap-add' in installer
    assert '--network' in installer and '--privileged' not in installer
    proxy = next(args for args in manager.calls if args[:2] == ['run', '--detach'])
    assert '--cap-add' not in proxy
    commit = next(args for args in manager.calls if args[0] == 'commit')
    assert 'LABEL org.monoplanner.wine-cmd=1' in commit
    assert (work / 'wine/install-wine.py').is_file()
    assert not list((work / 'source').iterdir())
    with patch.object(manager, '_setup', side_effect=AssertionError('must reuse')):
        again = manager._empty_environment(plan, 'sha256:' + 'a' * 64, work, lambda *_: None, None)
    assert again == binding
