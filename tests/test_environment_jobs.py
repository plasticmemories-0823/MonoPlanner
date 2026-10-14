"""环境准备任务可共享准备工作，但不共享模型或源码执行。"""

import json
import hashlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.execution.preparation import PreparationCancelled, PreparationJobs
from agent.execution.projects import EnvironmentRequired, ProjectEnvironments
from agent.storage.preferences import atomic_write


@pytest.fixture
def job_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'state'
    repos = [tmp_path / f'project-{i}' for i in range(3)]
    for repo in repos:
        repo.mkdir()
    entered, release, calls, ready = threading.Event(), threading.Event(), [], {}
    binding = {
        'backend': 'docker',
        'image': 'sha256:' + 'a' * 64,
        'runtime': {'python': '/usr/local/bin/python', 'conda_env': ''},
    }
    behavior = {'error': None, 'block': True}

    class FakeManaged:
        def __init__(self, selected_root):
            assert selected_root == root

        def status(self, repo):
            return (
                {'state': 'ready', 'stage': 'ready', 'environment': ready[repo]}
                if repo in ready
                else {'state': 'needs_preparation', 'stage': 'checking'}
            )

        def prepare(self, repo, *, progress, cancelled, project_scope):
            project_scope.validate()
            calls.append(repo)
            entered.set()
            progress({'stage': 'installing', 'message': '正在安装项目依赖'})
            while behavior['block'] and not release.wait(0.01):
                if cancelled():
                    raise PreparationCancelled('cancelled')
            if behavior['error']:
                raise behavior['error']
            ready[repo] = dict(binding)
            return binding

    monkeypatch.setitem(
        sys.modules, 'agent.execution.managed', SimpleNamespace(ManagedEnvironments=FakeManaged)
    )
    jobs = PreparationJobs(root)
    yield SimpleNamespace(
        root=root,
        repos=repos,
        jobs=jobs,
        binding=binding,
        entered=entered,
        release=release,
        calls=calls,
        ready=ready,
        behavior=behavior,
    )
    release.set()
    jobs.close()


def settled(jobs, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with jobs.lock:
            if not jobs.workers:
                return
        time.sleep(0.01)
    pytest.fail('Environment jobs did not finish')


def test_status_is_read_only_and_ready_cache_does_not_start_job(job_fixture):
    fixture = job_fixture
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'idle'
    assert fixture.calls == []
    fixture.ready[str(fixture.repos[0])] = fixture.binding
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'ready'
    assert fixture.calls == []


def test_start_deduplicates_same_project_and_bounds_total_jobs(job_fixture):
    fixture = job_fixture
    first = fixture.jobs.start(fixture.repos[0])
    assert fixture.entered.wait(2)
    again = fixture.jobs.start(fixture.repos[0])
    assert again['id'] == first['id']
    with pytest.raises(EnvironmentRequired, match='一个'):
        fixture.jobs.start(fixture.repos[1])
    assert len(fixture.jobs.workers) == 1
    fixture.release.set()
    settled(fixture.jobs)
    assert fixture.calls == [str(fixture.repos[0])]
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'ready'


def test_failure_is_saved_redacted_and_explicit_retry_can_succeed(job_fixture):
    fixture = job_fixture
    fixture.behavior.update(block=False, error=RuntimeError('PRIVATE_AUTH_TOKEN'))
    fixture.jobs.start(fixture.repos[0])
    settled(fixture.jobs)
    failed = fixture.jobs.status(fixture.repos[0])
    assert failed['status'] == 'failed'
    assert 'PRIVATE_AUTH_TOKEN' not in json.dumps(failed)
    fixture.behavior['error'] = None
    assert fixture.jobs.start(fixture.repos[0])['id'] == failed['id']
    with pytest.raises(EnvironmentRequired):
        fixture.jobs.wait(fixture.repos[0])
    assert len(fixture.calls) == 1
    second = fixture.jobs.retry(fixture.repos[0])
    assert second['id'] != failed['id']
    settled(fixture.jobs)
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'ready'
    assert len(fixture.calls) == 2


@pytest.mark.parametrize('status', ['cancelled', 'interrupted', 'preparing'])
def test_stopped_or_unconfirmed_preparation_is_not_automatically_recreated(job_fixture, status):
    fixture = job_fixture
    saved = fixture.jobs.catalog.save_environment_job({
        'id': 'old', 'repo': str(fixture.repos[0]), 'owner': 'prepare:old', 'mode': 'auto',
        'status': status, 'stage': status, 'message': 'saved'})
    assert fixture.jobs.start(fixture.repos[0])['id'] == saved['id']
    with pytest.raises(EnvironmentRequired):
        fixture.jobs.wait(fixture.repos[0])
    assert not fixture.calls
    if status == 'preparing':
        assert fixture.jobs.status(fixture.repos[0])['status'] == 'interrupted'


def test_wait_cancel_detaches_without_canceling_other_conversations(job_fixture):
    fixture = job_fixture
    fixture.jobs.start(fixture.repos[0])
    with pytest.raises(PreparationCancelled, match='等待'):
        fixture.jobs.wait(fixture.repos[0], cancelled=lambda: True)
    assert fixture.entered.wait(2)
    job = next(iter(fixture.jobs.workers.values()))
    assert not job['cancel'].is_set()
    fixture.jobs.cancel(fixture.repos[0])
    settled(fixture.jobs)
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'cancelled'


def test_only_task_waiter_cancel_stops_unshared_preparation(job_fixture):
    fixture = job_fixture
    with pytest.raises(PreparationCancelled, match='等待'):
        fixture.jobs.wait(fixture.repos[0], cancelled=lambda: True)
    settled(fixture.jobs)
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'cancelled'


def test_close_cancels_all_workers_and_joins_threads(job_fixture):
    fixture = job_fixture
    fixture.jobs.start(fixture.repos[0])
    assert fixture.entered.wait(2)
    fixture.jobs.close()
    assert not fixture.jobs.workers
    assert all(not thread.is_alive() for thread in fixture.jobs.pool._threads)
    with pytest.raises(EnvironmentRequired, match='关闭'):
        fixture.jobs.start(fixture.repos[2])


def test_orphaned_preparation_never_replays_on_status(job_fixture):
    fixture = job_fixture
    path = fixture.root / 'environment-jobs' / (
        hashlib.sha256(str(fixture.repos[0]).encode()).hexdigest() + '.json')
    atomic_write(path, b'{"status":"preparing","id":"old","stage":"installing"}')
    original = path.read_bytes()
    result = fixture.jobs.status(fixture.repos[0])
    assert result['status'] == 'interrupted' and result['stage'] == 'interrupted'
    assert path.read_bytes() == original
    assert fixture.calls == []


def test_manual_binding_wins_and_automatic_reset_does_not_change_frozen_copy(job_fixture):
    fixture = job_fixture
    repo = str(fixture.repos[0])
    atomic_write(
        ProjectEnvironments(fixture.root).path, json.dumps({repo: fixture.binding}).encode()
    )
    frozen = dict(fixture.binding)
    assert fixture.jobs.wait(repo) == frozen
    assert not fixture.calls
    ProjectEnvironments(fixture.root).automatic(repo)
    assert fixture.jobs.status(repo)['mode'] == 'auto'
    assert frozen == fixture.binding


def test_latest_manual_binding_overrides_old_failed_preparation(job_fixture):
    fixture = job_fixture
    repo = str(fixture.repos[0])
    fixture.jobs.catalog.save_environment_job({
        'id': 'old', 'repo': repo, 'status': 'failed', 'error': 'old dependency failure'})
    atomic_write(
        ProjectEnvironments(fixture.root).path, json.dumps({repo: fixture.binding}).encode()
    )
    assert fixture.jobs.wait(repo) == fixture.binding
    assert not fixture.calls


def test_task_pause_cancels_single_owned_preparation_even_if_started_from_directory(job_fixture):
    from concurrent.futures import ThreadPoolExecutor
    from agent.storage.resources import container_owner, cancel_owned_containers

    fixture = job_fixture
    first = fixture.jobs.start(fixture.repos[0], background=True)
    def wait():
        with container_owner('task:only', fixture.root):
            return fixture.jobs.wait(fixture.repos[0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(wait)
        deadline = time.monotonic() + 2
        while not fixture.jobs.catalog.environment_waiters(first['id']):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        results = cancel_owned_containers('task:only', fixture.root)
        assert results[0]['preparation_job'] == first['id']
        assert not results[0]['detached']
        with pytest.raises(PreparationCancelled):
            future.result(timeout=2)
    settled(fixture.jobs)
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'cancelled'


def test_shared_preparation_detaches_one_task_and_other_task_finishes(job_fixture):
    from concurrent.futures import ThreadPoolExecutor
    from agent.storage.resources import container_owner, cancel_owned_containers

    fixture = job_fixture
    def wait(owner):
        with container_owner(owner, fixture.root):
            return fixture.jobs.wait(fixture.repos[0])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(wait, 'task:one')
        assert fixture.entered.wait(2)
        second = pool.submit(wait, 'task:two')
        value = fixture.jobs.catalog.environment_job(str(fixture.repos[0]))
        deadline = time.monotonic() + 2
        while len(fixture.jobs.catalog.environment_waiters(value['id'])) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert cancel_owned_containers('task:one', fixture.root)[0]['detached']
        with pytest.raises(PreparationCancelled):
            first.result(timeout=2)
        assert not next(iter(fixture.jobs.workers.values()))['cancel'].is_set()
        fixture.release.set()
        assert second.result(timeout=2) == fixture.binding
    assert len(fixture.calls) == 1
    assert not fixture.jobs.catalog.environment_waiters(value['id'])


def test_only_one_environment_manager_can_prepare_same_data_root(job_fixture):
    fixture = job_fixture
    second = PreparationJobs(fixture.root)
    try:
        with pytest.raises(EnvironmentRequired, match='环境管理者'):
            second.start(fixture.repos[0])
        assert second.status(fixture.repos[0])['status'] == 'idle'
    finally:
        second.close()


def test_restart_control_uses_persisted_waiter_and_keeps_unconfirmed_stop(job_fixture):
    fixture = job_fixture
    value = fixture.jobs.catalog.save_environment_job({
        'id': 'a' * 32, 'repo': str(fixture.repos[0]), 'owner': 'prepare:' + 'a' * 32,
        'status': 'interrupted', 'stage': 'interrupted', 'message': 'saved', 'background': False})
    fixture.jobs.catalog.attach_environment_waiter(value['id'], 'task:saved')
    with patch('agent.execution.preparation.cancel_owned_containers',
               return_value=[{'container': 'original', 'stopped': False}]) as stop:
        result = fixture.jobs.cancel_owner('task:saved')
        assert not result[0]['stopped']
        assert fixture.jobs.catalog.environment_waiters(value['id'])
        stop.assert_called_once_with(value['owner'], fixture.root)
    assert not fixture.calls


def test_waiter_remains_bound_to_original_job_when_newer_project_job_exists(job_fixture):
    fixture = job_fixture
    old = fixture.jobs.catalog.save_environment_job({
        'id': 'old', 'repo': str(fixture.repos[0]), 'owner': 'prepare:old', 'mode': 'auto',
        'status': 'ready', 'stage': 'ready', 'message': 'old', 'environment': {'image': 'old'}})
    fixture.jobs.catalog.save_environment_job({
        **old, 'id': 'new', 'owner': 'prepare:new', 'environment': {'image': 'new'}})
    with patch.object(fixture.jobs, 'start', return_value=old):
        assert fixture.jobs.wait(fixture.repos[0]) == {'image': 'old'}
    assert not fixture.calls


@pytest.mark.parametrize('recorded_id', ['a' * 64, None])
def test_explicit_retry_reconciles_removed_preparation_without_replaying_old_job(
    job_fixture, monkeypatch, recorded_id
):
    from agent.storage.resources import ownership_labels

    fixture, owner = job_fixture, 'prepare:failed-create'
    saved = fixture.jobs.catalog.save_environment_job({
        'id': 'failed-create', 'repo': str(fixture.repos[0]), 'owner': owner,
        'status': 'failed', 'stage': 'failed', 'message': '保存的下载失败', 'error': 'fixture'})
    fixture.jobs.catalog.register('container:old', 'container', 'monoplanner-old', owner=owner,
        metadata={'container_id': recorded_id, 'image': 'sha256:' + 'b' * 64,
                  'labels': ownership_labels(owner, fixture.root)})
    query_calls = []
    def query(command, **kwargs):
        query_calls.append(command)
        return SimpleNamespace(returncode=1, stdout='', stderr='No such container: old')
    monkeypatch.setattr('subprocess.run', query)
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'failed'
    assert fixture.jobs.start(fixture.repos[0])['id'] == saved['id']
    assert query_calls == [] and fixture.calls == []
    fixture.behavior['block'] = False
    retried = fixture.jobs.retry(fixture.repos[0])
    settled(fixture.jobs)
    assert retried['id'] != saved['id']
    assert fixture.calls == [str(fixture.repos[0])]
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'ready'
    assert len(query_calls) == 1
    assert fixture.jobs.catalog.environment_job(job_id=saved['id']) == saved


def test_retry_does_not_start_when_original_container_stop_is_unknown(job_fixture):
    fixture = job_fixture
    saved = fixture.jobs.catalog.save_environment_job({
        'id': 'old', 'repo': str(fixture.repos[0]), 'owner': 'prepare:old',
        'status': 'failed', 'stage': 'failed', 'message': 'saved'})
    with patch('agent.execution.preparation.cancel_owned_containers',
               return_value=[{'container': 'original', 'stopped': False}]):
        with pytest.raises(EnvironmentRequired, match='停止未确认'):
            fixture.jobs.retry(fixture.repos[0])
    assert fixture.jobs.catalog.environment_job(fixture.repos[0]) == saved
    assert not fixture.calls


def test_stopped_original_preparation_container_does_not_block_explicit_retry(job_fixture, monkeypatch):
    from agent.storage.resources import ownership_labels

    fixture, owner, identity = job_fixture, 'prepare:old', 'a' * 64
    labels = ownership_labels(owner, fixture.root)
    fixture.jobs.catalog.save_environment_job({
        'id': 'old', 'repo': str(fixture.repos[0]), 'owner': owner,
        'status': 'failed', 'stage': 'failed', 'message': 'saved'})
    fixture.jobs.catalog.register('container:old', 'container', 'monoplanner-old', owner=owner,
        metadata={'container_id': identity, 'image': 'sha256:' + 'b' * 64, 'labels': labels})
    actual = {'Id': identity, 'Name': '/monoplanner-old', 'Image': 'sha256:' + 'b' * 64,
              'Config': {'Labels': labels}, 'State': {'Running': False}}
    with patch('subprocess.run', return_value=SimpleNamespace(
        returncode=0, stdout=json.dumps([actual]), stderr='')) as query:
        fixture.behavior['block'] = False
        assert fixture.jobs.retry(fixture.repos[0])['id'] != 'old'
        settled(fixture.jobs)
        assert query.call_count == 1
        assert query.call_args.args[0][-3:] == ['container', 'inspect', identity]
        assert fixture.jobs.catalog.usage()['resources'][0]['metadata']['resource_state'] == 'stopped'
        assert fixture.calls == [str(fixture.repos[0])]
        # 后续关闭也核对保留的旧资源，始终不重跑其安装操作。
        assert fixture.jobs.close()['stopped']
        assert query.call_count == 2


def test_failed_preparation_can_be_explicitly_stopped_without_losing_failure(job_fixture):
    fixture = job_fixture
    saved = fixture.jobs.catalog.save_environment_job({
        'id': 'old', 'repo': str(fixture.repos[0]), 'owner': 'prepare:old',
        'status': 'failed', 'stage': 'failed', 'message': 'saved', 'error': 'download failed'})
    with patch('agent.execution.preparation.cancel_owned_containers', return_value=[]) as stop:
        assert fixture.jobs.cancel(fixture.repos[0]) == saved
        stop.assert_called_once_with(saved['owner'], fixture.root)


def test_recovered_preparation_cancel_updates_terminal_status(job_fixture):
    fixture = job_fixture
    fixture.jobs.catalog.save_environment_job({
        'id': 'old', 'repo': str(fixture.repos[0]), 'owner': 'prepare:old',
        'status': 'interrupted', 'stage': 'interrupted', 'message': 'saved'})
    with patch('agent.execution.preparation.cancel_owned_containers', return_value=[]):
        assert fixture.jobs.cancel(fixture.repos[0])['status'] == 'cancelled'
    assert not fixture.calls


def test_close_jobs_fences_late_preparation_and_preserves_closed_manager(tmp_path):
    from agent.execution.preparation import jobs, close_jobs, cancel_owner

    root, repo = tmp_path / 'state', tmp_path / 'repo'
    repo.mkdir()
    service = jobs(root)
    assert close_jobs(root)['stopped']
    assert jobs(root) is service
    with pytest.raises(EnvironmentRequired, match='关闭'):
        jobs(root).start(repo)
    assert cancel_owner('task:none', root) == []


def test_close_empty_manager_does_not_allow_late_creation(tmp_path):
    from agent.execution.preparation import jobs, close_jobs, cancel_owner

    root = tmp_path / 'state'
    assert close_jobs(root)['stopped']
    with pytest.raises(EnvironmentRequired, match='关闭'):
        jobs(root)
    assert cancel_owner('task:none', root) == []
    assert not root.exists()


@pytest.mark.parametrize('state', ['missing', 'stopped', 'running'])
def test_close_after_restart_stops_persisted_preparation_without_creating_manager(
    tmp_path, monkeypatch, state
):
    from agent.execution.preparation import close_jobs, cancel_owner
    from agent.storage.resources import ResourceCatalog, ownership_labels

    root, identity, owner = tmp_path / 'state', 'a' * 64, 'prepare:old'
    catalog = ResourceCatalog(root)
    labels = ownership_labels(owner, root)
    catalog.save_environment_job({'id': 'old', 'repo': str(tmp_path / 'repo'), 'owner': owner,
        'status': 'interrupted', 'stage': 'interrupted', 'message': 'saved'})
    catalog.attach_environment_waiter('old', 'task:one')
    catalog.attach_environment_waiter('old', 'task:two')
    metadata = {'container_id': identity, 'image': 'sha256:' + 'b' * 64, 'labels': labels}
    catalog.register('container:old', 'container', 'monoplanner-old',
                     owner=owner, metadata=metadata)
    # 不相关任务和未声明归属的资源不能因为前缀相似被纳入退出清理。
    catalog.register('container:other', 'container', 'other', owner='task:other')
    catalog.register('container:user', 'container', 'user', owner='prepare:user', owned=False)
    calls = []
    def query(command, **kwargs):
        arguments = command[1:]
        calls.append(arguments)
        assert arguments[-1] == identity
        if state == 'missing':
            return SimpleNamespace(returncode=1, stdout='', stderr='No such container: original')
        if arguments[0] == 'kill':
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        running = state == 'running' and len(calls) == 1
        return SimpleNamespace(returncode=0, stdout=json.dumps([{
            'Id': identity, 'Name': '/monoplanner-old', 'Image': metadata['image'],
            'Config': {'Labels': labels}, 'State': {'Running': running}}]), stderr='')
    monkeypatch.setattr('subprocess.run', query)
    with patch('agent.execution.preparation.PreparationJobs',
               side_effect=AssertionError('must not create an environment manager')):
        result = close_jobs(root)
        assert result['stopped']
        assert result['resources'][0]['resource_state'] == (
            'missing' if state == 'missing' else 'stopped')
        assert cancel_owner('task:one', root) == [{'preparation_job': 'old', 'stopped': True}]
    assert catalog.environment_job(job_id='old')['status'] == 'cancelled'
    assert [row['owner'] for row in catalog.environment_waiters('old')] == ['task:two']
    assert all(command[-1] == identity for command in calls)
    assert len([command for command in calls if command[0] == 'kill']) == (state == 'running')


def test_close_after_restart_preserves_unknown_preparation_and_waiter(tmp_path):
    from agent.execution.preparation import close_jobs, cancel_owner
    from agent.storage.resources import ResourceCatalog, ownership_labels

    root, owner = tmp_path / 'state', 'prepare:old'
    catalog = ResourceCatalog(root)
    catalog.save_environment_job({'id': 'old', 'repo': str(tmp_path / 'repo'), 'owner': owner,
        'status': 'interrupted', 'stage': 'interrupted', 'message': 'saved'})
    catalog.attach_environment_waiter('old', 'task:one')
    catalog.register('container:old', 'container', 'monoplanner-old', owner=owner,
        metadata={'container_id': 'a' * 64, 'labels': ownership_labels(owner, root)})
    with patch('subprocess.run', return_value=SimpleNamespace(
        returncode=1, stdout='', stderr='Cannot connect to Docker daemon')):
        result = close_jobs(root)
        assert not result['stopped'] and not result['resources'][0]['stopped']
        assert not cancel_owner('task:one', root)[0]['stopped']
    assert catalog.environment_waiters('old')
    assert catalog.environment_job(job_id='old')['status'] == 'interrupted'


def test_close_without_manager_does_not_claim_success_when_ledger_is_unreadable(tmp_path):
    from agent.execution.preparation import close_jobs

    root = tmp_path / 'state'
    root.mkdir()
    (root / 'resources.sqlite3').write_bytes(b'not a sqlite database')
    with patch('agent.execution.preparation.PreparationJobs',
               side_effect=AssertionError('must not create an environment manager')):
        result = close_jobs(root)
    assert not result['stopped']
    assert result['errors'][0]['scope'] == 'persistent_preparation_ownership'


def test_explicit_verification_updates_ready_job_without_starting_install(job_fixture):
    fixture = job_fixture
    fixture.jobs.catalog.save_environment_job({
        'id': 'saved', 'repo': str(fixture.repos[0]), 'owner': 'prepare:saved',
        'status': 'ready', 'stage': 'ready', 'message': 'old ready',
        'environment': fixture.binding})
    manager = sys.modules['agent.execution.managed'].ManagedEnvironments
    with patch.object(manager, 'verify', create=True, return_value={
        'state': 'missing', 'message': '镜像已删除'}) as verify:
        assert fixture.jobs.status(fixture.repos[0])['status'] == 'ready'
        verify.assert_not_called()
        missing = fixture.jobs.status(fixture.repos[0], verify=True)
        assert missing['status'] == 'failed' and missing['verification']['state'] == 'missing'
        assert fixture.jobs.status(fixture.repos[0])['status'] == 'failed'
        assert fixture.jobs.start(fixture.repos[0])['id'] == 'saved'
        assert not fixture.calls
        verify.return_value = {'state': 'ready', 'message': '已核验'}
        restored = fixture.jobs.status(fixture.repos[0], verify=True)
        assert restored['status'] == 'ready' and 'error' not in restored
        assert not fixture.calls


def test_unknown_environment_verification_is_not_reported_as_ready(job_fixture):
    fixture = job_fixture
    fixture.jobs.catalog.save_environment_job({
        'id': 'saved', 'repo': str(fixture.repos[0]), 'owner': 'prepare:saved',
        'status': 'ready', 'stage': 'ready', 'message': 'old ready',
        'environment': fixture.binding})
    manager = sys.modules['agent.execution.managed'].ManagedEnvironments
    with patch.object(manager, 'verify', create=True, return_value={
        'state': 'unknown', 'message': 'Docker 暂时无法连接'}):
        value = fixture.jobs.status(fixture.repos[0], verify=True)
    assert value['status'] == 'failed' and value['verification']['state'] == 'unknown'
    assert value['environment'] == fixture.binding
    assert not fixture.calls


def test_shared_preparation_rejects_a_different_frozen_scope(job_fixture):
    from agent.source.scope import ProjectScope, ProjectScopeError

    fixture = job_fixture
    repo = fixture.repos[0]
    scope = ProjectScope.discover(repo)
    first = fixture.jobs.start(repo, project_scope=scope)
    assert fixture.entered.wait(2)
    assert fixture.jobs.catalog.environment_job(job_id=first['id'])['project_scope'] == scope.to_record()
    assert fixture.jobs.start(repo, project_scope=scope)['id'] == first['id']
    other = repo / 'other'
    other.mkdir()
    (other / '.git').write_text('gitdir: elsewhere')
    different = ProjectScope.discover(repo)
    with pytest.raises(ProjectScopeError, match='不同项目范围'):
        fixture.jobs.start(repo, project_scope=different)
    assert fixture.calls == [str(repo)]
    assert fixture.jobs.catalog.environment_job(job_id=first['id'])['project_scope'] == scope.to_record()


def test_scope_diagnostic_survives_environment_job_failure(job_fixture):
    from agent.source.scope import ProjectScopeError

    fixture = job_fixture
    fixture.behavior.update(block=False, error=ProjectScopeError('引用指向已排除的独立项目：other'))
    fixture.jobs.start(fixture.repos[0])
    settled(fixture.jobs)
    failed = fixture.jobs.status(fixture.repos[0])
    assert failed['status'] == 'failed'
    assert failed['error'] == '引用指向已排除的独立项目：other'
    with pytest.raises(EnvironmentRequired, match='已排除'):
        fixture.jobs.wait(fixture.repos[0])
    assert len(fixture.calls) == 1
