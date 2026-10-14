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

        def prepare(self, repo, *, progress, cancelled):
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
    second = fixture.jobs.start(fixture.repos[0])
    assert second['id'] != failed['id']
    settled(fixture.jobs)
    assert fixture.jobs.status(fixture.repos[0])['status'] == 'ready'
    assert len(fixture.calls) == 2


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
