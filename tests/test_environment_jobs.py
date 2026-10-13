"""Preparation jobs share work without sharing model or source execution."""
import json
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
    root = tmp_path/'state'
    repos = [tmp_path/f'project-{i}' for i in range(3)]
    for repo in repos:
        repo.mkdir()
    entered, release, calls, ready = threading.Event(), threading.Event(), [], {}
    binding = {'backend': 'docker', 'image': 'sha256:' + 'a' * 64,
               'runtime': {'python': '/usr/local/bin/python', 'conda_env': ''}}
    behavior = {'error': None, 'block': True}

    class FakeManaged:
        def __init__(self, selected_root):
            assert selected_root == root

        def status(self, repo):
            return ({'state': 'ready', 'stage': 'ready', 'environment': ready[repo]}
                    if repo in ready else {'state': 'needs_preparation', 'stage': 'checking'})

        def prepare(self, repo, *, progress, cancelled):
            calls.append(repo)
            entered.set()
            progress({'stage': 'installing', 'message': '正在安装项目依赖'})
            while behavior['block'] and not release.wait(.01):
                if cancelled():
                    raise PreparationCancelled('cancelled')
            if behavior['error']:
                raise behavior['error']
            ready[repo] = dict(binding)
            return binding

    monkeypatch.setitem(sys.modules, 'agent.execution.managed', SimpleNamespace(ManagedEnvironments=FakeManaged))
    jobs = PreparationJobs(root)
    yield SimpleNamespace(root=root, repos=repos, jobs=jobs, binding=binding, entered=entered,
        release=release, calls=calls, ready=ready, behavior=behavior)
    release.set()
    jobs.close()


def settled(jobs, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with jobs.lock:
            if not jobs.jobs:
                return
        time.sleep(.01)
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
    fixture.jobs.start(fixture.repos[1])
    with pytest.raises(EnvironmentRequired, match='两个'):
        fixture.jobs.start(fixture.repos[2])
    assert len(fixture.jobs.jobs) == 2
    fixture.release.set()
    settled(fixture.jobs)
    assert sorted(fixture.calls) == sorted(map(str, fixture.repos[:2]))
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
    job = fixture.jobs.jobs[str(fixture.repos[0])]
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
    fixture.jobs.start(fixture.repos[1])
    assert fixture.entered.wait(2)
    fixture.jobs.close()
    assert not fixture.jobs.jobs
    assert all(not thread.is_alive() for thread in fixture.jobs.pool._threads)
    with pytest.raises(EnvironmentRequired, match='关闭'):
        fixture.jobs.start(fixture.repos[2])


def test_orphaned_preparation_never_replays_on_status(job_fixture):
    fixture = job_fixture
    fixture.jobs._save(str(fixture.repos[0]), {'status': 'preparing', 'id': 'old', 'stage': 'installing'})
    result = fixture.jobs.status(fixture.repos[0])
    assert result['status'] == 'failed' and result['stage'] == 'interrupted'
    assert fixture.calls == []


def test_manual_binding_wins_and_automatic_reset_does_not_change_frozen_copy(job_fixture):
    fixture = job_fixture
    repo = str(fixture.repos[0])
    atomic_write(ProjectEnvironments(fixture.root).path, json.dumps({repo: fixture.binding}).encode())
    frozen = dict(fixture.binding)
    assert fixture.jobs.wait(repo) == frozen
    assert not fixture.calls
    ProjectEnvironments(fixture.root).automatic(repo)
    assert fixture.jobs.status(repo)['mode'] == 'auto'
    assert frozen == fixture.binding


def test_latest_manual_binding_overrides_old_failed_preparation(job_fixture):
    fixture = job_fixture
    repo = str(fixture.repos[0])
    fixture.jobs._save(repo, {'status': 'failed', 'error': 'old dependency failure'})
    atomic_write(ProjectEnvironments(fixture.root).path, json.dumps({repo: fixture.binding}).encode())
    assert fixture.jobs.wait(repo) == fixture.binding
    assert not fixture.calls
