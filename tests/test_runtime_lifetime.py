"""失联停止协议离线测试；不拉镜像、不调用模型。"""

import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import pytest

from agent.runtime.liveness import Lease, host_scope
from agent.execution.docker import DockerEnvironment


def test_guard_wrap_preserves_argv_and_only_mounts_a_readonly_private_lease(tmp_path):
    lease = Lease(tmp_path, 'guard-fixture')
    try:
        args = lease.wrap(['create', '--name', 'guard-fixture', '--network', 'none',
                           '--entrypoint', '/bin/echo', 'image', 'spaces ; $ are literal'], 'image')
        assert args[-5:] == ['image', '/monoplanner-lifetime/lifetime.sh', lease.token,
                             '/bin/echo', 'spaces ; $ are literal']
        assert args[args.index('--entrypoint') + 1] == '/bin/sh'
        assert args.count('--init') == 1
        assert 'target=/monoplanner-lifetime,readonly' in args[args.index('--mount') + 1]
    finally:
        lease.revoke()
    assert (lease.directory / 'heartbeat').read_text().strip() == 'revoked'


def test_lost_host_does_not_keep_renewing_a_live_process_lease(tmp_path):
    alive = [True]
    with host_scope(lambda: alive[0]):
        lease = Lease(tmp_path, 'host-stall')
    lease.arm()
    first = (lease.directory / 'heartbeat').read_bytes()
    alive[0] = False
    lease.thread.join(timeout=3)
    assert not lease.thread.is_alive()
    assert (lease.directory / 'heartbeat').read_bytes() != first
    assert (lease.directory / 'heartbeat').read_text().strip() == 'revoked'
    with pytest.raises(InterruptedError):
        lease.arm()


def test_write_failure_stops_renewal_and_interrupts_the_task(tmp_path):
    from agent.task.control import active
    with active(tmp_path, 'owner') as signal:
        lease = Lease(tmp_path, 'write-failure')
        with patch.object(lease, '_write', side_effect=[None, OSError('disk full')]):
            lease.arm()
            lease.thread.join(timeout=3)
        assert lease.stopped.is_set()
        assert signal.kind == 'pause'


def test_guard_binding_cannot_be_replaced_by_new_script_or_another_owner(tmp_path):
    lease = Lease(tmp_path, 'identity')
    raw = json.loads(lease.binding.read_bytes())
    raw['script_sha256'] = '0' * 64
    lease.binding.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='binding changed'):
        Lease(tmp_path, 'identity')


def test_pause_ack_requires_a_stopped_container_not_just_cli_success(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    env = DockerEnvironment(repo, 'offline', 'owned', artifact_root=tmp_path / 'execution')
    env._container_id = 'a' * 64
    def docker(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps([
            {'Id': env._container_id, 'State': {'Running': True}}]), '')
    with patch.object(env, '_docker', side_effect=docker):
        with pytest.raises(RuntimeError, match='仍在运行'):
            env._kill_now()


def test_explicit_rearm_retains_token_but_starts_a_new_renewal(tmp_path):
    lease = Lease(tmp_path, 'retained')
    lease.arm()
    token = lease.token
    lease.revoke()
    lease.arm()
    try:
        assert lease.token == token and not lease.stopped.is_set()
        assert (lease.directory / 'heartbeat').read_text().startswith(token + ' ')
    finally:
        lease.revoke()


def test_native_pause_is_not_projected_as_confirmed_before_worker_cleanup(tmp_path):
    import uuid
    from agent.application.store import ApplicationStore
    from agent.application.scheduler import Scheduler
    repo = tmp_path / 'repo'
    repo.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(repo))['thread_id']
    command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Inspect'}
    store.submit(conversation, command)
    task = store.create_work(conversation, command['id'])['task_id']
    scheduler = Scheduler(store)
    scheduler.works.add(task)
    scheduler._emit(task, 'work_state', {'status': 'paused', 'event': 'task_returned'})
    assert store.work(task)['status'] == 'pausing'
    scheduler._emit(task, 'work_state', {'status': 'paused'})
    assert store.work(task)['status'] == 'paused'
    scheduler.works.clear()


def test_live_but_stalled_dispatcher_revokes_controls_without_reading_database(tmp_path):
    from agent.application.store import ApplicationStore
    from agent.application.scheduler import Scheduler
    scheduler = Scheduler(ApplicationStore(tmp_path))
    scheduler.last_dispatch_tick = time.monotonic() - 9
    with patch.object(scheduler.store, 'commands', side_effect=AssertionError('must not block')):
        assert scheduler.controls('owned')[0]['kind'] == 'pause'
