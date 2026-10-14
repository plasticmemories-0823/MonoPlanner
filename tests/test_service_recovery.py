"""恢复生命周期记录不能重放请求、跳过活动执行者或改动任务原件。"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from functools import wraps
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from agent.task import control as interrupts
from tests.legacy.lifecycle import reconcile
from tests.legacy.conversations import ConversationStore
from agent.storage.history import LocalHistory


def offline_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


@pytest.fixture
def recovery(tmp_path):
    thread, task = str(uuid.uuid4()), uuid.uuid4().hex
    store = ConversationStore(tmp_path)
    session = {'task_id': task, 'status': 'paused'}
    store.save_session(thread, session)
    history = LocalHistory(tmp_path / 'tasks' / task)
    with history.lease():
        history.record({'history': [], 'status': 'paused'}, 'task_returned', {})
    history.close()
    run = {
        'run_id': str(uuid.uuid4()),
        'thread_id': thread,
        'status': 'running',
        'created_at': datetime.now(timezone.utc) - timedelta(seconds=10),
    }
    value = SimpleNamespace(
        root=tmp_path,
        history=history,
        thread=thread,
        task=task,
        run=run,
        session=session,
        workers=0,
        pending=0,
        cancelled=[],
        updated=[],
        interrupts={},
        thread_status='idle',
        missing_thread=False,
    )

    @asynccontextmanager
    async def connect():
        yield object()

    class Runs:
        @staticmethod
        async def stats(connection):
            return {'n_pending': value.pending, 'n_running': int(run['status'] == 'running')}

        @staticmethod
        async def search(connection, identity, **kwargs):
            async def records():
                if kwargs.get('status') is None or run['status'] == kwargs['status']:
                    yield run

            return records()

        @staticmethod
        async def cancel(connection, run_ids, **kwargs):
            value.cancelled.append((run_ids, kwargs))
            run['status'] = 'interrupted'

    class Threads:
        @staticmethod
        async def get(connection, identity):
            if value.missing_thread:
                from starlette.exceptions import HTTPException

                raise HTTPException(status_code=404)

            async def records():
                yield {
                    'status': value.thread_status,
                    'interrupts': value.interrupts,
                    'values': {'original': 'preserved'},
                }

            return records()

        @staticmethod
        async def set_status(connection, identity, checkpoint, exception):
            value.updated.append((checkpoint, exception))
            value.thread_status = 'idle'

    value.api = connect, Runs, Threads, lambda: {'workers': {'active': value.workers}}
    return value


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*')
        if p.is_file() and 'service-recovery' not in p.parts
    }


@offline_async
async def test_paused_orphan_is_interrupted_once_without_changing_originals(recovery):
    before = hashes(recovery.root)
    assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 1}
    assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 0}
    assert len(recovery.cancelled) == 1
    assert recovery.cancelled[0][1]['action'] == 'interrupt'
    assert recovery.updated == [(None, None)]
    assert hashes(recovery.root) == before
    saved = list((recovery.root / 'service-recovery').glob('*.json'))
    assert json.loads(saved[0].read_text())['run']['status'] == 'running'


@offline_async
@pytest.mark.parametrize(
    'reason', ['worker', 'pending', 'owner', 'lease', 'new_run', 'tamper', 'unknown', 'active']
)
async def test_uncertain_or_active_runs_are_not_reconciled(recovery, reason):
    if reason == 'worker':
        recovery.workers = 1
    elif reason == 'pending':
        recovery.pending = 1
    elif reason == 'new_run':
        recovery.run['created_at'] = datetime.now(timezone.utc) + timedelta(seconds=1)
    elif reason == 'active':
        recovery.session['status'] = 'executing'
        ConversationStore(recovery.root).save_session(recovery.thread, recovery.session)
    elif reason == 'unknown':
        with recovery.history.lease():
            recovery.history.record(
                {'history': [], 'status': 'paused', 'request_pending': {'call': 1}},
                'task_returned',
                {},
            )
        recovery.history.close()
    elif reason == 'tamper':
        file = next(
            (recovery.root / 'tasks' / recovery.task / 'history/checkpoints').glob('*.json')
        )
        file.write_text('{}')
    if reason == 'owner':
        with interrupts.active(recovery.root, recovery.thread):
            assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 0}
    elif reason == 'lease':
        with recovery.history.lease():
            assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 0}
        recovery.history.close()
    else:
        assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 0}
    assert not recovery.cancelled
    assert not recovery.updated


@offline_async
async def test_framework_interrupts_are_not_replaced_by_an_empty_checkpoint(recovery):
    recovery.interrupts = {'original': ['awaiting-user']}
    assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 1}
    assert recovery.updated == []


@offline_async
async def test_interrupted_run_does_not_leave_a_permanently_busy_thread(recovery):
    recovery.run['status'] = 'interrupted'
    recovery.thread_status = 'busy'
    assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 1}
    assert not recovery.cancelled
    assert recovery.updated == [(None, None)]
    assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 0}


@offline_async
async def test_deleted_framework_thread_does_not_block_other_service_startup(recovery):
    recovery.run['status'] = 'interrupted'
    recovery.missing_thread = True
    assert await reconcile(recovery.root, api=recovery.api) == {'reconciled': 0}
    assert not recovery.cancelled
    assert not recovery.updated


def test_service_recovery_requires_authentication_and_accepts_no_execution_fields():
    from tests.http_client import TestClient
    from agent.server.app import app

    with TestClient(app) as client:
        assert client.post('/service/reconcile', json={'command': 'do-not-run'}).status_code == 400
        assert client.get('/service/reconcile').status_code == 405
        client.headers.pop('X-MonoPlanner-Service')
        assert client.post('/service/reconcile', json={}).status_code == 401
