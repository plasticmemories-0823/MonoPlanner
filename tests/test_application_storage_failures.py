"""磁盘故障后的派发闭锁、直接错误提示和紧急控制；不用真实模型或容器。"""

import json
import asyncio
import sqlite3
import threading
import time
import uuid
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient
from starlette.requests import Request

from agent.application.api import build_routes
from agent.application.scheduler import Scheduler
from agent.application.store import ApplicationStore
from agent.application.store import Conflict


@pytest.fixture
def application(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(repo))['thread_id']
    message = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '请开工'}
    store.submit(conversation, message)
    work = store.create_work(conversation, message['id'])
    scheduler = Scheduler(store, work_factory=lambda *_: pytest.fail('Unexpected work dispatch'))
    yield store, conversation, message, work, scheduler
    scheduler.close()


def full():
    return sqlite3.OperationalError('database or disk is full')


def test_work_registration_failure_releases_owner_and_never_retries_after_space_returns(application):
    store, _, _, work, scheduler = application
    with patch.object(store, 'work_state', side_effect=full()):
        with pytest.raises(OSError):
            scheduler._schedule_work(work)
    assert not scheduler.works
    assert scheduler.status()['storage_error']['dispatch_suspended']
    scheduler.tick()
    assert store.work(work['task_id'])['status'] == 'queued'
    assert not scheduler.works


def test_chat_registration_failure_releases_owner_and_stops_dispatch(application):
    store, _, _, _, scheduler = application
    with patch.object(store, 'command_state', side_effect=full()):
        with pytest.raises(OSError):
            scheduler.tick()
    assert not scheduler.chats
    assert scheduler.status()['storage_error']
    scheduler.tick()
    assert not scheduler.chats


def test_failed_result_projection_is_latched_and_stops_next_kernel_boundary(application):
    store, _, _, work, scheduler = application
    store.work_state(work['task_id'], 'running')
    scheduler.works.add(work['task_id'])
    scheduler.work_factory = lambda *_: (_ for _ in ()).throw(OSError('copy disk full'))
    with patch.object(store, 'work_state', side_effect=full()):
        scheduler._work(work)
    assert not scheduler.works
    assert scheduler.status()['storage_error']
    assert store.work(work['task_id'])['status'] == 'running'
    stop = scheduler.controls(work['task_id'])
    assert len(stop) == 1 and stop[0]['kind'] == 'pause'
    assert stop == scheduler.controls(work['task_id'])


@pytest.mark.parametrize('failure_status', ['preparation_failed', 'runtime_error'])
def test_known_failure_is_one_immediate_public_error_without_summary(application, failure_status):
    store, conversation, message, work, scheduler = application
    store.command_state(conversation, message['id'], 'applied')
    scheduler.work_factory = lambda *_: {
        'status': failure_status, 'reason': '源码副本准备失败：磁盘空间不足。',
    }
    scheduler._work(work)
    scheduler._work(work)
    snapshot = store.snapshot(conversation)
    failures = [row for row in snapshot['messages'] if row['source'] == 'system']
    assert len(failures) == 1 and '磁盘空间不足' in failures[0]['content']
    assert snapshot['work']['status'] == failure_status
    assert store.reserve_progress(now=time.time() + 60) is None
    assert not scheduler.status()['storage_error']


def test_pause_still_requests_owned_termination_when_command_write_fails(application):
    store, conversation, _, work, scheduler = application
    store.work_state(work['task_id'], 'running')
    called = threading.Event()
    scheduler.pause_callback = lambda task, command: (called.set() or {'termination': []})
    pause = {'id': str(uuid.uuid4()), 'kind': 'pause'}
    with patch.object(store, 'submit', side_effect=full()):
        result = scheduler.submit(conversation, pause)
    assert called.wait(2)
    assert result['control_requested'] and not result['recorded']
    assert not result['termination_confirmed']
    assert scheduler.status()['storage_error']
    # 空间恢复也不自动开启派发；同一紧急命令不会再排一次控制。
    result = scheduler.submit(conversation, pause)
    assert not result['recorded']
    # 派发仍被闭锁；只有明确继续且写探针与原件核对成功后才能解除。
    receipt = scheduler.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume'})
    assert receipt['received'] and not scheduler.status()['storage_error']


def test_storage_failure_never_grants_emergency_control_over_unknown_or_legacy_conversation(application):
    store, _, _, _, scheduler = application
    with patch.object(store, 'submit', side_effect=full()), patch.object(scheduler, 'pause_callback') as cancel:
        with pytest.raises(KeyError):
            scheduler.submit(str(uuid.uuid4()), {'id': str(uuid.uuid4()), 'kind': 'pause'})
        cancel.assert_not_called()


def test_snapshot_read_failure_returns_storage_health_and_preserves_records(application):
    store, conversation, _, _, scheduler = application
    client = TestClient(Starlette(routes=build_routes(store, scheduler)))
    with patch.object(store, 'snapshot', side_effect=full()):
        result = client.get('/conversations/' + conversation + '/snapshot')
    assert result.status_code == 503
    assert result.json()['storage_error']['code'] == 'storage_unavailable'
    assert store.get(conversation)['thread_id'] == conversation


@pytest.mark.parametrize('readonly,visibility', [(True, 'active'), (False, 'archived'),
                                                (False, 'purged')])
def test_emergency_pause_rejects_readonly_or_hidden_identity_before_any_signal(application, readonly, visibility):
    store, conversation, _, _, scheduler = application
    value = {**store.get(conversation), 'read_only': readonly, 'visibility': visibility}
    with patch.object(store, 'get', return_value=value), patch('agent.task.control.interrupt') as signal:
        with pytest.raises(Conflict):
            scheduler.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'pause'})
        signal.assert_not_called()


def test_reused_nonpause_identity_cannot_interrupt_an_active_owner(application):
    _, conversation, message, _, scheduler = application
    with patch('agent.task.control.interrupt') as signal:
        with pytest.raises(Conflict):
            scheduler.submit(conversation, {'id': message['id'], 'kind': 'pause'})
        signal.assert_not_called()


def test_storage_fault_interrupts_registered_wait_and_keeps_one_stop_boundary(application):
    store, conversation, _, work, scheduler = application
    from agent.task.control import active

    scheduler.active_jobs[work['task_id']] = conversation
    with active(store.root, work['task_id']) as signal:
        scheduler.storage_failed('checkpoint_write', full())
        first = scheduler.controls(work['task_id'])
        assert signal.kind == 'pause'
        scheduler.storage_failed('another_failed_write', full())
        assert scheduler.controls(work['task_id']) == first


def test_unreadable_database_allows_only_the_registered_live_owner_to_pause(application):
    store, conversation, _, work, scheduler = application
    called = threading.Event()
    scheduler.active_jobs[work['task_id']] = conversation
    scheduler.pause_callback = lambda *_: (called.set() or {'termination': []})
    with patch.object(store, 'get', side_effect=full()):
        result = scheduler.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'pause'})
        assert result['control_requested'] and called.wait(2)
        with pytest.raises(OSError):
            scheduler.submit(str(uuid.uuid4()), {'id': str(uuid.uuid4()), 'kind': 'pause'})


def test_sse_reports_health_without_writing_an_event_or_changing_cursor(application):
    store, conversation, _, _, scheduler = application
    routes = build_routes(store, scheduler)
    endpoint = next(route.endpoint for route in routes if route.path.endswith('/events'))
    cursor = store.snapshot(conversation)['event_cursor']

    async def run():
        async def receive():
            await asyncio.sleep(60)
        request = Request({'type': 'http', 'method': 'GET', 'path': '/', 'headers': [],
                           'query_string': ('after=' + str(cursor)).encode(),
                           'path_params': {'conversation': conversation}}, receive)
        response = await endpoint(request)
        scheduler.storage_failed('work_projection', full())
        value = await asyncio.wait_for(anext(response.body_iterator), timeout=2)
        await response.body_iterator.aclose()
        return value

    value = asyncio.run(run())
    assert 'event: health' in value and 'storage_unavailable' in value
    assert '\nid:' not in value
    assert store.snapshot(conversation)['event_cursor'] == cursor


def test_storage_stop_between_tools_cancels_only_unexecuted_calls_and_pairs_every_identity(application):
    store, conversation, message, _, scheduler = application
    from langchain_core.messages import AIMessage

    class Model:
        calls = 0

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            self.calls += 1
            return AIMessage(content='', response_metadata={'finish_reason': 'stop'}, tool_calls=[
                {'id': 'already-ran', 'name': 'read_file', 'args': {}},
                {'id': 'must-not-run', 'name': 'web_search', 'args': {}},
            ])

    model, executed = Model(), []
    scheduler.chatbot.factory = lambda _: model

    def tool(_conversation, call, _operation):
        executed.append(call['id'])
        scheduler.storage_failed('synthetic_record_failure', full())
        return {'status': 'completed', 'text': 'original successful result'}

    scheduler.chatbot.invoke_tool = tool
    scheduler.chats.add(conversation)
    scheduler._chat(conversation, message)
    assert executed == ['already-ran']
    assert model.calls == 1
    tools = [row['data'] for _, row in store.model_history(conversation) if row['type'] == 'tool']
    assert [row['tool_call_id'] for row in tools] == ['already-ran', 'must-not-run']
    assert json.loads(tools[0]['content']) == {'status': 'completed', 'text': 'original successful result'}
    assert json.loads(tools[1]['content'])['actions_executed'] is False
    assert json.loads(tools[1]['content'])['status'] == 'cancelled'
