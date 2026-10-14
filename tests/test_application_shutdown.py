"""桌面退出的停止边界、排队取消及未知结果；全部使用离线假执行者。"""

import asyncio
import json
import os
import threading
import time
import uuid
from unittest.mock import patch

import pytest
from starlette.requests import Request

from agent.application.scheduler import Scheduler, ServiceClosing
from agent.application.store import ApplicationStore, Conflict
from agent.task.control import active


def message(text='请处理项目'):
    return {'id': str(uuid.uuid4()), 'kind': 'message', 'text': text}


@pytest.fixture
def application(tmp_path, monkeypatch):
    from agent.execution import preparation

    monkeypatch.setattr(preparation, 'close_jobs', lambda root: {'stopped': True})
    repo = tmp_path / 'repo'
    repo.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(repo))['thread_id']
    scheduler = Scheduler(store, work_factory=lambda *_: pytest.fail('Unexpected execution'))
    yield store, conversation, scheduler
    scheduler.close()


def wait_complete(shutdown, timeout=3):
    assert shutdown.finished.wait(timeout), shutdown.status()
    return shutdown.status()


def test_shutdown_cancels_pending_without_replaying_saved_user_message(application):
    store, conversation, scheduler = application
    item = message()
    store.submit(conversation, item)
    work = store.create_work(conversation, item['id'])
    scheduler.shutdown.start()
    first = wait_complete(scheduler.shutdown)
    scheduler.shutdown.start()
    assert scheduler.shutdown.status() == first
    assert first['termination_confirmed'] is True
    assert not store.commands()
    assert store.work(work['task_id'])['status'] == 'paused'
    assert store.snapshot(conversation)['messages'][0]['content'] == item['text']
    with pytest.raises(ServiceClosing):
        scheduler.submit(conversation, message())
    with pytest.raises(Conflict):
        store.submit(conversation, message())
    reopened = ApplicationStore(store.root)
    reopened.recover_interrupted()
    assert not reopened.commands()
    assert reopened.queued_work_count() == 0
    assert reopened.work(work['task_id'])['status'] == 'paused'


def test_shutdown_fences_worker_that_has_not_registered_model_yet(application):
    store, conversation, scheduler = application
    item = message()
    store.submit(conversation, item)
    work = store.create_work(conversation, item['id'])
    scheduler.shutdown.start()
    wait_complete(scheduler.shutdown)
    with patch.object(scheduler.chatbot, 'run') as chat:
        scheduler._chat(conversation, item)
        chat.assert_not_called()
    scheduler._work(work)
    assert store.work(work['task_id'])['status'] == 'paused'


def test_shutdown_interrupts_all_model_lanes_and_pauses_owned_work(application):
    store, conversation, scheduler = application
    item = message()
    store.submit(conversation, item)
    work = store.create_work(conversation, item['id'])
    task = work['task_id']
    interrupted = []
    with scheduler.lock:
        scheduler.chats.add(conversation)
        scheduler.titles.add(conversation)
        scheduler.works.add(task)
        scheduler.active_jobs[task] = conversation
        scheduler.reporting = True
        scheduler.report_task = task

    ready, release = threading.Event(), threading.Event()

    def registered():
        from contextlib import ExitStack
        with ExitStack() as stack:
            for scope in ('chat:' + conversation, 'title:' + conversation, task, 'progress:' + task):
                signal = stack.enter_context(active(store.root, scope))
                stack.enter_context(signal.subscribe(lambda scope=scope: interrupted.append(scope)))
            ready.set()
            release.wait(3)

    worker = threading.Thread(target=registered)
    worker.start()
    assert ready.wait(1)
    calls = []

    def pause(owner, command):
        calls.append((owner, command))
        release.set()
        with scheduler.lock:
            scheduler.chats.clear()
            scheduler.titles.clear()
            scheduler.works.clear()
            scheduler.active_jobs.clear()
            scheduler.reporting = False
        return {'termination': [{'container': 'owned', 'stopped': True}]}

    scheduler.pause_callback = pause
    try:
        scheduler.shutdown.start()
        assert wait_complete(scheduler.shutdown)['termination_confirmed'] is True
        assert len(calls) == 1 and calls[0][0] == task
        assert set(interrupted) == {'chat:' + conversation, 'title:' + conversation,
                                    task, 'progress:' + task}
    finally:
        release.set()
        worker.join(2)


def test_shutdown_storage_failure_still_stops_work_and_reports_unknown(application):
    store, conversation, scheduler = application
    task = 'a' * 32
    scheduler.active_jobs[task] = conversation
    calls = []
    scheduler.pause_callback = lambda *args: calls.append(args) or {'termination': []}
    with patch.object(store, 'stop_pending', side_effect=OSError('disk full')):
        scheduler.shutdown.start()
        status = wait_complete(scheduler.shutdown)
    assert len(calls) == 1
    assert status['complete'] is True
    assert status['termination_confirmed'] is False
    assert status['storage_error']['dispatch_suspended']
    assert status['errors'] == [{'operation': 'pending_records', 'error': 'OSError'}]


def test_restart_after_failed_shutdown_record_does_not_dispatch_pending(application):
    store, conversation, scheduler = application
    item = message()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    with patch.object(store, 'stop_pending', side_effect=OSError('disk full')):
        scheduler.shutdown.start()
        assert wait_complete(scheduler.shutdown)['termination_confirmed'] is False
    assert store.commands(), 'Failure fixture must leave the original queue on disk'
    reopened = ApplicationStore(store.root)
    reopened.recover_interrupted()
    assert not reopened.commands()
    assert reopened.work(task)['status'] == 'paused'
    assert reopened.snapshot(conversation)['chat_status'] == 'paused'
    assert reopened.snapshot(conversation)['messages'][0]['content'] == item['text']
    resumed = reopened.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume'})
    assert resumed['received'] is True


def test_shutdown_http_is_bounded_and_repeated_requests_do_not_repeat_kill(application):
    from agent.server.app import service_shutdown

    store, conversation, scheduler = application
    entered, release = threading.Event(), threading.Event()
    scheduler.active_jobs['b' * 32] = conversation
    calls = []

    def pause(*args):
        calls.append(args)
        entered.set()
        release.wait(6)
        return {'termination': [{'stopped': False}]}

    scheduler.pause_callback = pause

    def request(body=b'{}'):
        async def receive():
            return {'type': 'http.request', 'body': body, 'more_body': False}
        return Request({'type': 'http', 'method': 'POST', 'path': '/service/shutdown',
                        'headers': []}, receive)

    try:
        with patch('agent.server.app.get_service', return_value=scheduler):
            started = time.monotonic()
            response = asyncio.run(service_shutdown(request()))
            assert time.monotonic() - started < 3
            assert entered.is_set()
            assert json.loads(response.body)['termination_confirmed'] is False
            assert json.loads(response.body)['pending_operations']
            bad = asyncio.run(service_shutdown(request(b'{"pid":123}')))
            assert bad.status_code == 400
            scheduler.shutdown.start()
            assert len(calls) == 1
    finally:
        release.set()
    status = wait_complete(scheduler.shutdown)
    assert status['complete'] is True and status['termination_confirmed'] is False
    assert status['errors'][0]['error'] == 'StopUnconfirmed'


def test_shutdown_endpoint_requires_instance_authentication(application):
    from starlette.testclient import TestClient
    from agent.server.app import app

    _, _, scheduler = application
    with patch('agent.server.app.get_service', return_value=scheduler):
        client = TestClient(app, base_url='http://127.0.0.1:2024')
        assert client.post('/service/shutdown', json={}).status_code == 401
        assert not scheduler.stop.is_set()
        result = client.post('/service/shutdown', json={}, headers={
            'X-MonoPlanner-Service': os.environ['MONOPLANER_SERVICE_TOKEN']})
        assert result.status_code == 200
        assert result.json()['termination_confirmed'] is True


@pytest.mark.skipif(os.name != 'nt', reason='Windows desktop close event')
@pytest.mark.parametrize('already_closed', [False, True])
def test_desktop_close_event_cancels_without_http_readiness(application, monkeypatch, already_closed):
    import ctypes
    from ctypes import wintypes
    from agent.application.shutdown import watch_desktop_close

    _, _, scheduler = application
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    name = 'Local\\MonoPlanner-test-close-' + uuid.uuid4().hex
    handle = kernel.CreateEventW(None, True, already_closed, name)
    assert handle
    monkeypatch.setenv('MONOPLANER_CLOSE_SIGNAL', name)
    close = watch_desktop_close(scheduler.shutdown)
    try:
        if already_closed:
            assert scheduler.stop.is_set(), 'Pre-existing close must be observed synchronously'
        else:
            assert not scheduler.stop.is_set()
            assert kernel.SetEvent(handle)
        assert scheduler.stop.wait(1)
        scheduler.start()
        assert scheduler.worker is None and scheduler.lease is None
        assert wait_complete(scheduler.shutdown)['termination_confirmed'] is True
    finally:
        close()
        kernel.CloseHandle(handle)


def test_lifespan_registers_desktop_close_before_dispatch(application, monkeypatch):
    from agent.application import api, shutdown

    _, _, scheduler = application
    monkeypatch.setattr(api, 'get_service', lambda: scheduler)

    def already_closed(controller):
        assert scheduler.worker is None and not scheduler.starting
        controller.start()
        return lambda: None

    monkeypatch.setattr(shutdown, 'watch_desktop_close', already_closed)

    async def open_then_close():
        async with api.lifespan(None):
            assert scheduler.worker is None and scheduler.lease is None
            assert scheduler.stop.is_set()

    asyncio.run(open_then_close())


def test_environment_prepare_is_fenced_but_cancel_remains_allowed(application, monkeypatch):
    from agent.execution import preparation
    from agent.server import app as server

    _, _, scheduler = application
    scheduler.shutdown.start()
    wait_complete(scheduler.shutdown)
    monkeypatch.setattr(server, 'get_service', lambda: scheduler)
    cancelled = []

    class Environment:
        def cancel(self, repo):
            cancelled.append(repo)
            return {'status': 'cancelled'}

    def request(action):
        async def receive():
            return {'type': 'http.request', 'body': json.dumps({
                'repo': 'sample-project', 'action': action}).encode(), 'more_body': False}
        return Request({'type': 'http', 'method': 'POST', 'path': '/workspace/environment',
                        'headers': []}, receive)

    with patch.object(preparation, 'jobs', return_value=Environment()) as jobs:
        for action in ('prepare', 'automatic'):
            result = asyncio.run(server.project_environment(request(action)))
            assert result.status_code == 400
            assert '正在退出' in json.loads(result.body)['error']
        jobs.assert_not_called()
        result = asyncio.run(server.project_environment(request('cancel')))
        assert result.status_code == 200 and cancelled == ['sample-project']


def test_reconcile_cannot_reset_closed_service(application, monkeypatch):
    from agent.application import api
    from agent.server.app import service_reconcile

    store, _, scheduler = application
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(store.root))
    monkeypatch.setattr(api, '_service', scheduler)
    scheduler.shutdown.start()
    wait_complete(scheduler.shutdown)

    async def receive():
        return {'type': 'http.request', 'body': b'{}', 'more_body': False}

    request = Request({'type': 'http', 'method': 'POST', 'path': '/service/reconcile',
                       'headers': []}, receive)
    assert asyncio.run(service_reconcile(request)).status_code == 200
    assert api.get_service() is scheduler
    assert scheduler.stop.is_set() and store.closing.is_set()
