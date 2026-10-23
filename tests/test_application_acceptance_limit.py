"""验收额度的人机交接、稳定公开事实和控制命令幂等性。"""

from concurrent.futures import ThreadPoolExecutor
import uuid
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from agent.application.api import build_routes
from agent.application.scheduler import Scheduler
from agent.application.store import ACTIVE, TERMINAL, ApplicationStore, Conflict
from agent.application.work_bridge import public_fact
from agent.task.acceptance_limit import STATUS


def command(kind, **fields):
    return {'id': str(uuid.uuid4()), 'kind': kind, **fields}


def limit(rounds=3):
    return {
        'active': True, 'completed_rounds': rounds, 'round_limit': rounds,
        'round_increment': 3, 'pause_id': 'pause-' + str(rounds),
        'last_verification_status': 'failed',
    }


@pytest.fixture
def work(tmp_path, monkeypatch):
    root, repo = tmp_path / 'state', tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(root))
    store = ApplicationStore(root)
    conversation = store.create(repo=str(repo))['thread_id']
    first = command('message', text='修改项目并实际验收')
    store.submit(conversation, first)
    job = store.create_work(conversation, first['id'])
    store.command_state(conversation, first['id'], 'applied')
    store.work_state(job['task_id'], STATUS, acceptance_limit=limit())
    return store, conversation, job


def test_acceptance_pause_has_durable_notice_snapshot_and_sse_without_model(work):
    store, conversation, job = work
    from agent.task.resumption import stopped
    assert stopped(STATUS) and STATUS not in TERMINAL and STATUS not in ACTIVE
    store.work_state(job['task_id'], STATUS, acceptance_limit=limit())
    client = TestClient(Starlette(routes=build_routes(store, object())))
    snapshot = client.get(f'/conversations/{conversation}/snapshot').json()
    assert snapshot['work']['acceptance_limit'] == limit()
    notices = [row for row in snapshot['messages'] if row['source'] == 'system']
    assert len(notices) == 1
    assert '3 轮' in notices[0]['content'] and '未通过' in notices[0]['content']
    assert '继续任务' in notices[0]['content'] and '断点' in notices[0]['content']
    events = client.get(f'/conversations/{conversation}/events?once=1').json()['events']
    states = [row['payload'] for row in events if row['kind'] == 'work_state']
    assert states[-1]['acceptance_limit'] == limit()
    store.recover_interrupted()
    restored = ApplicationStore(store.root).snapshot(conversation)
    assert restored['status'] == STATUS
    assert restored['work']['acceptance_limit'] == limit()
    assert [row['id'] for row in restored['messages']] == [row['id'] for row in snapshot['messages']]


def test_acceptance_pause_rejects_supplement_and_new_handoff_but_allows_chat(work):
    store, conversation, job = work
    with pytest.raises(Conflict):
        store.submit(conversation, command('supplement', text='继续'))
    message = command('message', text='解释一下这次失败')
    store.submit(conversation, message)
    assert not store.commands(task=job['task_id'])
    scheduler = Scheduler(store)
    try:
        with pytest.raises(Conflict):
            scheduler.dispatch_work(conversation, message)
        assert store.get(conversation)['work_id'] == job['task_id']
        assert store.work(job['task_id'])['status'] == STATUS
    finally:
        scheduler.close()


def test_concurrent_resume_clicks_reserve_only_one_batch_and_bind_pause(work):
    store, conversation, job = work
    commands = [command('resume', acceptance_pause_id='pause-3') for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda value: store.submit(conversation, value), commands))
    assert sum(not receipt['applied'] for receipt in receipts) == 1
    pending = store.commands(task=job['task_id'])
    assert len(pending) == 1
    assert pending[0]['acceptance_pause_id'] == 'pause-3'
    scheduler = Scheduler(store)
    try:
        assert scheduler.controls(job['task_id'])[0]['acceptance_pause_id'] == 'pause-3'
    finally:
        scheduler.close()


def test_stale_page_cannot_extend_new_pause_and_command_retry_keeps_identity(work):
    store, conversation, job = work
    resume = command('resume', acceptance_pause_id='pause-3')
    store.submit(conversation, resume)
    store.work_state(job['task_id'], STATUS, acceptance_limit=limit(6),
                     applied_inputs=[resume['id']])
    assert store.submit(conversation, resume)['applied']
    with pytest.raises(Conflict, match='已变化'):
        store.submit(conversation, command('resume', acceptance_pause_id='pause-3'))
    fresh = command('resume', acceptance_pause_id='pause-6')
    assert not store.submit(conversation, fresh)['applied']
    assert store.commands(task=job['task_id'])[0]['acceptance_pause_id'] == 'pause-6'
    assert len([m for m in store.snapshot(conversation)['messages'] if m['source'] == 'system']) == 2


@pytest.mark.parametrize('rounds', [3, 6])
def test_unbound_resume_cannot_authorize_a_later_acceptance_pause(work, rounds):
    store, conversation, job = work
    delayed_resume = command('resume')
    store.work_state(job['task_id'], STATUS, acceptance_limit=limit(rounds))
    with pytest.raises(Conflict, match='已变化'):
        store.submit(conversation, delayed_resume)
    assert not store.commands(task=job['task_id'])
    assert store.work(job['task_id'])['acceptance_limit'] == limit(rounds)


def test_ordinary_pause_resume_keeps_existing_unbound_behavior(work):
    store, conversation, job = work
    store.work_state(job['task_id'], 'paused', acceptance_limit=None)
    resume = command('resume')
    assert not store.submit(conversation, resume)['applied']
    row = store.commands(task=job['task_id'])[0]
    assert row['acceptance_pause_id'] is None
    assert 'acceptance_pause_id' not in row['value']


def test_pause_and_shutdown_keep_acceptance_pause_and_do_not_dispatch(work):
    store, conversation, job = work
    scheduler = Scheduler(store, work_factory=lambda *_: pytest.fail('Unexpected execution'),
                          pause_callback=lambda *_: {'termination': []})
    try:
        pause = command('pause')
        scheduler.submit(conversation, pause)
        scheduler.tick()
        store.work_state(job['task_id'], 'paused')
        store.stop_pending()
        assert store.work(job['task_id'])['status'] == STATUS
        assert store.work(job['task_id'])['acceptance_limit'] == limit()
        assert scheduler.status()['running_work'] == 0
    finally:
        scheduler.close()


@pytest.mark.parametrize('outcome', ['same_pause', 'unknown', 'exception'])
def test_unconsumed_resume_does_not_spin_after_work_returns(work, outcome):
    store, conversation, job = work
    resume = command('resume', acceptance_pause_id='pause-3')
    store.submit(conversation, resume)
    calls = []

    def run(*_):
        calls.append(True)
        if outcome == 'exception':
            raise ValueError('Original checkpoint failed validation')
        return {'status': STATUS if outcome == 'same_pause' else 'execution_unknown'}

    scheduler = Scheduler(store, work_factory=run)
    try:
        scheduler._work(job)
        scheduler.tick()
        assert len(calls) == 1
        assert not store.commands(task=job['task_id'])
        assert store.commands(task=job['task_id'], states=('failed',))[0]['id'] == resume['id']
    finally:
        scheduler.close()


def test_acceptance_notice_does_not_require_progress_model(work):
    store, conversation, job = work
    scheduler = Scheduler(store)
    scheduler.chatbot.progress = lambda *_: pytest.fail('Pause notice requested a model')
    try:
        scheduler.emit(job['task_id'], 'work_state', {
            'status': STATUS, 'event': STATUS, 'acceptance_limit': limit(),
        })
        assert store.reserve_progress(now=1_000_000) is None
        scheduler._report({'task_id': job['task_id'], 'conversation': conversation,
                           'reported_seq': 0, 'pending_seq': 10})
        assert any('继续任务' in row['content'] for row in store.snapshot(conversation)['messages'])
    finally:
        scheduler.close()




def test_public_progress_facts_include_only_acceptance_summary():
    state = {'acceptance_limit': {**limit(), 'private_test': 'DO_NOT_PUBLISH'}}
    details = public_fact(state, STATUS)
    assert details['acceptance_limit'] == limit()
    assert 'DO_NOT_PUBLISH' not in str(details)


def test_coalesced_clicks_do_not_block_new_decision_after_cancelled_dispatch(work):
    store, conversation, job = work
    first = command('resume', acceptance_pause_id='pause-3')
    repeated = command('resume', acceptance_pause_id='pause-3')
    store.submit(conversation, first)
    assert store.submit(conversation, repeated)['applied']
    store.command_state(conversation, first['id'], 'failed')
    assert store.submit(conversation, repeated)['applied']
    fresh = command('resume', acceptance_pause_id='pause-3')
    assert not store.submit(conversation, fresh)['applied']
    assert [row['id'] for row in store.commands(task=job['task_id'])] == [fresh['id']]




def test_chatbot_status_includes_public_acceptance_budget_after_reload(work):
    from agent.application.chat import Chatbot

    store, conversation, job = work
    store.work_state(job['task_id'], STATUS,
                     acceptance_limit={**limit(), 'private_test': 'DO_NOT_PUBLISH'})
    restored = ApplicationStore(store.root)
    bot = Chatbot(restored, lambda *_: pytest.fail('Status read dispatched work'))
    facts = bot.work_facts(conversation)
    assert facts['task']['acceptance_limit'] == limit()
    assert 'private_test' not in facts['task']['acceptance_limit']
