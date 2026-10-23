"""普通进度由事件投影，不让模型把启动阶段猜成文件修改。"""

from unittest.mock import patch

import pytest

from agent.application.chat import Chatbot
from agent.application.scheduler import Scheduler
from tests.test_chatbot_work_context import app, job


def notices(store, conversation):
    return [row['content'] for row in store.snapshot(conversation)['messages']
            if row['source'] == 'progress']


def test_startup_is_explicit_and_does_not_claim_editing(app):
    store, conversation, _ = app
    job(app, status='running')
    activity = Chatbot(store, lambda *_: None).work_facts(conversation)['task']['activity']
    assert activity['phase'] == 'starting'
    assert activity['file_changes_confirmed'] is False
    assert '启动准备' in activity['message']


def test_normal_progress_needs_no_model_and_deduplicates_same_phase(app):
    store, conversation, _ = app
    task = job(app, status='needs_plan')
    scheduler = Scheduler(store)
    try:
        with patch.object(scheduler.chatbot, 'progress', side_effect=AssertionError('No model needed')):
            for now in (100, 131):
                seq = store.work_state(task['task_id'], 'needs_plan', reason='evidence_ready')
                store.progress_pending(task['task_id'], seq)
                scheduler._report(store.reserve_progress(now=now))
        saved = notices(store, conversation)
        assert len(saved) == 1
        assert '准备下一步' in saved[0]
        assert all(word not in saved[0] for word in ['revision', 'evidence_ready', '确认方案'])
        assert store.usage(conversation)['progress']['calls'] == 0
    finally:
        scheduler.close()


def test_verification_progress_comes_from_saved_acceptance_event(app):
    store, conversation, _ = app
    task = job(app, status='executing')
    seq = store.work_state(task['task_id'], 'executing', event='final_acceptance_started')
    store.progress_pending(task['task_id'], seq)
    scheduler = Scheduler(store)
    def no_model(*_):
        raise AssertionError('No model needed')

    scheduler.chatbot.progress = no_model
    try:
        scheduler._report(store.reserve_progress(now=100))
        saved = notices(store, conversation)
        assert len(saved) == 1 and '验收' in saved[0]
        assert '通过' not in saved[0]
        assert store.usage(conversation)['progress']['calls'] == 0
    finally:
        scheduler.close()


@pytest.mark.parametrize('status,old,new,phase', [
    ('needs_plan', 'plan_rejected', 'acceptance_trial_started', 'preparing_checks'),
    ('executing', 'final_acceptance_started', 'plan_accepted', 'working'),
])
def test_latest_phase_replaces_older_event_in_the_same_page(app, status, old, new, phase):
    store, conversation, _ = app
    task = job(app, status=status)
    store.work_state(task['task_id'], status, event=old)
    store.work_state(task['task_id'], status, event=new)
    store.work_state(task['task_id'], status, event='model_request_started')
    facts = Chatbot(store, lambda *_: None).work_facts(conversation)
    assert facts['task']['activity']['phase'] == phase
