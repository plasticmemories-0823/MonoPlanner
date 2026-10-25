"""完成通知与旧停止汇报隔离，重启和并发发布不改写历史。"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.application.scheduler import Scheduler
from agent.application.store import Conflict
from tests.test_chatbot_work_context import app, job


@pytest.mark.parametrize('status', ['verified_success', 'existing_complete'])
def test_startup_preserves_legacy_report_and_publishes_completion_once(app, monkeypatch, status):
    store, conversation, _ = app
    task = job(app, status=status)
    old_id = 'stop-report:' + task['task_id'] + ':' + task['stop_report_id']
    store.publish(conversation, old_id, '此前保存的模型汇报。', 'progress')
    original = store.snapshot(conversation)['messages'][-1]

    for _ in range(2):
        scheduler = Scheduler(store)
        monkeypatch.setattr(scheduler, '_loop', lambda: scheduler.stop.wait(5))
        scheduler.chatbot.progress = lambda *_: pytest.fail('Startup requested a model')
        try:
            scheduler.start()
            assert scheduler.worker.is_alive()
        finally:
            scheduler.close()

    messages = store.snapshot(conversation)['messages']
    assert next(row for row in messages if row['id'] == old_id) == original
    completed = [row for row in messages if row['id'].startswith('completion:')]
    assert len(completed) == 1 and str(task['repo']) in completed[0]['content']
    assert store.usage(conversation)['progress']['calls'] == 0


def test_saved_completion_is_not_rewritten_when_display_wording_changes(app, monkeypatch):
    store, conversation, _ = app
    task = job(app, status='verified_success')
    scheduler = Scheduler(store)
    try:
        assert scheduler._publish_completion(task)
        saved = store.snapshot(conversation)['messages']
        monkeypatch.setattr('agent.application.work_context.completion_notice',
                            lambda _: '更新后的显示文案，不是新的任务结果。')
        assert scheduler._publish_completion(task)
        assert store.snapshot(conversation)['messages'] == saved
    finally:
        scheduler.close()


def test_stale_completed_snapshot_cannot_announce_after_task_resumes(app):
    store, conversation, _ = app
    task = job(app, status='verified_success')
    store.work_state(task['task_id'], 'needs_plan')
    scheduler = Scheduler(store)
    try:
        assert not scheduler._publish_completion(task)
        assert not [row for row in store.snapshot(conversation)['messages']
                    if row['source'] == 'progress']
    finally:
        scheduler.close()


def test_concurrent_completion_publishers_commit_one_notice(app):
    store, conversation, _ = app
    task = job(app, status='verified_success')
    scheduler = Scheduler(store)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert all(pool.map(lambda _: scheduler._publish_completion(task), range(2)))
        assert len([row for row in store.snapshot(conversation)['messages']
                    if row['source'] == 'progress']) == 1
    finally:
        scheduler.close()


def test_ordinary_message_collision_remains_an_error(app):
    store, conversation, _ = app
    store.publish(conversation, 'ordinary-message', '原内容')
    with pytest.raises(Conflict, match='different content'):
        store.publish(conversation, 'ordinary-message', '另一内容')


def test_completion_storage_failure_is_not_silently_accepted(app, monkeypatch):
    store, conversation, _ = app
    task = job(app, status='verified_success')
    scheduler = Scheduler(store)

    def fail(*_args, **_kwargs):
        raise OSError('disk write unavailable')

    monkeypatch.setattr(store, '_message', fail)
    try:
        with pytest.raises(OSError, match='disk write unavailable'):
            scheduler._publish_completion(task)
        assert not [row for row in store.snapshot(conversation)['messages']
                    if row['source'] == 'progress']
    finally:
        scheduler.close()
