"""按钮、实时任务事实和完成通知进入聊天；查询不改变原工作。"""

import json
from unittest.mock import patch

import pytest

from agent.application.chat import Chatbot
from agent.application.scheduler import Scheduler
from agent.application.store import ApplicationStore
from tests.test_application_runtime import FakeModel, answer, command, delivery_args


@pytest.fixture
def app(tmp_path):
    repo = tmp_path / 'calculator'
    repo.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(repo))['thread_id']
    return store, conversation, repo


def job(app, *, status='needs_plan', kind='coding'):
    store, conversation, _ = app
    request = command('制作加减乘除计算器')
    store.submit(conversation, request)
    task = store.create_work(conversation, request['id'], kind=kind)
    store.command_state(conversation, request['id'], 'applied')
    store.work_state(task['task_id'], status)
    return store.work(task['task_id'])


def context_in(messages):
    found = []
    for message in messages:
        if message.type != 'human':
            continue
        try:
            value = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get('source') == 'saved_work_context':
            found.append(value)
    return found[-1] if found else {}


def test_chat_receives_confirmed_button_before_its_first_reply(app, monkeypatch):
    from agent.application.requirements import prepare

    store, conversation, _ = app
    request = command('制作计算器')
    store.submit(conversation, request)
    proposal = prepare(store, conversation, request, **delivery_args(request))
    store.command_state(conversation, request['id'], 'applied')
    confirmation = {**command(kind='confirm_work'), 'proposal_id': proposal['proposal_id']}
    store.submit(conversation, confirmation)
    scheduler = Scheduler(store)
    monkeypatch.setattr(scheduler, '_schedule_work', lambda *_: None)
    try:
        scheduler.tick()
        task = store.work(store.get(conversation)['work_id'])
        store.work_state(task['task_id'], 'needs_plan', reason='evidence_ready')
        question = command('我点过了，现在呢')
        store.submit(conversation, question)
        model = FakeModel([answer('已确认，正在准备下一步。')])
        bot = Chatbot(store, lambda *_: pytest.fail('Status query started work'),
                      factory=lambda _: model)
        bot.run(conversation, question)
        facts = context_in(model.requests[0])
        assert facts['task']['confirmation']['state'] == 'applied'
        assert facts['task']['confirmation']['id'] == confirmation['id']
        assert facts['task']['lifecycle'] == 'working'
        assert facts['task']['needs_user_action'] is False
        assert len(store.work_for_conversation(conversation)) == 1
    finally:
        scheduler.close()


def test_work_facts_include_applied_pause_not_only_pending_controls(app):
    store, conversation, _ = app
    task = job(app, status='executing')
    pause = command(kind='pause')
    store.submit(conversation, pause)
    store.command_state(conversation, pause['id'], 'applied')
    store.work_state(task['task_id'], 'paused', reason='User requested pause')
    facts = Chatbot(store, lambda *_: None).work_facts(conversation)
    assert facts['task']['pause_commands'] == [{'id': pause['id'], 'state': 'applied'}]


def test_later_command_does_not_hide_completed_coding_in_chat(app):
    store, conversation, _ = app
    coding = job(app, status='verified_success')
    existing = job(app, status='existing_complete', kind='existing')
    question = command('计算器完成了吗，放哪了')
    store.submit(conversation, question)
    model = FakeModel([answer('计算器已完成。')])
    Chatbot(store, lambda *_: pytest.fail('Question must stay read-only'),
            factory=lambda _: model).run(conversation, question)
    facts = context_in(model.requests[0])
    assert facts['task']['task_id'] == existing['task_id']
    assert facts['task']['kind'] == 'existing'
    assert facts['coding_task']['task_id'] == coding['task_id']
    assert facts['coding_task']['status'] == 'verified_success'
    assert facts['coding_task']['repo'] == coding['repo']


def test_context_refresh_preserves_old_history_and_does_not_repeat_unchanged_facts(app):
    store, conversation, _ = app
    task = job(app)
    model = FakeModel([answer('正在准备。'), answer('已经完成。')])
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model)
    first = command('进度怎么样')
    store.submit(conversation, first)
    bot.run(conversation, first)
    original = list(store.model_history(conversation))
    store.work_state(task['task_id'], 'verified_success')
    second = command('现在呢')
    store.submit(conversation, second)
    bot.run(conversation, second)
    assert store.model_history(conversation)[:len(original)] == original
    facts = context_in(model.requests[-1])
    assert facts['task']['status'] == 'verified_success'
    assert facts['task']['needs_user_action'] is False
    assert facts['task']['lifecycle'] == 'completed'


def test_verified_completion_is_immediate_once_and_needs_no_summary_model(app):
    store, conversation, repo = app
    task = job(app, status='executing')
    scheduler = Scheduler(store)
    scheduler.chatbot.progress = lambda *_: pytest.fail('Completion must not wait for a model')
    try:
        scheduler.emit(task['task_id'], 'work_state', {
            'status': 'verified_success', 'stopped_confirmed': True,
            'reason': 'Fixed final acceptance passed',
        })
        scheduler.emit(task['task_id'], 'work_state', {
            'status': 'verified_success', 'stopped_confirmed': True,
        })
        notices = [m for m in store.snapshot(conversation)['messages']
                   if m['source'] == 'progress']
        assert len(notices) == 1
        assert '完成' in notices[0]['content'] and str(repo) in notices[0]['content']
        assert all(word not in notices[0]['content'] for word in
                   ['暂停', 'revision', 'verified_success', '继续复核'])
        assert store.usage(conversation)['progress']['calls'] == 0
    finally:
        scheduler.close()


def test_completed_notice_survives_later_work_selection_without_new_request(app):
    store, conversation, _ = app
    coding = job(app, status='verified_success')
    seq = store.work_state(coding['task_id'], 'verified_success')
    store.progress_pending(coding['task_id'], seq)
    pending = store.reserve_progress(now=100)
    job(app, kind='existing', status='existing_complete')
    scheduler = Scheduler(store)
    scheduler.chatbot.progress = lambda *_: pytest.fail('No model needed for saved completion')
    try:
        scheduler._report(pending)
        notices = [m for m in store.snapshot(conversation)['messages']
                   if m['source'] == 'progress']
        assert len(notices) == 1 and '验收' in notices[0]['content']
    finally:
        scheduler.close()


def test_running_progress_cannot_publish_after_its_revision_changes(app):
    store, conversation, _ = app
    task = job(app, status='executing')
    seq = store.work_state(task['task_id'], 'executing')
    store.progress_pending(task['task_id'], seq)
    pending = store.reserve_progress(now=100)
    scheduler = Scheduler(store)

    original = scheduler.chatbot.work_facts

    def raced(*args, **kwargs):
        facts = original(*args, **kwargs)
        store.work_state(task['task_id'], 'executing', reason='new evidence')
        return facts

    scheduler.chatbot.work_facts = raced
    try:
        scheduler._report(pending)
        assert not [m for m in store.snapshot(conversation)['messages']
                    if m['source'] == 'progress']
    finally:
        scheduler.close()


def test_foreign_task_cannot_be_selected_for_report(app, tmp_path):
    store, conversation, _ = app
    other = store.create(repo=str(tmp_path))['thread_id']
    request = command()
    store.submit(other, request)
    foreign = store.create_work(other, request['id'])
    with pytest.raises(ValueError, match='conversation'):
        Chatbot(store, lambda *_: None).work_facts(conversation, task_id=foreign['task_id'])


def test_unrequested_existing_check_cannot_start_or_replace_finished_coding(app, monkeypatch):
    store, conversation, _ = app
    coding = job(app, status='verified_success')
    request = command('完成了吗')
    store.submit(conversation, request)
    model = FakeModel([answer(tool_calls=[{
        'name': 'run_existing', 'id': 'mistaken-extra-run',
        'args': {'entry': 'calc.js', 'args': ['2', '+', '3'],
                 'timeout': 30, 'usage_file': ''},
    }])])
    bot = Chatbot(store, lambda *_args, **_kwargs: pytest.fail('A question dispatched a command'),
                  factory=lambda _: model)
    bot.run(conversation, request)
    assert store.get(conversation)['work_id'] == coding['task_id']
    assert len(store.work_for_conversation(conversation)) == 1
    proposal = next(m for m in store.snapshot(conversation)['messages']
                    if m['source'] == 'work_proposal')
    assert 'calc.js' in proposal['content'] and '确认' in proposal['content']
    confirmation = {**command(kind='confirm_work'),
                    'proposal_id': proposal['id'].split(':')[1]}
    store.command_state(conversation, request['id'], 'applied')
    store.submit(conversation, confirmation)
    scheduler = Scheduler(store)
    monkeypatch.setattr(scheduler, '_schedule_work', lambda *_: None)
    try:
        scheduler.tick()
        current = store.work(store.get(conversation)['work_id'])
        assert current['kind'] == 'existing' and current['arguments']['entry'] == 'calc.js'
        assert store.work(coding['task_id'])['status'] == 'verified_success'
        store.submit(conversation, {**confirmation, 'id': command()['id']})
        scheduler.tick()
        assert len(store.work_for_conversation(conversation)) == 2
    finally:
        scheduler.close()


def test_old_task_notification_cannot_overwrite_active_task_status(app):
    store, conversation, _ = app
    coding = job(app, status='verified_success')
    current = job(app, kind='existing', status='executing')
    store.work_state(coding['task_id'], 'verified_success')
    assert store.get(conversation)['work_id'] == current['task_id']
    assert store.get(conversation)['status'] == 'executing'


@pytest.mark.parametrize('fault', ['unconfirmed', 'changed_task', 'tampered'])
def test_existing_run_rejects_missing_confirmation_or_changed_proposal(app, fault):
    from agent.application.existing_proposal import accept, prepare

    store, conversation, _ = app
    job(app, status='verified_success')
    request = command('运行一次现有命令')
    store.submit(conversation, request)
    proposal = prepare(store, conversation, request, {
        'entry': 'calc.js', 'args': ['2', '+', '3'], 'timeout': 30, 'usage_file': '',
    })
    confirmation = {**command(kind='confirm_work'), 'proposal_id': proposal['proposal_id']}
    if fault != 'unconfirmed':
        store.submit(conversation, confirmation)
    if fault == 'changed_task':
        job(app, status='verified_success')
    elif fault == 'tampered':
        path = store.root / 'work-proposals' / conversation / (proposal['proposal_id'] + '.json')
        value = json.loads(path.read_text('utf-8'))
        value['arguments']['entry'] = 'different.js'
        path.write_text(json.dumps(value), 'utf-8')
    count = len(store.work_for_conversation(conversation))
    with pytest.raises(ValueError):
        accept(store, conversation, confirmation, proposal['proposal_id'])
    assert len(store.work_for_conversation(conversation)) == count
