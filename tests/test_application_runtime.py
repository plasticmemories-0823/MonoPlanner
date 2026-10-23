"""独立前厅的持久命令、隔离上下文和无重放验证；全部使用假模型。"""

import json
import threading
import time
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage, message_to_dict
from starlette.applications import Starlette
from starlette.testclient import TestClient

from agent.application.api import build_routes
from agent.application.chat import Chatbot
from agent.application.scheduler import Scheduler
from agent.application.store import ApplicationStore, Conflict


def command(text='请直接修改项目', kind='message'):
    return {'id': str(uuid.uuid4()), 'kind': kind, 'text': text}


def test_missing_frozen_runtime_stops_before_work_and_can_be_explicitly_retried(setup):
    from agent.runtime.bridge import WorkerUnavailable

    store, conversation, _ = setup
    incoming = command()
    store.submit(conversation, incoming)
    job = store.create_work(conversation, incoming['id'])
    store.command_state(conversation, incoming['id'], 'applied')

    def unavailable(*_):
        raise WorkerUnavailable('Engine file integrity failure')

    scheduler = Scheduler(store, work_factory=unavailable)
    try:
        scheduler._work(job)
        saved = store.work(job['task_id'])
        assert saved['status'] == 'paused'
        assert 'Engine file integrity failure' in saved['reason']
        assert store.usage(conversation)['work']['calls'] == 0
        result = store.submit(conversation, command(kind='resume'))
        assert result['received']
    finally:
        scheduler.close()


def answer(content='', **kwargs):
    return AIMessage(content=content, response_metadata={'finish_reason': 'stop'},
                     usage_metadata={'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
                                     'input_token_details': {'cache_read': 60}}, **kwargs)


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.tools = []

    def bind_tools(self, tools):
        self.tools.append(tuple(tool.name for tool in tools))
        return self

    def invoke(self, messages):
        self.requests.append(messages)
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'state'
    repo = tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(root))
    store = ApplicationStore(root)
    conversation = store.create(repo=str(repo))['thread_id']
    return store, conversation, repo


def test_creation_and_command_are_immediate_and_idempotent(setup):
    store, conversation, _ = setup
    item = command('即时出现')
    store.submit(conversation, item)
    store.submit(conversation, item)
    snapshot = store.snapshot(conversation)
    assert [m['content'] for m in snapshot['messages']] == ['即时出现']
    assert store.list()[0]['thread_id'] == conversation
    with pytest.raises(Conflict):
        store.submit(conversation, {**item, 'text': '不同内容'})


def test_work_handoff_preserves_user_and_excludes_progress(setup):
    store, conversation, _ = setup
    earlier = command('必须保留现有接口')
    store.submit(conversation, earlier)
    store.publish(conversation, 'plan', '已讨论的方案')
    store.publish(conversation, 'report', '自动进度不能进入工作', 'progress')
    item = command('现在执行，不改验收')
    store.submit(conversation, item)
    work = store.create_work(conversation, item['id'])
    assert work['request'] == item['text']
    original = json.dumps(work['public_conversation'], ensure_ascii=False)
    assert earlier['text'] in original and '已讨论的方案' in original
    assert '自动进度不能进入工作' not in original
    assert store.create_work(conversation, item['id'])['task_id'] == work['task_id']


def test_ordinary_chat_does_not_enter_work_controls(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    work = store.create_work(conversation, item['id'])
    store.submit(conversation, command('现在进度怎么样'))
    assert store.commands(task=work['task_id']) == []
    supplement = command('还要保留日志', 'supplement')
    store.submit(conversation, supplement)
    assert store.commands(task=work['task_id'])[0]['value']['text'] == supplement['text']


def test_active_workspace_change_is_rejected(setup, tmp_path):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    store.create_work(conversation, item['id'])
    with pytest.raises(Conflict):
        store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'workspace', 'repo': str(tmp_path)})


def test_recovery_classifies_inflight_without_replaying(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    work = store.create_work(conversation, item['id'])
    store.work_state(work['task_id'], 'running')
    store.command_state(conversation, item['id'], 'running')
    store.chat_state(conversation, 'running')
    call = store.begin_call(conversation, 'chat', {'message_ids': []})
    store.mark_sent(call)
    store.recover_interrupted()
    snapshot = store.snapshot(conversation)
    assert snapshot['work']['status'] == 'interrupted'
    assert snapshot['chat_status'] == 'model_request_unknown'
    assert snapshot['usage']['chat']['unknown_calls'] == 1
    assert not store.commands(states=('running',))
    assert store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume'})['received']
    assert store.work(work['task_id'])['status'] == 'interrupted'


def test_startup_stops_orphan_resources_without_resuming_work(setup, monkeypatch):
    from agent.runtime import bridge

    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    job = store.create_work(conversation, item['id'])
    store.work_state(job['task_id'], 'running')
    stopped = []
    monkeypatch.setattr(bridge, 'pause_work', lambda task, control, **options:
                        stopped.append((task, options['root'])) or {'termination': []})
    scheduler = Scheduler(store, work_factory=lambda *_: pytest.fail('Crash cannot auto-resume'))
    try:
        scheduler.start()
        assert stopped == [(job['task_id'], store.root)]
        assert store.work(job['task_id'])['status'] == 'interrupted'
    finally:
        scheduler.close()


def test_only_validated_native_resume_clears_old_unconfirmed_stop(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    store.control_unconfirmed(task, 'old-pause')
    store.work_state(task, 'paused', event='task_resumed')
    assert store.work(task)['status'] == 'execution_unknown'
    resume = command(kind='resume')
    store.submit(conversation, resume)
    store.work_state(task, 'needs_plan', event='task_resumed', applied_inputs=[resume['id']])
    assert store.work(task)['status'] == 'needs_plan'
    assert not store.work(task)['control_unconfirmed']


def test_chat_preserves_rejected_tool_reasoning_and_call_pair(setup):
    store, conversation, _ = setup
    item = command('读取文件')
    store.submit(conversation, item)
    first = answer('先读取', additional_kwargs={'reasoning_content': 'saved reasoning'},
                   invalid_tool_calls=[{'name': 'read_file', 'id': 'bad', 'args': '{broken',
                                        'error': 'invalid json'}])
    model = FakeModel([first, answer('参数未执行')])
    invoked = []
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model,
                  invoke_tool=lambda *args: invoked.append(args))
    bot.run(conversation, item)
    assert not invoked
    next_messages = model.requests[1]
    retained = next(m for m in next_messages if isinstance(m, AIMessage))
    assert retained.additional_kwargs['reasoning_content'] == 'saved reasoning'
    assert retained.invalid_tool_calls[0]['args'] == '{broken'
    paired = next(m for m in next_messages if isinstance(m, ToolMessage))
    assert paired.tool_call_id == 'bad' and 'actions_executed' in paired.content
    assert model.tools[0] == model.tools[1]


def delivery_args(item, project_path=".", create=False):
    return {"project_path": project_path, "create": create, "request_message_ids": [item["id"]],
            "targets": [{"id": "application", "artifact_type": "repository_entry",
                         "target_runtime": "python", "entry": "app.py", "behavior": item["text"],
                         "input_scope": "All requested inputs", "imports": "", "setup": "", "expression": "",
                         "variables": [], "required_values": [], "argv": [],
                         "quotes": [{"source_id": item["id"], "text": item["text"]}]}]}


def test_handoff_calls_work_once_after_human_confirms_proposal(setup, monkeypatch):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    model = FakeModel([answer(tool_calls=[{
        'name': 'prepare_work', 'args': delivery_args(item), 'id': 'proposal',
    }])])
    bot = Chatbot(store, lambda *_: pytest.fail('Model self-confirmed work'), factory=lambda _: model)
    bot.run(conversation, item)
    assert store.snapshot(conversation)['work'] is None and len(model.requests) == 1
    proposal = next(row for row in store.snapshot(conversation)['messages'] if row['source'] == 'work_proposal')
    confirmation = {'id': str(uuid.uuid4()), 'kind': 'confirm_work',
                    'proposal_id': proposal['id'].split(':')[1]}
    store.command_state(conversation, item['id'], 'applied')
    store.submit(conversation, confirmation)
    scheduler = Scheduler(store)
    monkeypatch.setattr(scheduler, '_schedule_work', lambda *_: None)
    try:
        scheduler.tick()
        task = store.snapshot(conversation)['work']
        assert task['status'] == 'queued'
        original = store.work(task['task_id'])
        assert original['request'] == item['text']
        assert original['confirmed_requirements']['confirmation']['id'] == confirmation['id']
        store.submit(conversation, {**confirmation, 'id': str(uuid.uuid4())})
        scheduler.tick()
        assert len(store.work_for_conversation(conversation)) == 1
    finally:
        scheduler.close()


def test_progress_has_read_only_tools_and_no_work_history_effect(setup):
    store, conversation, _ = setup
    model = FakeModel([
        answer(tool_calls=[{'name': 'prepare_work', 'args': {}, 'id': 'forbidden'}]),
        answer('已完成调查，尚未验收'),
    ])
    dispatches = []
    bot = Chatbot(store, lambda *_: dispatches.append(True), factory=lambda _: model)
    result = bot.progress(conversation, {'task': {'status': 'executing'}})
    assert not dispatches
    assert set(model.tools[0]) == {'get_work_status', 'read_work_events', 'read_work_record'}
    assert result['output'] == '已完成调查，尚未验收'
    assert not store.model_history(conversation, 'chat')
    assert store.snapshot(conversation)['messages'] == []
    assert store.usage(conversation)['progress']['calls'] == 2


def test_progress_debounces_events_for_at_least_30_seconds(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    store.progress_pending(task, 5)
    assert store.reserve_progress(now=100)['pending_seq'] == 5
    assert store.reserve_progress(now=140) is None
    store.finish_progress(task, 5)
    assert store.reserve_progress(now=140) is None
    store.progress_pending(task, 10)
    assert store.reserve_progress(now=129) is None
    assert store.reserve_progress(now=130)['pending_seq'] == 10


def test_sse_read_and_refresh_never_invoke_model(setup):
    store, conversation, _ = setup

    class NoCalls:
        def submit(self, *_):
            raise AssertionError('read attempted to execute a command')

    client = TestClient(Starlette(routes=build_routes(store, NoCalls())))
    first = client.get(f'/conversations/{conversation}/events?once=1').json()
    assert first['events']
    again = client.get(f'/conversations/{conversation}/events?once=1&after={first["cursor"]}').json()
    assert again['events'] == []
    assert client.get(f'/conversations/{conversation}/snapshot').status_code == 200
    assert store.usage(conversation)['chat']['calls'] == 0


def test_user_chat_can_complete_while_work_executor_is_blocked(setup):
    store, conversation, _ = setup
    entered, release = threading.Event(), threading.Event()

    def work(job, emit, controls):
        entered.set()
        release.wait(5)
        return {'status': 'paused'}

    model = FakeModel([answer('正在执行，不影响这次问答')])
    scheduler = Scheduler(store, work_factory=work, model_factory=lambda _: model)
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])
    store.command_state(conversation, item['id'], 'applied')
    scheduler._schedule_work(task)
    assert entered.wait(1)
    incoming = command('解释一下，不修改当前任务')
    store.submit(conversation, incoming)
    try:
        scheduler._chat(conversation, incoming)
        assert not release.is_set()
        assert store.snapshot(conversation)['messages'][-1]['content'] == '正在执行，不影响这次问答'
        assert scheduler.controls(task['task_id']) == []
    finally:
        release.set()
        scheduler.close()


def test_api_reports_conflicting_duplicate_as_409(setup):
    store, conversation, _ = setup

    class Receiver:
        submit = staticmethod(store.submit)

    client = TestClient(Starlette(routes=build_routes(store, Receiver())))
    item = command('一')
    assert client.post(f'/conversations/{conversation}/commands', json=item).status_code == 202
    assert client.post(f'/conversations/{conversation}/commands', json={**item, 'text': '二'}).status_code == 409


def test_purge_tombstone_prevents_late_message_recreation(setup):
    store, conversation, _ = setup
    store.visibility(conversation, 'deleted')
    calls = []
    store.purge(conversation, lambda thread, tasks: calls.append((thread, tasks)))
    assert calls == [(conversation, [])]
    with pytest.raises(Conflict):
        store.publish(conversation, 'late-title', '迟到的输出')
    with pytest.raises(Conflict):
        store.submit(conversation, command())
    assert store.list() == []


def test_snapshot_hides_private_handoff_and_keeps_usage_separate(setup):
    store, conversation, _ = setup
    item = command('私有原始需求')
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])
    store.work_state(task['task_id'], 'executing', usage={'calls': 7, 'total_tokens': 2000})
    snapshot = store.snapshot(conversation)
    assert 'public_conversation' not in snapshot['work']
    assert snapshot['work']['input_id'] == item['id']
    assert snapshot['usage']['work']['calls'] == 7
    assert snapshot['usage']['chat']['calls'] == 0


def test_complete_but_untrusted_tool_identity_stops_without_execution(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    model = FakeModel([answer(tool_calls=[{'name': 'prepare_work', 'args': {}, 'id': ''}])])
    executed = []
    bot = Chatbot(store, lambda *_: executed.append(True), factory=lambda _: model)
    with pytest.raises(ValueError, match='identity'):
        bot.run(conversation, item)
    assert not executed


def test_progress_keeps_selected_model_thinking(setup):
    store, conversation, _ = setup
    selected = []
    model = FakeModel([answer('当前调查结束')])

    def factory(settings):
        selected.append(settings)
        return model

    Chatbot(store, lambda *_: None, factory=factory).progress(conversation, {})
    assert selected[0]['reasoning_effort'] == 'high'


def test_pause_before_owner_registration_sends_no_model_request(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'pause'})
    model = FakeModel([answer('must not send')])
    scheduler = Scheduler(store, model_factory=lambda _: model)
    scheduler._chat(conversation, item)
    assert not model.requests
    assert store.snapshot(conversation)['chat_status'] == 'paused'
    assert store.usage(conversation)['chat']['calls'] == 0
    scheduler.close()


def test_pause_endpoint_does_not_wait_for_container_termination(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    store.create_work(conversation, item['id'])
    entered, release = threading.Event(), threading.Event()

    def terminate(*args):
        entered.set()
        release.wait(5)

    scheduler = Scheduler(store, pause_callback=terminate)
    try:
        started = time.monotonic()
        scheduler.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'pause'})
        assert time.monotonic() - started < 1
        assert entered.wait(1)
        assert store.snapshot(conversation)['work']['status'] == 'pausing'
    finally:
        release.set()
        scheduler.close()


def test_duplicate_service_owner_cannot_reconcile_live_state(setup):
    store, _, _ = setup
    from agent.application.lease import ServiceLease

    first, second = ServiceLease(store.root), ServiceLease(store.root)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match='已有运行'):
            second.acquire()
    finally:
        first.close()
    second.acquire()
    second.close()




def test_display_failure_does_not_cancel_saved_chat_response(setup, monkeypatch):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    original = store.record_event

    def record(thread, kind, payload):
        if kind == 'activity':
            raise OSError('display unavailable')
        return original(thread, kind, payload)

    monkeypatch.setattr(store, 'record_event', record)
    model = FakeModel([answer('仍然完成')])
    Chatbot(store, lambda *_: None, factory=lambda _: model).run(conversation, item)
    assert store.snapshot(conversation)['messages'][-1]['content'] == '仍然完成'
    assert store.usage(conversation)['chat']['unknown_calls'] == 0


def test_work_fact_window_ignores_chat_noise_and_includes_latest_result(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    for index in range(60):
        store.record_event(conversation, 'activity', {'id': str(index), 'detail': 'display'})
    store.work_state(task, 'verified_success', reason='真实验收通过')
    facts = Chatbot(store, lambda *_: None).work_facts(conversation, limit=3)
    assert facts['events'][-1]['payload']['status'] == 'verified_success'
    assert all(row['kind'] in {'work_state', 'work_fact'} for row in facts['events'])


def test_title_uses_injected_fake_and_publishes_once(setup):
    store, conversation, _ = setup
    title_model = FakeModel([answer('短标题')])
    scheduler = Scheduler(store, title_factory=lambda: title_model)
    item = command('第一条公开消息')
    scheduler.submit(conversation, item)
    scheduler.submit(conversation, item)
    until = time.monotonic() + 2
    while store.snapshot(conversation)['title']['status'] != 'complete' and time.monotonic() < until:
        time.sleep(0.01)
    assert store.snapshot(conversation)['title']['title'] == '短标题'
    assert len(title_model.requests) == 1
    assert store.usage(conversation)['title']['calls'] == 1
    scheduler.close()


def test_chat_compaction_preserves_originals_and_is_separate_from_work(setup, monkeypatch):
    store, conversation, _ = setup
    from agent.model import compaction

    for index in range(8):
        raw = message_to_dict((HumanMessage if index % 2 == 0 else AIMessage)(
            content='旧记录-' + str(index) + ':' + ('原始内容' * 6000)))
        store.model_message(conversation, 'chat', 'old-' + str(index), raw)
    original = store.model_history(conversation)
    item = command('接着回答最新问题')
    store.submit(conversation, item)
    monkeypatch.setattr(compaction, 'limits', lambda output: (40000, 100))
    model = FakeModel([answer('## 任务要求\n保留原始需求\n## 已完成\n已读取记录\n'
                              '## 未完成\n最新问题\n## 正在做\n继续对话'), answer('已回答')])
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model)
    bot.run(conversation, item)
    assert store.model_history(conversation)[:len(original)] == original
    assert store.context(conversation)['context_view']['generation'] == 1
    assert store.work_for_conversation(conversation) == []
    assert store.usage(conversation)['chat']['calls'] == 2
    assert any('checkpoint condenses EARLIER' in str(message.content)
               for message in model.requests[-1])


def test_chat_compaction_unknown_is_not_automatically_repeated(setup, monkeypatch):
    store, conversation, _ = setup
    from agent.model import compaction

    for index in range(6):
        store.model_message(conversation, 'chat', 'old-' + str(index),
                            message_to_dict(HumanMessage(content='长记录' * 10000)))
    monkeypatch.setattr(compaction, 'limits', lambda output: (20000, 100))
    item = command('最新请求')
    store.submit(conversation, item)
    model = FakeModel([TimeoutError('unknown')])
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model)
    with pytest.raises(TimeoutError):
        bot.run(conversation, item)
    assert store.context(conversation)['request_pending']
    next_item = command('继续')
    store.submit(conversation, next_item)
    with pytest.raises(Exception, match='not retried'):
        bot.run(conversation, next_item)
    assert len(model.requests) == 1


def test_chat_prefix_change_is_rejected_but_explicit_settings_revision_is_allowed(setup):
    store, conversation, repo = setup
    item = command('第一问')
    store.submit(conversation, item)
    model = FakeModel([answer('第一答'), answer('第二答')])
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model)
    bot.run(conversation, item)
    second = command('第二问')
    second['runtime_settings'] = {'model_name': 'deepseek-flash', 'reasoning_effort': 'low', 'online': True}
    store.submit(conversation, second)
    bot.run(conversation, second)
    assert store.context(conversation)['configuration_revision'] == 1
    assert store.context(conversation)['prior_views']
    with store.connection() as db:
        row = db.execute("SELECT id,data FROM app_model_messages WHERE conversation=? AND lane='chat' ORDER BY seq LIMIT 1",
                         (conversation,)).fetchone()
        raw = json.loads(row['data'])
        raw['data']['content'] = 'tampered'
        db.execute('UPDATE app_model_messages SET data=? WHERE id=?', (json.dumps(raw), row['id']))
    third = command('第三问')
    store.submit(conversation, third)
    with pytest.raises(Exception, match='Original transcript changed'):
        bot.run(conversation, third)
    assert len(model.requests) == 2


def test_conversation_usage_keeps_previous_work_cost(setup):
    store, conversation, _ = setup
    first = command()
    store.submit(conversation, first)
    one = store.create_work(conversation, first['id'])
    store.work_state(one['task_id'], 'verified_success', usage={'calls': 3, 'total_tokens': 100})
    second = command('另一个任务')
    store.submit(conversation, second)
    two = store.create_work(conversation, second['id'])
    store.work_state(two['task_id'], 'executing', usage={'calls': 2, 'total_tokens': 200})
    assert store.snapshot(conversation)['usage']['work']['calls'] == 5
    assert store.snapshot(conversation)['usage']['work']['total_tokens'] == 300


def test_chat_record_reader_allows_committed_output_but_not_private_files(setup):
    store, conversation, _ = setup
    from agent.application.records import read_work_record
    from agent.storage.history import LocalHistory

    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    history = LocalHistory(store.root / 'tasks' / task)
    try:
        with history.lease():
            history.record({'history': []}, 'fixture', {})
            output = history.root / 'plans' / 'p' / 'p-s' / 'stdout.log'
            output.parent.mkdir(parents=True)
            output.write_text('first line\nsecond line', encoding='utf-8')
            history.register_derived(output, 'shell', '')
        result = read_work_record(store, conversation, 'plans/p/p-s/stdout.log', 0, 10)
        assert result['text'] == 'first line' and result['next_offset'] == 10
        for forbidden in ('../another/stdout.log', 'requests/call-0001/response.json',
                          'frozen-acceptance.json', '/private/stdout.log'):
            with pytest.raises(ValueError):
                read_work_record(store, conversation, forbidden, 0, 100)
        output.write_text('changed', encoding='utf-8')
        with pytest.raises(ValueError, match='原件已变化'):
            read_work_record(store, conversation, 'plans/p/p-s/stdout.log', 0, 100)
    finally:
        history.close()




def test_automatic_progress_stops_after_three_requests_without_controlling_work(setup):
    store, conversation, _ = setup
    from agent.application.chat import ChatStopped

    model = FakeModel([answer(tool_calls=[{'name': 'get_work_status', 'args': {},
                                          'id': 'status-' + str(index)}]) for index in range(4)])
    bot = Chatbot(store, lambda *_: pytest.fail('Report controlled work'), factory=lambda _: model)
    with pytest.raises(ChatStopped, match='three-request'):
        bot.progress(conversation, {})
    assert len(model.requests) == 3
    assert store.model_history(conversation, 'chat') == []
    assert store.usage(conversation)['progress']['calls'] == 3


def test_work_events_do_not_rewrite_the_frozen_handoff(setup):
    store, conversation, _ = setup
    item = command('原始需求' * 30000)
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    with store.connection() as db:
        original = db.execute('SELECT data FROM app_work WHERE id=?', (task,)).fetchone()[0]
    before = sum(path.stat().st_size for path in store.root.glob('application.sqlite3*'))
    for index in range(80):
        store.work_state(task, 'executing', reason='已记录步骤 ' + str(index), usage={'calls': index})
    with store.connection() as db:
        assert db.execute('SELECT data FROM app_work WHERE id=?', (task,)).fetchone()[0] == original
        assert db.execute('SELECT length(data) FROM app_work_state WHERE id=?', (task,)).fetchone()[0] < 2000
    after = sum(path.stat().st_size for path in store.root.glob('application.sqlite3*'))
    assert after - before < 300_000


def test_failed_container_stop_is_not_reported_as_delivered(setup):
    store, conversation, _ = setup
    scheduler = Scheduler(store)
    user = command()
    store.submit(conversation, user)
    task = store.create_work(conversation, user['id'])['task_id']
    item = {'id': str(uuid.uuid4()), 'kind': 'pause'}
    scheduler._control(conversation, task, item,
                       lambda *_: {'termination': [{'container': 'x', 'stopped': False}]})
    assert store.events(conversation)['events'][-1]['kind'] == 'control_unconfirmed'
    store.work_state(task, 'paused')
    assert store.work(task)['status'] == 'execution_unknown'
    store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'pause'})
    scheduler._control(conversation, task, {'id': str(uuid.uuid4()), 'kind': 'pause'},
                       lambda *_: {'termination': [{'container': 'x', 'stopped': True}]})
    store.work_state(task, 'paused')
    assert store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume'})['received']
    assert store.work(task)['status'] == 'execution_unknown'
    scheduler.close()


def test_work_binding_uses_the_application_root_not_global_preferences(setup, monkeypatch, tmp_path):
    store, conversation, repo = setup
    explicit = {'backend': 'docker', 'image': 'sha256:' + 'a' * 64,
                'runtime': {'python_executable': '/own/python', 'conda_environment': ''}}
    (store.root / 'project-environments.json').write_text(
        json.dumps({str(repo.resolve()): explicit}), encoding='utf-8')
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(tmp_path / 'another-service'))
    item = command()
    store.submit(conversation, item)
    assert store.create_work(conversation, item['id'])['execution_binding'] == explicit


def test_progress_pages_do_not_skip_facts_beyond_fifty(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    sequences = [store.work_state(task, 'executing', reason=str(index)) for index in range(65)]
    store.progress_pending(task, sequences[-1])
    scheduler = Scheduler(store)
    batches = []
    scheduler.chatbot.progress = lambda _conversation, facts: (
        batches.append(facts) or {'output': '已读取真实工作事实'})
    first = store.reserve_progress(now=100)
    scheduler._report(first)
    second = store.reserve_progress(now=131)
    assert second['reported_seq'] == sequences[49]
    scheduler._report(second)
    assert [event['seq'] for batch in batches for event in batch['events']] == sequences
    assert store.reserve_progress(now=162) is None
    scheduler.close()


def test_control_command_is_dispatched_once_while_unacknowledged(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    started, release = threading.Event(), threading.Event()
    calls = []

    def pause(task_id, command_id):
        calls.append((task_id, command_id))
        started.set()
        assert release.wait(2)
        return {'termination': []}

    scheduler = Scheduler(store, pause_callback=pause)
    request = {'id': str(uuid.uuid4()), 'kind': 'pause'}
    try:
        scheduler.submit(conversation, request)
        assert started.wait(2)
        scheduler.submit(conversation, request)
        release.set()
        until = time.monotonic() + 2
        while scheduler.controls_inflight and time.monotonic() < until:
            time.sleep(0.01)
        scheduler.submit(conversation, request)
        assert calls == [(task, request['id'])]
        assert not store.reserve_control(conversation, task, request)
    finally:
        release.set()
        scheduler.close()


def test_progress_for_a_previous_work_is_cancelled_before_request(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    seq = store.work_state(task, 'verified_success')
    store.progress_pending(task, seq)
    pending = store.reserve_progress(now=100)
    second = command('开始下一个任务')
    store.submit(conversation, second)
    store.create_work(conversation, second['id'])
    scheduler = Scheduler(store)
    scheduler.chatbot.progress = lambda *_: pytest.fail('Old work requested a new summary')
    scheduler._report(pending)
    assert store.reserve_progress(now=131) is None
    scheduler.close()


@pytest.mark.parametrize('status', ['verified_success', 'existing_complete', 'execution_unknown',
                                   'model_request_unknown', 'runtime_error', 'pausing'])
def test_finished_or_uncertain_work_does_not_accept_pending_control(setup, status):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    store.work_state(task, status)
    with pytest.raises(Conflict):
        store.submit(conversation, command('只针对原任务补充', 'supplement'))
    receipt = store.submit(conversation, command(kind='resume'))
    assert receipt['received']
    assert store.work(task)['status'] == status
    if status == 'pausing':
        assert receipt['applied'] and not store.commands(task=task)
    else:
        assert store.commands(task=task)[0]['kind'] == 'resume'


@pytest.mark.parametrize('restart', [False, True])
def test_control_that_loses_its_task_is_finished_without_replay(setup, restart):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    store.command_state(conversation, item['id'], 'applied')
    store.work_state(task, 'paused')
    control = command('保留原文', 'supplement')
    store.submit(conversation, control)
    store.work_state(task, 'verified_success')
    scheduler = Scheduler(store, work_factory=lambda *_: pytest.fail('Completed work was replayed'))
    try:
        if restart:
            store.recover_interrupted()
        else:
            scheduler.tick()
        assert store.commands(task=task, states=('failed',))[0]['id'] == control['id']
        assert scheduler.status()['pending'] == 0
    finally:
        scheduler.close()


def test_handoff_waiting_for_dispatch_keeps_service_busy(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    store.create_work(conversation, item['id'])
    store.command_state(conversation, item['id'], 'applied')
    scheduler = Scheduler(store)
    try:
        assert scheduler.status()['running_work'] == 0
        assert scheduler.status()['queued_work'] == 1
        assert scheduler.status()['pending'] == 1
    finally:
        scheduler.close()


def test_already_consumed_supplement_does_not_resume_paused_work(setup, monkeypatch):
    from support import fixture_delivery

    store, conversation, _ = setup
    first = command('Keep the application behavior')
    store.submit(conversation, first)
    work = store.create_work(conversation, first['id'],
                             work_options={'confirmed_requirements': fixture_delivery(first['text'])})
    store.command_state(conversation, first['id'], 'applied')
    update = command('Retain all required inputs', 'supplement')
    store.submit(conversation, update)
    store.work_state(work['task_id'], 'paused', applied_inputs=[update['id']])
    store.record_event(conversation, 'requirement_discussion_started', {'id': update['id']})
    scheduler = Scheduler(store)
    monkeypatch.setattr(scheduler, '_schedule_work', lambda *_: pytest.fail('Consumed input resumed work'))
    try:
        scheduler.tick()
        assert store.work(work['task_id'])['status'] == 'paused'
    finally:
        scheduler.close()


def test_output_limit_requires_explicit_resume_before_supplement(setup):
    store, conversation, _ = setup
    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    store.work_state(task, 'output_limit_reached')
    with pytest.raises(Conflict):
        store.submit(conversation, command('继续这条补充', 'supplement'))
    assert store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume'})['received']


def test_new_conversation_safety_and_title_share_one_database_without_legacy_mirror(setup):
    store, conversation, _ = setup
    from agent.storage.permissions import Permissions
    from agent.storage.read_sources import ReadSources
    from agent.application import titles

    Permissions(store.root).request(conversation, 'ask')
    ReadSources(store.root).set(conversation, [])
    assert titles.reserve(store, conversation, command('公开需求'))
    titles.generate(store, conversation, lambda: FakeModel([answer('新标题')]))
    assert not (store.root / 'conversations.sqlite3').exists()
    with store.connection() as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'app_titles', 'permissions', 'read_sources'} <= tables
    assert not {'sessions', 'inputs', 'presentation', 'conversation_titles'} & tables
    assert store.get(conversation)['title']['title'] == '新标题'


def test_new_title_is_bounded_once_and_unknown_is_not_replayed(setup):
    store, conversation, _ = setup
    from agent.application import titles

    text = '只使用首条公开消息' * 1000
    assert titles.reserve(store, conversation, command(text))
    assert not titles.reserve(store, conversation, command('不能覆盖首条要求'))
    model = FakeModel([TimeoutError('unknown')])
    titles.generate(store, conversation, lambda: model)
    assert model.requests[0][1].content == text[:4000]
    assert store.get(conversation)['title']['status'] == 'unknown'
    assert store.usage(conversation)['title']['unknown_calls'] == 1
    titles.generate(store, conversation, lambda: pytest.fail('Unknown title repeated'))
    store.recover_interrupted()
    titles.generate(store, conversation, lambda: pytest.fail('Restart repeated title'))
    assert len(model.requests) == 1


def test_new_title_missing_key_is_not_a_request_and_restart_settles_pending(setup):
    store, conversation, _ = setup
    from agent.application import titles

    titles.reserve(store, conversation, command())
    titles.generate(store, conversation, lambda: (_ for _ in ()).throw(ValueError('missing key')))
    assert store.get(conversation)['title']['status'] == 'failed'
    assert store.usage(conversation)['title']['calls'] == 0
    other = store.create()['thread_id']
    titles.reserve(store, other, command())
    store.recover_interrupted()
    assert store.get(other)['title']['status'] == 'failed'
    titles.generate(store, other, lambda: pytest.fail('Restart repeated a reserved title'))


def test_new_title_late_result_cannot_resurrect_a_purged_conversation(setup):
    store, conversation, _ = setup
    from agent.application import titles

    titles.reserve(store, conversation, command())
    class LateModel:
        def invoke(self, _messages):
            store.visibility(conversation, 'deleted')
            scheduler = Scheduler(store)
            try:
                scheduler.purge(conversation)
            finally:
                scheduler.close()
            return answer('迟到标题')

    titles.generate(store, conversation, LateModel)
    assert store.get(conversation)['visibility'] == 'purged'
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM app_titles').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM app_calls').fetchone()[0] == 0


@pytest.mark.parametrize('pending', ['request_pending', 'rag_running', 'rag_request_pending'])
def test_new_purge_checks_private_unknown_operations_and_keeps_user_workspace(setup, monkeypatch, pending):
    store, conversation, repo = setup
    from agent.storage.history import LocalHistory

    item = command()
    store.submit(conversation, item)
    task = store.create_work(conversation, item['id'])['task_id']
    store.command_state(conversation, item['id'], 'applied')
    store.work_state(task, 'paused')
    history = LocalHistory(store.root / 'tasks' / task)
    with history.lease():
        history.record({'history': [], pending: {'call': 1}}, 'unknown', {})
    history.close()
    (repo / 'keep.txt').write_text('user project')
    store.chat_state(conversation, 'chat_complete')
    store.visibility(conversation, 'deleted')
    scheduler = Scheduler(store)
    try:
        with pytest.raises(Conflict, match='结果不明'):
            scheduler.purge(conversation)
        assert store.get(conversation)['visibility'] == 'deleted'
        assert history.root.exists()
        assert (repo / 'keep.txt').read_text() == 'user project'
    finally:
        scheduler.close()


@pytest.mark.parametrize('approval', ['approve', 'reject'])
def test_external_approval_waits_then_appends_one_bound_result_before_next_message(setup, approval):
    store, conversation, _ = setup
    from agent.execution.desktop import DesktopStore
    from agent.storage.permissions import Permissions

    Permissions(store.root).request(conversation, 'ask')
    target = store.root.parent / 'outside-project.txt'
    first = command('请创建这个外部文件')
    store.submit(conversation, first)
    model = FakeModel([answer(tool_calls=[{
        'name': 'request_file_operation', 'id': 'create-file',
        'args': {'operation': 'create', 'path': str(target), 'destination': '', 'content': 'approved text'},
    }])])
    bot = Chatbot(store, lambda *_: pytest.fail('Approval entered work'), factory=lambda _: model)
    result = bot.run(conversation, first)
    assert result['waiting_approval'] and len(model.requests) == 1
    assert not target.exists()
    original = store.model_history(conversation)
    desktop = DesktopStore(store.root)
    request = desktop.list(conversation)[0]
    operation = getattr(desktop, approval)
    final = operation(conversation, request['id'], request['target'])
    assert final['status'] == ('completed' if approval == 'approve' else 'rejected')
    assert operation(conversation, request['id'], request['target']) == final
    assert len(model.requests) == 1  # 用户点击批准只执行目标，不触发模型轮询。
    if approval == 'approve':
        assert target.read_text() == 'approved text'
    followup = FakeModel([answer('已核对操作结果'), answer('仍沿用之前的结果')])
    bot.factory = lambda _: followup
    for text in ('继续', '再说明一下'):
        incoming = command(text)
        store.submit(conversation, incoming)
        bot.run(conversation, incoming)
    history = store.model_history(conversation)
    facts = [(identity, raw) for identity, raw in history if identity.startswith('external-result:')]
    assert len(facts) == 1
    assert store.model_history(conversation)[:len(original)] == original
    fact = json.loads(facts[0][1]['data']['content'])['external_results'][0]
    assert (fact['id'], fact['target'], fact['status']) == (
        request['id'], request['target'], final['status'])
    assert 'content' not in fact and 'diff' not in fact
    messages = followup.requests[0]
    fact_index = next(index for index, message in enumerate(messages)
                      if 'saved_application_result' in str(message.content))
    assert fact_index < len(messages) - 1
    assert messages[-1].content == '继续'
    assert not store.work_for_conversation(conversation)


def test_external_result_requires_original_tool_target_and_conversation(setup):
    store, conversation, _ = setup
    from agent.execution.desktop import DesktopStore

    desktop = DesktopStore(store.root)
    local = desktop.put(conversation, {'kind': 'external', 'target': 'actual-target',
                                      'path': '/example', 'application': {'id': 'app'}}, 'opened')
    other = store.create()['thread_id']
    unrelated = desktop.put(other, {'kind': 'external', 'target': 'unrelated-target',
                                    'path': '/unrelated', 'application': {'id': 'app'}}, 'opened')
    for index, row in enumerate((local, unrelated)):
        binding = 'changed-target' if row is local else row['target']
        raw = message_to_dict(ToolMessage(content=json.dumps({
            'id': row['id'], 'target': binding, 'status': 'awaiting_approval',
        }), tool_call_id=str(index)))
        store.model_message(conversation, 'chat', 'fixture-tool-' + str(index), raw)
    bot = Chatbot(store, lambda *_: None)
    bot._append_external_results(conversation)
    assert not any(identity.startswith('external-result:')
                   for identity, _ in store.model_history(conversation))



def test_default_plan_question_waits_before_preparing_confirmable_work(setup):
    """讨论和方案都不是执行权限；确认必须绑定保存的完整需求。"""
    store, conversation, _ = setup
    first = command('Create a runnable Python application; preserve the interface')
    confirmation = command('Prepare that plan')
    model = FakeModel([
        answer('I can implement the application in a new child project. Shall I prepare its delivery plan?'),
        answer(tool_calls=[{'name': 'prepare_work', 'args': delivery_args(first, 'application', True), 'id': 'proposal'}]),
    ])
    bot = Chatbot(store, lambda *_: pytest.fail('Chat model started work'), factory=lambda _: model)
    store.submit(conversation, first)
    bot.run(conversation, first)
    assert store.snapshot(conversation)['work'] is None
    assert not (store.root / 'tasks').exists()
    rules = model.requests[0][0].content
    assert 'A broad goal alone is not permission to create or change files' in rules
    assert 'human confirmation control' in rules
    store.submit(conversation, confirmation)
    bot.run(conversation, confirmation)
    assert store.snapshot(conversation)['work'] is None
    proposals = [m for m in store.snapshot(conversation)['messages'] if m['source'] == 'work_proposal']
    assert len(proposals) == 1 and first['text'] in proposals[0]['content']
    assert len(model.requests) == 2


def test_live_work_activity_reads_deltas_without_copying_or_touching_chat(setup, monkeypatch):
    """实时展示读取已有原件，不复制请求历史、不申请执行租约。"""
    from agent.storage.history import LocalHistory
    from agent.storage.live_reasoning import LiveReasoning

    store, conversation, _ = setup
    incoming = command('按方案开始')
    store.submit(conversation, incoming)
    task = store.create_work(conversation, incoming['id'])['task_id']
    history = LocalHistory(store.root / 'tasks' / task)
    with history.lease():
        history.record({'history': [], 'budget': {'used_steps': 1}},
                       'model_request_started', {'call': 1})
    live = LiveReasoning(history.root, 1, task + '-reply-private')
    live.append('已读取第一段')

    class NoDispatch:
        def submit(self, *_):
            pytest.fail('Activity read dispatched work')

    client = TestClient(Starlette(routes=build_routes(store, NoDispatch())))
    endpoint = f'/conversations/{conversation}/diagnostics'

    def read(after=None):
        response = client.get(endpoint, params={'after': json.dumps(after or {})})
        assert response.status_code == 200
        return response.json()

    first = read()
    assert first['live'][0]['detail'] == '已读取第一段'
    assert first['live'][0]['after_id'] == incoming['id']
    live.append('，正在继续核对')
    live.flush()
    second = read(first['cursors'])
    assert second['live'][0]['detail'] == '，正在继续核对'
    assert second['live'][0]['append'] is True
    assert second['live'][0]['id'] == first['live'][0]['id']
    assert not read(second['cursors'])['live']
    live.finish('interrupted')
    store.work_state(task, 'paused')
    final = read(second['cursors'])
    assert final['live'][0]['status'] == 'interrupted'
    assert final['live'][0]['detail'] == '已读取第一段，正在继续核对'

    before = {str(p): p.read_bytes() for p in history.root.rglob('*') if p.is_file()}
    with monkeypatch.context() as isolated:
        isolated.setattr(store, 'snapshot', lambda *_: pytest.fail('Activity rebuilt full chat'))
        assert not read(final['cursors'])['items']
    assert read()['live'][0]['detail'] == final['live'][0]['detail']
    assert before == {str(p): p.read_bytes() for p in history.root.rglob('*') if p.is_file()}
    assert not store.model_history(conversation)
    assert store.snapshot(conversation)['activities'] == []
    assert store.usage(conversation)['chat']['calls'] == 0
    history.close()



def test_project_context_reads_can_be_batched_without_losing_tool_results(setup):
    store, conversation, repo = setup
    incoming = command('查看当前目录和工作状态')
    store.submit(conversation, incoming)
    model = FakeModel([
        answer(tool_calls=[
            {'name': 'get_read_sources', 'args': {}, 'id': 'location'},
            {'name': 'get_work_status', 'args': {}, 'id': 'status'},
        ]),
        answer('项目位置仍待确认。'),
    ])
    bot = Chatbot(store, lambda *a, **k: pytest.fail('Read dispatched work'), factory=lambda _: model)
    bot.run(conversation, incoming)
    results = {message.tool_call_id: json.loads(message.content)
               for message in model.requests[1] if isinstance(message, ToolMessage)}
    assert set(results) == {'location', 'status'}
    assert results['location']['workspace'] == str(repo)
    assert results['location']['current_project'] is None and results['status'] is None
