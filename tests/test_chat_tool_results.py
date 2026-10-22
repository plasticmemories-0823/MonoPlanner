"""当前 Chatbot 工具结果的原件、单次结构去重和故障恢复；不调用外部服务。"""

from copy import deepcopy
import json
from pathlib import Path
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.application.chat import Chatbot
from agent.application.scheduler import Scheduler
from agent.application.store import ApplicationStore, Conflict
from agent.model.errors import ModelInterrupted, ModelRequestNotSent
from agent.task.control import Signal, using
from tests.fixtures.models import ScriptedModel, text_reply


def incoming(text='读取项目资料'):
    return {'id': str(uuid.uuid4()), 'kind': 'message', 'text': text}


def reply(*calls):
    return AIMessage(content='', tool_calls=list(calls),
                     usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2})


def call(name, identity='tool-1', **args):
    return {'name': name, 'id': identity, 'args': args}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(store.root))
    conversation = store.create(repo=str(repo))['thread_id']
    return store, conversation, repo


def originals(store, conversation):
    return sorted((store.root / 'application-records' / conversation).glob('*/tool-result.json'))


def tools(store, conversation):
    return [raw['data'] for _, raw in store.model_history(conversation) if raw['type'] == 'tool']


def test_invalid_requirement_code_is_paired_then_corrected_without_replay(setup):
    from tests.test_requirement_delivery import target

    store, conversation, _ = setup
    item = incoming('Generate runnable code')
    store.submit(conversation, item)
    candidate = target()
    candidate['quotes'][0]['source_id'] = item['id']
    arguments = dict(project_path='.', create=False, request_message_ids=[item['id']],
                     targets=[candidate], withdrawals=[])
    invalid = deepcopy(arguments)
    invalid['targets'][0]['imports'] = 'No imports；only standard library'
    model = ScriptedModel([
        reply(call('prepare_work', 'invalid', **invalid)),
        reply(call('prepare_work', 'corrected', **arguments)),
        text_reply(),
    ])
    bot = Chatbot(store, lambda *_: pytest.fail('A proposal cannot start work'),
                  factory=lambda _: model)
    result = bot.run(conversation, item)
    assert result['waiting_approval'] and len(model.calls) == 2
    saved = tools(store, conversation)
    assert [row['tool_call_id'] for row in saved] == ['invalid', 'corrected']
    rejected = json.loads(saved[0]['content'])
    assert rejected['actions_executed'] is False
    assert 'targets[0]' in rejected['error'] and 'imports' in rejected['error']
    assert store.usage(conversation)['chat']['unknown_calls'] == 0
    following = incoming('Explain the proposal')
    store.submit(conversation, following)
    bot.run(conversation, following)
    assert len(model.calls) == 3 and len(originals(store, conversation)) == 2


@pytest.mark.parametrize('failure', [RuntimeError, ValueError, TypeError])
def test_internal_tool_failure_is_not_a_model_unknown_or_unexecuted_result(setup, failure):
    store, conversation, _ = setup
    model = ScriptedModel([reply(call('request_file_operation'))])
    scheduler = Scheduler(store, model_factory=lambda _: model)
    scheduler.chatbot.invoke_tool = lambda *_: (_ for _ in ()).throw(failure('internal'))
    item = incoming()
    store.submit(conversation, item)
    try:
        scheduler._chat(conversation, item)
        assert store.get(conversation)['chat_status'] == 'tool_internal_error'
        assert store.usage(conversation)['chat']['unknown_calls'] == 0
        assert not tools(store, conversation) and len(model.calls) == 1
        scheduler.tick()
        assert len(model.calls) == 1
    finally:
        scheduler.close()


def test_chat_archives_before_normalizing_and_keeps_calls_separate(setup, monkeypatch):
    from agent.application import chat

    store, conversation, repo = setup
    raw = {'path': str(repo), 'entries': [
        {'path': str(repo / 'a.txt'), 'name': 'old', 'directory': False},
        {'path': str(repo / 'b.txt'), 'name': 'b', 'directory': False},
        {'path': str(repo / 'a.txt'), 'name': 'new', 'directory': False},
    ], 'next_offset': 200}
    original = deepcopy(raw)
    model = ScriptedModel([
        reply(call('list_files', '../../model-id'), call('list_files', 'second')),
        text_reply(), reply(call('list_files', 'later-call')), text_reply(),
    ])
    normalized = chat.model_result
    observed = []

    def checked(name, value, **options):
        archived = [json.loads(path.read_text('utf-8')) for path in originals(store, conversation)]
        assert len(archived) == len(observed) + 1
        assert all(item['result'] == value for item in archived)
        observed.append((name, deepcopy(value), options))
        return normalized(name, value, **options)

    monkeypatch.setattr(chat, 'model_result', checked)
    bot = Chatbot(store, lambda *_args, **_options: pytest.fail('Unexpected handoff'),
                  factory=lambda _: model, invoke_tool=lambda *_: raw)
    first = incoming()
    store.submit(conversation, first)
    bot.run(conversation, first)
    saved = store.model_history(conversation)
    tool_rows = tools(store, conversation)
    assert [row['tool_call_id'] for row in tool_rows] == ['../../model-id', 'second']
    expected = {**raw, 'entries': [raw['entries'][1], raw['entries'][2]]}
    assert [json.loads(row['content']) for row in tool_rows] == [expected, expected]
    assert [item.content for item in model.calls[1] if isinstance(item, ToolMessage)] == [
        tool_rows[0]['content'], tool_rows[1]['content'],
    ]
    later = incoming('再看一次')
    store.submit(conversation, later)
    bot.run(conversation, later)
    assert store.model_history(conversation)[:len(saved)] == saved
    assert json.loads(tools(store, conversation)[-1]['content']) == expected
    assert len(originals(store, conversation)) == 3
    assert raw == original and all(item[1] == original for item in observed)
    assert all(item[2] == {'project': str(repo)} for item in observed)
    assert all(path.parent.parent == store.root / 'application-records' / conversation
               for path in originals(store, conversation))


def test_chat_search_keeps_audit_local_and_sends_shared_sources(setup, monkeypatch):
    import hashlib
    from agent.rag import integration

    store, conversation, repo = setup
    text = '中文 needle needle\r\n'
    (repo / 'example.py').write_bytes(text.encode('utf-8'))
    model = ScriptedModel([
        reply(call('retrieve_evidence', query='needle', path='.', mode='exact')),
        text_reply(),
    ])
    bot = Chatbot(store, lambda *_: pytest.fail('Unexpected handoff'), factory=lambda _: model)
    item = incoming()
    store.submit(conversation, item)
    bot.run(conversation, item)
    tool = next(message for message in model.calls[1] if isinstance(message, ToolMessage))
    body = json.loads(tool.content)
    raw = json.loads(originals(store, conversation)[0].read_text('utf-8'))['result']
    assert tool.tool_call_id == 'tool-1' and tool.name == 'retrieve_evidence'
    assert len(body['hits']) == len(raw['hits']) == 2
    assert len(body['sources']) == len(body['results']) == 1
    assert body['results'][0]['text'] == raw['results'][0]['text']
    assert body['sources'][0]['sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert tool.content.count('example.py') == tool.content.count(raw['results'][0]['version']) == 1
    assert 'scope' in raw and 'scope' not in body


@pytest.mark.parametrize('name,result', [
    ('read_file', {'path': 'a', 'text': 'same\nsame\n', 'visible_ranges': [[0, 2]]}),
    ('web_fetch', {'url': 'https://example.test', 'text': 'same\nsame'}),
    ('list_applications', {'items': [{'id': 'x'}, {'id': 'x'}]}),
    ('open_file', {'id': 'approval', 'target': 'file', 'status': 'awaiting_approval'}),
    ('launch_application', {'status': 'opened', 'pid': 123, 'items': [{'id': 'x'}, {'id': 'x'}]}),
    ('request_file_operation', {'status': 'completed', 'actions_executed': True,
                                'items': [{'id': 'x'}, {'id': 'x'}]}),
])
def test_noncollection_execution_and_approval_results_are_unchanged(setup, name, result):
    store, conversation, _ = setup
    model = ScriptedModel([reply(call(name)), text_reply()])
    bot = Chatbot(store, lambda *_args, **_options: pytest.fail('Unexpected handoff'),
                  factory=lambda _: model, invoke_tool=lambda *_: result)
    item = incoming()
    store.submit(conversation, item)
    state = bot.run(conversation, item)
    assert json.loads(tools(store, conversation)[0]['content']) == result
    assert json.loads(originals(store, conversation)[0].read_text('utf-8'))['result'] == result
    assert state['waiting_approval'] == (result.get('status') == 'awaiting_approval')
    assert len(model.calls) == (1 if state['waiting_approval'] else 2)


@pytest.mark.parametrize('name', ['prepare_work', 'run_existing', 'get_work_status',
                                  'get_read_sources', 'read_work_events', 'read_work_record'])
def test_builtin_handoff_and_status_tools_also_save_raw_results(setup, monkeypatch, name):
    from agent.application import chat, records

    store, conversation, _ = setup
    task = {'task_id': 'a' * 32, 'status': 'queued', 'repo': 'project', 'project_path': '.'}
    arguments = {'prepare_work': {'project_path': '.', 'create': False},
                 'run_existing': {'entry': 'task.py', 'args': [], 'timeout': 10, 'usage_file': ''},
                 'read_work_events': {'after': 0, 'limit': 50},
                 'read_work_record': {'record': 'record', 'offset': 0, 'limit': 10}}.get(name, {})
    model = ScriptedModel([reply(call(name, **arguments)), text_reply()])
    result = {'items': [{'id': 'same'}, {'id': 'same'}], 'status': 'completed'}
    bot = Chatbot(store, lambda *_args, **_options: task, factory=lambda _: model)
    monkeypatch.setattr(bot, 'work_facts', lambda *_args: {'task': result, 'events': [result, result]})
    monkeypatch.setattr(records, 'read_work_record', lambda *_args, **_options: result)
    observed = []
    normalize = chat.model_result

    def normalized(tool_name, value, **options):
        observed.append(deepcopy(value))
        return normalize(tool_name, value, **options)

    monkeypatch.setattr(chat, 'model_result', normalized)
    item = incoming()
    store.submit(conversation, item)
    bot.run(conversation, item)
    raw = json.loads(originals(store, conversation)[0].read_text('utf-8'))
    assert raw['name'] == name
    assert json.loads(tools(store, conversation)[0]['content']) == raw['result'] == observed[0]


def test_archive_is_immutable_and_checks_space_only_before_first_save(setup, monkeypatch):
    from agent.application import chat

    store, conversation, _ = setup
    bot = Chatbot(store, lambda *_: None)
    checked = []
    monkeypatch.setattr(chat, 'preflight_space', lambda path, size: checked.append((path, size)))
    original_call, result = call('list_files'), {'entries': [{'path': 'a'}]}
    bot._archive_tool_result(conversation, 'a' * 32 + '-0', original_call, result)
    path = originals(store, conversation)[0]
    original = path.read_bytes(), path.stat().st_mtime_ns
    bot._archive_tool_result(conversation, 'a' * 32 + '-0', original_call, deepcopy(result))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert len(checked) == 1 and checked[0][1] == len(original[0])
    with pytest.raises(Conflict, match='not overwritten'):
        bot._archive_tool_result(conversation, 'a' * 32 + '-0', original_call, {'entries': []})
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    path.write_bytes(b'{"partial":')
    with pytest.raises(Conflict, match='not overwritten'):
        bot._archive_tool_result(conversation, 'a' * 32 + '-0', original_call, result)
    assert path.read_bytes() == b'{"partial":'


def test_archive_rejects_a_path_outside_owned_storage(setup):
    store, conversation, _ = setup
    bot = Chatbot(store, lambda *_: None)
    with pytest.raises(ValueError):
        bot._archive_tool_result(conversation, '../../../escaped', call('list_files'), {})
    assert not originals(store, conversation)


@pytest.mark.parametrize('failure', ['archive', 'message'])
def test_result_storage_failure_stops_dispatch_and_restart_does_not_replay(setup, monkeypatch, failure):
    from agent.application import chat

    store, conversation, _ = setup
    model = ScriptedModel([reply(call('request_file_operation')), text_reply()])
    executed = []
    scheduler = Scheduler(store, model_factory=lambda _: model)
    scheduler.chatbot.invoke_tool = lambda *_: executed.append('done') or {
        'status': 'completed', 'actions_executed': True,
    }
    item = incoming('Perform the authorized file operation')
    store.submit(conversation, item)
    store.command_state(conversation, item['id'], 'running')
    if failure == 'archive':
        monkeypatch.setattr(chat, 'preflight_space', lambda *_: (_ for _ in ()).throw(OSError('disk full')))
    else:
        original_message = store.model_message

        def fail_message(conversation, lane, identity, raw):
            if raw['type'] == 'tool':
                raise OSError('database full')
            return original_message(conversation, lane, identity, raw)

        monkeypatch.setattr(store, 'model_message', fail_message)
    try:
        scheduler.chats.add(conversation)
        scheduler._chat(conversation, item)
        assert scheduler.storage_error() and executed == ['done'] and len(model.calls) == 1
        assert not tools(store, conversation)
        if failure == 'message':
            assert json.loads(originals(store, conversation)[0].read_text('utf-8'))['result'] == {
                'status': 'completed', 'actions_executed': True,
            }
        scheduler.tick()
        if failure == 'message':
            with pytest.raises(OSError, match='database full'):
                store.recover_interrupted()
            monkeypatch.setattr(store, 'model_message', original_message)
        store.recover_interrupted()
        assert tools(store, conversation)
        scheduler.tick()
        assert executed == ['done'] and len(model.calls) == 1
    finally:
        scheduler.close()


def test_normalization_failure_preserves_raw_and_never_replays_operation(setup, monkeypatch):
    from agent.application import chat

    store, conversation, _ = setup
    model = ScriptedModel([reply(call('list_files')), text_reply()])
    executed = []
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model,
                  invoke_tool=lambda *_: executed.append('done') or {'entries': []})
    monkeypatch.setattr(chat, 'model_result', lambda *_args, **_options:
                        (_ for _ in ()).throw(ValueError('normalization failed')))
    item = incoming()
    store.submit(conversation, item)
    with pytest.raises(ValueError, match='normalization failed'):
        bot.run(conversation, item)
    assert json.loads(originals(store, conversation)[0].read_text('utf-8'))['result'] == {'entries': []}
    following = incoming('继续')
    store.submit(conversation, following)
    with pytest.raises(ModelRequestNotSent) as stopped:
        bot.run(conversation, following)
    assert stopped.value.status == 'invalid_message_history'
    assert executed == ['done'] and len(model.calls) == 1


def test_pause_archives_completed_and_cancelled_results_without_changing_call_ids(setup):
    store, conversation, _ = setup
    signal = Signal()
    model = ScriptedModel([reply(call('list_files', 'completed'), call('web_search', 'cancelled'))])
    executed = []

    def invoke(_conversation, tool_call, _operation):
        executed.append(tool_call['id'])
        signal.interrupt('pause')
        return {'entries': [{'path': 'a', 'name': 'old'}, {'path': 'a', 'name': 'new'}]}

    bot = Chatbot(store, lambda *_: None, factory=lambda _: model, invoke_tool=invoke)
    item = incoming()
    store.submit(conversation, item)
    with using(signal), pytest.raises(ModelInterrupted):
        bot.run(conversation, item)
    rows = tools(store, conversation)
    assert [row['tool_call_id'] for row in rows] == ['completed', 'cancelled']
    assert executed == ['completed'] and len(model.calls) == 1
    assert len(json.loads(rows[0]['content'])['entries']) == 1
    assert json.loads(rows[1]['content']) == {
        'status': 'cancelled', 'actions_executed': False, 'reason': 'Stopped before tool dispatch',
    }
    raw = [json.loads(path.read_text('utf-8')) for path in originals(store, conversation)]
    assert len(raw[0]['result']['entries']) == 2
    assert raw[1]['result'] == json.loads(rows[1]['content'])


def test_completed_tool_event_save_failure_keeps_actual_result_and_stops(setup, monkeypatch):
    from agent.application import chat

    store, conversation, _ = setup
    model = ScriptedModel([reply(call('request_file_operation', operation='create',
                                     path='a.txt', destination='', content='data')), text_reply()])
    result = {'status': 'completed', 'actions_executed': True}
    executed = []
    monkeypatch.setattr(chat, 'invoke_assistant',
                        lambda *_: executed.append('done') or result)
    original_event = store.record_event

    def fail_completed(conversation, kind, payload):
        if kind == 'chat_tool_finished':
            raise OSError('completion event disk full')
        return original_event(conversation, kind, payload)

    monkeypatch.setattr(store, 'record_event', fail_completed)
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model)
    item = incoming('Perform the authorized file operation')
    store.submit(conversation, item)
    with pytest.raises(OSError, match='completion event disk full'):
        bot.run(conversation, item)
    saved = json.loads(originals(store, conversation)[0].read_text('utf-8'))
    assert saved['result'] == result
    assert all(json.loads(row['content']).get('actions_executed') is not False
               for row in tools(store, conversation))
    assert executed == ['done'] and len(model.calls) == 1


@pytest.mark.parametrize('name,key,items', [
    ('web_search', 'sources', [
        {'url': 'https://example.test/a?x=1', 'title': 'old'},
        {'url': 'https://example.test/a?x=2', 'title': 'other'},
        {'url': 'https://example.test/a?x=1', 'title': 'new'},
    ]),
    ('retrieve_evidence', 'hits', [
        {'path': 'src/a.py', 'line': 3, 'text': 'old'},
        {'path': 'src/a.py', 'line': 4, 'text': 'other'},
        {'path': 'src/a.py', 'line': 3, 'text': 'new'},
    ]),
])
def test_chat_search_collection_normalization_keeps_original_and_order(
    setup, name, key, items, monkeypatch
):
    store, conversation, _ = setup
    raw = {key: items, 'next_offset': 12, 'record': 'unchanged-original-reference'}
    if name == 'retrieve_evidence':
        monkeypatch.setattr('agent.rag.integration.retrieve_chat', lambda *_: raw)
    arguments = {'query': 'new', 'path': '.'} if name == 'retrieve_evidence' else {}
    model = ScriptedModel([reply(call(name, **arguments)), text_reply()])
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model, invoke_tool=lambda *_: raw)
    item = incoming()
    store.submit(conversation, item)
    bot.run(conversation, item)
    visible = json.loads(tools(store, conversation)[0]['content'])
    assert visible == {**raw, key: [items[1], items[2]]}
    assert json.loads(originals(store, conversation)[0].read_text('utf-8'))['result'] == raw


def test_normal_completion_event_is_once_after_raw_and_model_result(setup, monkeypatch):
    from agent.application import chat

    store, conversation, _ = setup
    model = ScriptedModel([reply(call('list_files', path='.', offset=0, limit=10)), text_reply()])
    result = {'entries': [{'path': 'a'}, {'path': 'a'}]}
    monkeypatch.setattr(chat, 'invoke_assistant', lambda *_: result)
    original_event = store.record_event
    completed = []

    def observe(conversation, kind, payload):
        if kind == 'chat_tool_finished':
            assert json.loads(originals(store, conversation)[0].read_text('utf-8'))['result'] == result
            assert json.loads(tools(store, conversation)[0]['content']) == {'entries': [{'path': 'a'}]}
            completed.append(payload)
        return original_event(conversation, kind, payload)

    monkeypatch.setattr(store, 'record_event', observe)
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model)
    item = incoming()
    store.submit(conversation, item)
    bot.run(conversation, item)
    assert len(completed) == 1 and completed[0]['status'] == 'completed'


def test_restart_reconciles_saved_response_before_sql_without_an_api_call(setup, monkeypatch):
    from agent.application import chat
    from agent.storage.records import save

    class Crash(BaseException):
        pass

    store, conversation, _ = setup
    item = incoming()
    store.submit(conversation, item)
    model = ScriptedModel([reply(call('read_file', path='x', offset=0, limit=2))])
    invoked = []
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model,
                  invoke_tool=lambda *_: invoked.append(1))

    def crash_after_save(path, value):
        save(path, value)
        if Path(path).name == 'response.json':
            raise Crash()

    with monkeypatch.context() as changes:
        changes.setattr('agent.storage.records.save', crash_after_save)
        with pytest.raises(Crash):
            bot.run(conversation, item)
    store.recover_interrupted()
    assert not invoked and len(model.calls) == 1
    assert store.usage(conversation)['chat']['unknown_calls'] == 0
    result = json.loads(tools(store, conversation)[0]['content'])
    assert result['actions_executed'] is False
    before = store.model_history(conversation)
    store.recover_interrupted()
    assert store.model_history(conversation) == before
