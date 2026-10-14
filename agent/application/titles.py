"""一次性辅助标题；应用库保存原件，调度只由应用执行者负责。"""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, message_to_dict, messages_from_dict

from agent.application.store import encode, timestamp
from agent.model.client import ModelPolicy, create_model
from agent.model.transport import close_model
from agent.settings import PROJECT_ROOT, load_settings


POLICY = ModelPolicy(max_output_tokens=128, api_timeout=15, reasoning_effort='none')


def model_factory():
    return create_model(load_settings({
        'model_name': 'deepseek-flash', 'reasoning_effort': 'none', 'online': False,
    }), POLICY)


def reserve(store, conversation, command):
    record = {'title': ' '.join(command['text'].split())[:80] or '新对话',
              'status': 'pending', 'revision': 1, 'request_sent': False,
              'input_id': command['id'], 'created_at': timestamp()}
    try:
        record['request'] = [message_to_dict(message) for message in (
            SystemMessage(content=(PROJECT_ROOT / 'resources/prompts/title.txt').read_text(encoding='utf-8')),
            HumanMessage(content=command['text'][:4000]),
        )]
    except (OSError, ValueError) as exc:
        record.update(status='failed', error_type=type(exc).__name__)
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE')
        value = store._get(db, conversation, writable=True)
        if db.execute('SELECT 1 FROM app_titles WHERE conversation=?', (conversation,)).fetchone():
            return False
        db.execute('INSERT INTO app_titles VALUES(?,?)', (conversation, encode(record)))
        value['title'] = {key: record[key] for key in ('title', 'status', 'revision')}
        store._put(db, value)
        store._event(db, conversation, 'title', value['title'])
    return record['status'] == 'pending'


def update(store, conversation, expected, **changes):
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT data FROM app_titles WHERE conversation=?', (conversation,)).fetchone()
        value = store._get(db, conversation)
        if row is None or value['visibility'] in {'purging', 'purged'}:
            return None
        record = json.loads(row[0])
        if record['status'] not in expected:
            return None
        record.update(changes, revision=record['revision'] + 1)
        db.execute('UPDATE app_titles SET data=? WHERE conversation=?', (encode(record), conversation))
        value['title'] = {key: record[key] for key in ('title', 'status', 'revision')}
        store._put(db, value)
        store._event(db, conversation, 'title', value['title'])
        return record


def generate(store, conversation, factory=None):
    record = update(store, conversation, {'pending'}, status='running')
    if record is None:
        return
    call = store.begin_call(conversation, 'title', {'title_record': conversation})
    model, sent, started = None, False, time.monotonic()
    try:
        model = (factory or model_factory)()
        if update(store, conversation, {'running'}, request_sent=True) is None:
            store.finish_call(call, error='ConversationDeleted')
            return
        store.mark_sent(call)
        sent = True
        response = model.invoke(messages_from_dict(record['request']))
        text = response.content.strip() if isinstance(response.content, str) else ''
        valid = (bool(text) and len(text) <= 80 and '\n' not in text and '\r' not in text
                 and not response.tool_calls and not response.invalid_tool_calls
                 and response.response_metadata.get('finish_reason') == 'stop')
        changes = {'status': 'complete' if valid else 'failed',
                   'response': message_to_dict(response), 'usage': response.usage_metadata,
                   'seconds': time.monotonic() - started, 'finished_at': timestamp()}
        changes.update({'title': text} if valid else {'error_type': 'InvalidTitleResponse'})
        update(store, conversation, {'running'}, **changes)
        store.finish_call(call, response_id=conversation, usage=response.usage_metadata,
                          seconds=changes['seconds'])
    except Exception as exc:
        store.finish_call(call, seconds=time.monotonic() - started, error=type(exc).__name__)
        update(store, conversation, {'running'}, status='unknown' if sent else 'failed',
               error_type=type(exc).__name__, finished_at=timestamp())
    finally:
        close_model(model)
