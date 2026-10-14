"""旧会话适配的离线回归参考，不由产品加载。"""
import asyncio
import json
import os
import sqlite3
from urllib.parse import urlsplit
from starlette.responses import JSONResponse
from starlette.requests import Request
from agent.storage.preferences import state_root
from tests.legacy.conversations import ConversationStore
from agent.storage.activity import read_activity
from tests.legacy.storage_titles import Titles

def control_value(thread, value=None):
    store = ConversationStore(state_root())
    thread = store.thread(thread)
    session = store.session(thread)
    if not session and value is None:
        return JSONResponse({'error': 'Conversation has no task'}, status_code=404)
    if value is None:
        pending = store.pending(thread)
        return JSONResponse(
            {
                'status': session['status'],
                'task_id': session['task_id'],
                'runtime_settings': session.get('runtime_settings'),
                'workspace_bound': session.get('workspace_bound', False),
                'read_roots': session.get('read_roots', []),
                'read_sources': session.get('read_sources', []),
                'repo': session['repo'],
                'messages': [
                    {'id': m['data'].get('id'), 'type': m['type'], 'content': m['data']['content']}
                    for m in session.get('public_messages', [])
                ],
                'pending': len(pending),
                'pending_messages': [
                    {'id': row['id'], 'type': 'human', 'content': row['text']}
                    for row in pending
                    if row['kind'] == 'message'
                ],
            }
        )
    record = store.enqueue(thread, value['id'], value['kind'], value.get('text', ''))
    if not record['applied']:
        from agent.task.control import interrupt

        interrupt(state_root(), thread, record['kind'], record['id'])
    return JSONResponse({'received': True, 'id': record['id'], 'applied': bool(record['applied'])})


async def controls(request: Request):
    try:
        value = await request.json() if request.method == 'POST' else None
        return await asyncio.to_thread(control_value, request.path_params['thread'], value)
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


def activity_value(thread, cursors):
    store = ConversationStore(state_root())
    session = store.session(thread)
    tasks = (
        (
            session.get('activity_tasks')
            or [{'id': session['task_id'], 'input_id': session['last_input']}]
        )
        if session
        else []
    )
    result = read_activity(state_root(), tasks, cursors)
    auxiliary = Titles(state_root()).activity(thread)
    result['usage'].update(auxiliary.pop('usage'))
    result.update(auxiliary)
    return JSONResponse(result)


async def activity(request: Request):
    try:
        cursors = json.loads(request.query_params.get('after', '{}'))
        if not isinstance(cursors, dict):
            raise ValueError('Activity cursors must be an object')
        return await asyncio.to_thread(activity_value, request.path_params['thread'], cursors)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


async def conversations(request: Request):
    def read():
        store = ConversationStore(state_root())
        return {
            'visibility': store.visibility(),
            'pending_deletions': store.pending_deletions(),
            'titles': Titles(state_root()).public(),
        }

    return JSONResponse(await asyncio.to_thread(read), headers={'Cache-Control': 'no-store'})


async def visibility(request: Request):
    try:
        origin = request.headers.get('origin')
        parsed = urlsplit(origin) if origin else None
        if parsed and (
            parsed.scheme != 'http'
            or parsed.hostname not in {'127.0.0.1', 'localhost', 'monoplanner.localhost'}
            or parsed.port != int(os.getenv('MONOPLANER_UI_PORT', '3000'))
        ):
            return JSONResponse({'error': '仅允许本机前端管理对话'}, status_code=403)
        if not request.headers.get('content-type', '').startswith('application/json'):
            return JSONResponse({'error': '请求必须使用 JSON'}, status_code=415)
        value = await request.json()
        if set(value) != {'visibility'}:
            raise ValueError('只接受 visibility 字段')
        return JSONResponse(
            await asyncio.to_thread(
                lambda: ConversationStore(state_root()).set_visibility(
                    request.path_params['thread'], value['visibility']
                )
            )
        )
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)
