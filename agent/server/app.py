"""Small local workspace endpoint mounted by the official Agent Server."""
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent.cli.workspace import load_saved_repo
from agent.storage.conversations import ConversationStore
from agent.storage.activity import read_activity
from agent.server.graph import state_root
from agent.storage.preferences import Preferences
from agent.server.workspaces import directory_page
from agent.server.service import service_identity
from pathlib import Path
import asyncio
import json
import os
import sqlite3
from urllib.parse import urlsplit


LOADED_SERVICE = service_identity()


async def service(request: Request):
    return JSONResponse(LOADED_SERVICE, headers={'Cache-Control': 'no-store'})


def workspace_value(value=None):
    if value is None:
        try:
            return {'repo': load_saved_repo() or ''}
        except (OSError, ValueError):
            return {'repo': ''}
    directory = Path(value['repo']).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError('Workspace must be an existing directory')
    return {'repo': str(directory)}


async def workspace(request: Request):
    try:
        value = await request.json() if request.method == 'POST' else None
        return JSONResponse(await asyncio.to_thread(workspace_value, value))
    except (KeyError, TypeError, OSError, ValueError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


async def directories(request: Request):
    try:
        result = await asyncio.to_thread(directory_page, request.query_params.get('path', ''))
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except (OSError, ValueError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


def control_value(thread, value=None):
    store = ConversationStore(state_root())
    thread = store.thread(thread)
    session = store.session(thread)
    if not session and value is None:
        return JSONResponse({'error': 'Conversation has no task'}, status_code=404)
    if value is None:
        pending = store.pending(thread)
        return JSONResponse({'status': session['status'], 'task_id': session['task_id'],
                             'runtime_settings': session.get('runtime_settings'),
                             'workspace_bound': session.get('workspace_bound', False),
                             'read_roots': session.get('read_roots', []),
                             'repo': session['repo'], 'messages': [
                                 {'id': m['data'].get('id'), 'type': m['type'], 'content': m['data']['content']}
                                 for m in session.get('public_messages', [])],
                             'pending': len(pending), 'pending_messages': [
                                 {'id': row['id'], 'type': 'human', 'content': row['text']}
                                 for row in pending if row['kind'] == 'message']})
    record = store.enqueue(thread, value['id'], value['kind'], value.get('text', ''))
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
    if not session:
        return JSONResponse({'items': [], 'cursors': {}, 'more': False})
    tasks = session.get('activity_tasks') or [{'id': session['task_id'], 'input_id': session['last_input']}]
    return JSONResponse(read_activity(state_root(), tasks, cursors))


async def activity(request: Request):
    try:
        cursors = json.loads(request.query_params.get('after', '{}'))
        if not isinstance(cursors, dict):
            raise ValueError('Activity cursors must be an object')
        return await asyncio.to_thread(activity_value, request.path_params['thread'], cursors)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


async def conversations(request: Request):
    result = await asyncio.to_thread(lambda: ConversationStore(state_root()).visibility())
    return JSONResponse({'visibility': result},
                        headers={'Cache-Control': 'no-store'})


async def visibility(request: Request):
    try:
        origin = request.headers.get('origin')
        parsed = urlsplit(origin) if origin else None
        if parsed and (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', 'monoplanner.localhost'}
                       or parsed.port != int(os.getenv('MONOPLANER_UI_PORT', '3000'))):
            return JSONResponse({'error': '仅允许本机前端管理对话'}, status_code=403)
        if not request.headers.get('content-type', '').startswith('application/json'):
            return JSONResponse({'error': '请求必须使用 JSON'}, status_code=415)
        value = await request.json()
        if set(value) != {'visibility'}:
            raise ValueError('只接受 visibility 字段')
        return JSONResponse(await asyncio.to_thread(lambda: ConversationStore(state_root()).set_visibility(
            request.path_params['thread'], value['visibility'])))
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


async def settings(request: Request):
    try:
        if request.method != 'GET':
            origin = request.headers.get('origin')
            parsed = urlsplit(origin) if origin else None
            if parsed and (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', 'monoplanner.localhost'}
                           or parsed.port != int(os.getenv('MONOPLANER_UI_PORT', '3000'))):
                return JSONResponse({'error': '仅允许本机前端修改配置'}, status_code=403)
            if request.method == 'POST':
                if not request.headers.get('content-type', '').startswith('application/json'):
                    return JSONResponse({'error': '配置必须使用 JSON 请求'}, status_code=415)
                raw = await request.body()
                if len(raw) > 8192:
                    raise ValueError('配置请求过大')
                result = await asyncio.to_thread(lambda: Preferences().update(json.loads(raw)))
            else:
                result = await asyncio.to_thread(lambda: Preferences().delete_key())
        else:
            result = await asyncio.to_thread(lambda: Preferences().public())
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400, headers={'Cache-Control': 'no-store'})


app = Starlette(routes=[Route('/service', service, methods=['GET']),
                       Route('/workspace', workspace, methods=['GET', 'POST']),
                       Route('/workspace/directories', directories, methods=['GET']),
                       Route('/settings', settings, methods=['GET', 'POST']),
                       Route('/settings/api-key', settings, methods=['DELETE']),
                       Route('/conversations', conversations, methods=['GET']),
                       Route('/threads/{thread}/visibility', visibility, methods=['POST']),
                       Route('/threads/{thread}/control', controls, methods=['GET', 'POST']),
                       Route('/threads/{thread}/activity', activity, methods=['GET'])])
