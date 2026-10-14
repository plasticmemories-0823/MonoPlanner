"""由官方 Agent Server 挂载的本地工作目录接口。"""

from starlette.middleware import Middleware
from agent.server.access import InstanceAccess, service_headers
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
import httpx
from langgraph_sdk import get_client
from agent.storage.purge import begin_purge, finish_purge
from urllib.parse import urlsplit
from agent.server.desktop import routes as desktop_routes
from agent.execution.projects import ProjectEnvironments, local_images, workspace_binding
from agent.storage.titles import Titles
from agent.server.titles import lifespan


async def project_environment(request: Request):
    """仅允许经过认证的准备操作；GET 不安装依赖，也不调用模型。"""
    from agent.execution.preparation import jobs

    try:
        if request.method == 'GET':
            result = await asyncio.to_thread(
                lambda: jobs().status(request.query_params.get('repo', ''))
            )
        else:
            raw = await request.body()
            if len(raw) > 8192:
                raise ValueError('环境准备请求过大')
            value = json.loads(raw)
            if (
                not isinstance(value, dict)
                or set(value) != {'repo', 'action'}
                or not isinstance(value['repo'], str)
                or value['action'] not in {'prepare', 'automatic', 'cancel'}
            ):
                raise ValueError('请选择准备环境、自动模式或取消')
            if value['action'] == 'automatic':
                await asyncio.to_thread(lambda: ProjectEnvironments().automatic(value['repo']))
                result = await asyncio.to_thread(lambda: jobs().status(value['repo']))
            else:

                def change():
                    service = jobs()
                    return (service.start if value['action'] == 'prepare' else service.cancel)(
                        value['repo']
                    )

                result = await asyncio.to_thread(change)
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse(
            {'error': str(exc)}, status_code=400, headers={'Cache-Control': 'no-store'}
        )


LOADED_SERVICE = service_identity()


async def service(request: Request):
    return JSONResponse(LOADED_SERVICE, headers={'Cache-Control': 'no-store'})


async def service_status(request: Request):
    # 仅提供最小生命周期状态，保持指标和内部诊断关闭。
    try:
        from langgraph_runtime.database import connect
        from langgraph_runtime.ops import Runs

        async with connect() as connection:
            queue = await Runs.stats(connection)
        from agent.execution.preparation import active_count

        preparing = await asyncio.to_thread(active_count)
        return JSONResponse(
            {
                'queue': {k: queue[k] for k in ('n_running', 'n_pending')},
                'environment_preparations': preparing,
            },
            headers={'Cache-Control': 'no-store'},
        )
    except Exception:
        return JSONResponse({'error': 'Cannot confirm whether tasks are idle'}, status_code=503)


async def service_reconcile(request: Request):
    # 实例认证由外层统一检查；只修复生命周期记录，不执行任务。
    try:
        if await request.json() != {}:
            raise ValueError('Service recovery accepts no task or execution arguments')
        from agent.server.lifecycle import reconcile

        root = await asyncio.to_thread(state_root)
        return JSONResponse(await reconcile(root), headers={'Cache-Control': 'no-store'})
    except ValueError as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)
    except Exception:
        import logging

        logging.getLogger(__name__).exception('Service lifecycle recovery failed')
        return JSONResponse({'error': 'Cannot confirm service recovery'}, status_code=503)


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


async def execution_images(request: Request):
    try:
        return JSONResponse(
            await asyncio.to_thread(local_images), headers={'Cache-Control': 'no-store'}
        )
    except (OSError, ValueError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=503)


async def project_execution(request: Request):
    try:
        if request.method == 'POST':
            origin = urlsplit(request.headers['origin']) if request.headers.get('origin') else None
            if origin and (
                origin.scheme != 'http'
                or origin.hostname not in {'127.0.0.1', 'localhost', 'monoplanner.localhost'}
                or origin.port != int(os.getenv('MONOPLANER_UI_PORT', '3000'))
            ):
                return JSONResponse({'error': '仅允许本机前端配置执行环境'}, status_code=403)
            if not request.headers.get('content-type', '').startswith('application/json'):
                return JSONResponse({'error': '配置必须使用 JSON'}, status_code=415)
            raw = await request.body()
            if len(raw) > 8192:
                raise ValueError('配置请求过大')
            result = await asyncio.to_thread(lambda: ProjectEnvironments().save(json.loads(raw)))
        else:
            result = await asyncio.to_thread(
                workspace_binding, request.query_params.get('repo', '')
            )
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


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
        from agent.server.interrupts import interrupt

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


async def purge(request: Request):
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
        thread = ConversationStore.thread(request.path_params['thread'])
        if value != {'confirm_thread_id': thread}:
            raise ValueError('彻底删除须明确确认当前对话编号。')
        # 进程内 SDK 仍需通过实例访问检查。
        # 不建立网络连接，同时保留认证。
        client = get_client(
            url=None,
            api_key=None,
            headers={
                **service_headers(),
                'Host': '127.0.0.1:' + os.getenv('MONOPLANER_BACKEND_PORT', '2024'),
            },
        )
        try:
            try:
                record = await client.threads.get(thread)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
                record = None
            if record and record['status'] == 'busy':
                raise ValueError('请先暂停任务，并等待当前请求或操作结束。')
            store = await asyncio.to_thread(lambda: ConversationStore(state_root()))
            identities = await asyncio.to_thread(begin_purge, store, thread)
            if record:
                try:
                    await client.threads.delete(thread)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 404:
                        raise
            result = await asyncio.to_thread(finish_purge, store, thread, identities)
        finally:
            await client.aclose()
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)
    except (httpx.HTTPError, OSError, sqlite3.Error):
        return JSONResponse(
            {'error': '彻底删除尚未完成，请检查服务后重试；不会自动重复操作。'}, status_code=503
        )


async def settings(request: Request):
    try:
        if request.method != 'GET':
            origin = request.headers.get('origin')
            parsed = urlsplit(origin) if origin else None
            if parsed and (
                parsed.scheme != 'http'
                or parsed.hostname not in {'127.0.0.1', 'localhost', 'monoplanner.localhost'}
                or parsed.port != int(os.getenv('MONOPLANER_UI_PORT', '3000'))
            ):
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
        return JSONResponse(
            {'error': str(exc)}, status_code=400, headers={'Cache-Control': 'no-store'}
        )


app = Starlette(
    lifespan=lifespan,
    middleware=[Middleware(InstanceAccess)],
    routes=[
        *desktop_routes,
        Route('/service', service, methods=['GET']),
        Route('/service/status', service_status, methods=['GET']),
        Route('/service/reconcile', service_reconcile, methods=['POST']),
        Route('/execution/images', execution_images, methods=['GET']),
        Route('/workspace/execution', project_execution, methods=['GET', 'POST']),
        Route('/workspace/environment', project_environment, methods=['GET', 'POST']),
        Route('/workspace', workspace, methods=['GET', 'POST']),
        Route('/workspace/directories', directories, methods=['GET']),
        Route('/settings', settings, methods=['GET', 'POST']),
        Route('/settings/api-key', settings, methods=['DELETE']),
        Route('/conversations', conversations, methods=['GET']),
        Route('/threads/{thread}/visibility', visibility, methods=['POST']),
        Route('/threads/{thread}/purge', purge, methods=['POST']),
        Route('/threads/{thread}/control', controls, methods=['GET', 'POST']),
        Route('/threads/{thread}/activity', activity, methods=['GET']),
    ],
)
