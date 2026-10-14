"""新产品接口。认证由服务外层统一提供，读事件不触发执行。"""

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from threading import Lock

from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from agent.application.store import Conflict, identity


def build_routes(store, scheduler):
    def health():
        return scheduler.storage_error() if hasattr(scheduler, 'storage_error') else None

    def storage_failure(operation, error):
        if hasattr(scheduler, 'storage_failed'):
            return scheduler.storage_failed(operation, error)
        return None

    def response(value, status=200):
        return JSONResponse(value, status_code=status, headers={'Cache-Control': 'no-store'})

    async def guarded(call):
        try:
            return await call()
        except Conflict as exc:
            return response({'error': str(exc)}, 409)
        except KeyError:
            return response({'error': 'Conversation does not exist'}, 404)
        except (ValueError, TypeError) as exc:
            return response({'error': str(exc)}, 400)
        except (OSError, sqlite3.Error) as exc:
            fault = storage_failure('api_record', exc)
            return response({'error': (fault or {}).get('message', '本地记录操作未完成，请核对后重试。'),
                             'storage_error': fault}, 503)

    async def conversations(request):
        async def run():
            if request.method == 'GET':
                rows = await asyncio.to_thread(store.list)
                return response({'conversations': rows, 'legacy_read_only': True})
            value = await request.json()
            if hasattr(scheduler, 'ensure_dispatch_available'):
                scheduler.ensure_dispatch_available()
            if not isinstance(value, dict) or set(value) - {'id', 'repo', 'runtime_settings'}:
                raise ValueError('Invalid conversation fields')
            created = await asyncio.to_thread(store.create, value.get('id'),
                                               value.get('repo', ''), value.get('runtime_settings'))
            return response(created, 201)
        return await guarded(run)

    async def snapshot(request):
        async def run():
            value = await asyncio.to_thread(store.snapshot, request.path_params['conversation'])
            return response({**value, 'storage_error': health()})
        return await guarded(run)

    async def commands(request):
        async def run():
            conversation = identity(request.path_params['conversation'])
            command = await request.json()
            result = await asyncio.to_thread(scheduler.submit, conversation, command)
            if result.get('recorded') is False:
                return response(result, 202)
            current = await asyncio.to_thread(store.snapshot, conversation)
            return response({**result, 'status': current['status'],
                             'event_cursor': current['event_cursor']}, 202)
        return await guarded(run)

    async def diagnostics(request):
        async def run():
            from agent.storage.activity import read_activity

            conversation = identity(request.path_params['conversation'])
            snapshot_value = await asyncio.to_thread(store.snapshot, conversation)
            if snapshot_value['read_only']:
                raise Conflict('旧版本详细诊断请使用对应冻结版本查看')
            jobs = await asyncio.to_thread(store.work_for_conversation, conversation)
            tasks = [{'id': job['task_id'], 'input_id': job['input_id']} for job in jobs]
            cursors = json.loads(request.query_params.get('after', '{}'))
            if not isinstance(cursors, dict):
                raise ValueError('诊断游标必须为对象')
            return response(await asyncio.to_thread(read_activity, store.root, tasks, cursors))
        return await guarded(run)

    async def events(request):
        async def run():
            conversation = identity(request.path_params['conversation'])
            after = int(request.query_params.get('after', request.headers.get('last-event-id', '0')))
            first = await asyncio.to_thread(store.events, conversation, after)
            if request.query_params.get('once') == '1':
                return response(first)

            async def stream():
                cursor, batch = after, first
                idle = 0
                previous_health = None
                while not await request.is_disconnected():
                    for event in batch['events']:
                        yield ('id: ' + str(event['seq']) + '\nevent: update\ndata: '
                               + json.dumps(event, ensure_ascii=False) + '\n\n')
                    cursor = batch['cursor']
                    current_health = health()
                    if current_health != previous_health:
                        yield ('event: health\ndata: ' + json.dumps(
                            {'storage_error': current_health}, ensure_ascii=False) + '\n\n')
                        previous_health = current_health
                    idle += 1
                    if idle % 20 == 0:
                        yield ': heartbeat\n\n'
                    await asyncio.sleep(0 if batch['more'] else 0.5)
                    if current_health:
                        batch = {'events': [], 'cursor': cursor, 'more': False}
                        continue
                    try:
                        batch = await asyncio.to_thread(store.events, conversation, cursor)
                    except (OSError, sqlite3.Error) as exc:
                        storage_failure('event_record_read', exc)
                        batch = {'events': [], 'cursor': cursor, 'more': False}
            return StreamingResponse(stream(), media_type='text/event-stream', headers={
                'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no',
            })
        return await guarded(run)

    async def visibility(request):
        async def run():
            value = await request.json()
            if set(value) != {'visibility'}:
                raise ValueError('Only visibility is accepted')
            return response(await asyncio.to_thread(store.visibility,
                identity(request.path_params.get('conversation') or request.path_params['thread']), value['visibility']))
        return await guarded(run)

    async def purge(request):
        async def run():
            conversation = identity(request.path_params.get('conversation') or request.path_params['thread'])
            value = await request.json()
            if (not isinstance(value, dict)
                    or set(value) - {'confirm_thread_id', 'retain_environment'}
                    or value.get('confirm_thread_id') != conversation
                    or type(value.get('retain_environment', False)) is not bool):
                raise ValueError('Permanent deletion requires the exact conversation ID')
            return response(await asyncio.to_thread(scheduler.purge, conversation,
                                                      value.get('retain_environment', False)))
        return await guarded(run)

    return [
        Route('/conversations', conversations, methods=['GET', 'POST']),
        Route('/conversations/{conversation}/snapshot', snapshot, methods=['GET']),
        Route('/conversations/{conversation}/commands', commands, methods=['POST']),
        Route('/conversations/{conversation}/diagnostics', diagnostics, methods=['GET']),
        Route('/conversations/{conversation}/events', events, methods=['GET']),
        Route('/conversations/{conversation}/visibility', visibility, methods=['POST']),
        Route('/conversations/{conversation}/purge', purge, methods=['POST']),
        Route('/threads/{thread}/visibility', visibility, methods=['POST']),
        Route('/threads/{thread}/purge', purge, methods=['POST']),
    ]


_service = None
_lock = Lock()


def get_service():
    global _service
    from agent.storage.preferences import state_root
    from agent.application.store import ApplicationStore
    from agent.application.scheduler import Scheduler

    root = state_root()
    with _lock:
        if _service is None:
            _service = Scheduler(ApplicationStore(root))
        elif _service.store.root != root:
            raise RuntimeError('Application service data directory cannot change while loaded')
        return _service


class _LazyStore:
    def __getattr__(self, name):
        return getattr(get_service().store, name)


class _LazyScheduler:
    def __getattr__(self, name):
        return getattr(get_service(), name)


routes = build_routes(_LazyStore(), _LazyScheduler())


@asynccontextmanager
async def lifespan(app):
    global _service
    service = await asyncio.to_thread(get_service)
    from agent.application.shutdown import watch_desktop_close

    close_watcher = watch_desktop_close(service.shutdown)
    try:
        await asyncio.to_thread(service.start)
        yield
    finally:
        close_watcher()
        await asyncio.to_thread(service.close)
        with _lock:
            if _service is service:
                _service = None
