"""可信界面接口；预览框架不能发起控制请求。"""

import asyncio
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route

from agent.execution.applications import _launch_environment
from agent.execution.desktop import DesktopStore, fingerprint
from agent.storage.preferences import state_root
from agent.storage.permissions import Permissions
from agent.storage.read_sources import ReadSources
from agent.settings import PROJECT_ROOT


def trusted_ui(request):
    thread = request.path_params.get('thread')
    if thread:
        from agent.application.api import get_service
        saved = get_service().store.get(thread)
        if saved.get('read_only'):
            raise ValueError("当前对话只读，不能设置权限或操作文件。")
    origin = request.headers.get('origin', '')
    parsed = urlsplit(origin)
    if (
        parsed.scheme != 'http'
        or parsed.hostname not in {'127.0.0.1', 'localhost', 'monoplanner.localhost'}
        or parsed.port != int(os.getenv('MONOPLANER_UI_PORT', '3000'))
    ):
        raise ValueError('只接受本机主界面的操作，不接受预览页面或外部网站')
    if not request.headers.get('content-type', '').startswith('application/json'):
        raise ValueError('操作必须使用 JSON')


async def permission(request):
    try:
        store = await asyncio.to_thread(lambda: Permissions(state_root()))
        thread = request.path_params['thread']
        if request.method == 'POST':
            trusted_ui(request)
            value = await request.json()
            if set(value) != {'mode'}:
                raise ValueError('只接受 mode 字段')
            result = await asyncio.to_thread(store.request, thread, value['mode'])
        else:
            result = await asyncio.to_thread(store.get, thread)
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except (OSError, TypeError, KeyError, ValueError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def opens(request):
    try:
        store = await asyncio.to_thread(lambda: DesktopStore(state_root()))
        thread = request.path_params['thread']
        if request.method == 'POST':
            trusted_ui(request)
            value = await request.json()
            if set(value) not in ({'id', 'target'}, {'id', 'target', 'action'}):
                raise ValueError('只接受打开请求编号及原目标')
            action = value.get('action', 'approve')
            if action not in {'approve', 'reject'}:
                raise ValueError('action 必须为 approve 或 reject')
            result = await asyncio.to_thread(
                store.approve if action == 'approve' else store.reject,
                thread,
                value['id'],
                value['target'],
            )
        else:
            result = {'items': await asyncio.to_thread(store.list, thread)}
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except (OSError, TypeError, KeyError, ValueError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def preview(request):
    store = await asyncio.to_thread(lambda: DesktopStore(state_root()))
    record = await asyncio.to_thread(
        store.get, request.path_params['thread'], request.path_params['identity']
    )
    if not record or record['kind'] != 'preview':
        return JSONResponse({'error': '找不到预览'}, status_code=404)
    path = Path(record['record'])
    valid = await asyncio.to_thread(
        lambda: path.resolve().parent == store.root / 'previews'
        and fingerprint(path) == record['sha256']
    )
    if not valid:
        return JSONResponse({'error': '预览原件已变化'}, status_code=409)
    return FileResponse(
        path,
        media_type=record['mime'],
        headers={
            'Cache-Control': 'no-store',
            'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; frame-ancestors "
            + ' '.join(
                f"http://{host}:{int(os.getenv('MONOPLANER_UI_PORT', '3000'))}"
                for host in ('127.0.0.1', 'localhost', 'monoplanner.localhost')
            ),
            'Referrer-Policy': 'no-referrer',
        },
    )


async def choose_file(request):
    try:
        trusted_ui(request)
        value = await request.json()
        if value not in ({}, {'kind': 'file'}, {'kind': 'directory'}):
            raise ValueError('选择窗口只接受 file 或 directory 类型')
        kind = value.get('kind', 'file')
        helper = PROJECT_ROOT / 'launcher/runtime/MonoPlanner.FilePicker.exe'
        if not await asyncio.to_thread(helper.is_file):
            raise ValueError('本机文件选择组件不可用')
        result = await asyncio.to_thread(
            subprocess.run,
            [str(helper), kind],
            capture_output=True,
            encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW,
            env=_launch_environment(),
        )
        if result.returncode:
            raise ValueError('文件选择未完成')
        path = result.stdout.strip()
        if path:
            store = await asyncio.to_thread(lambda: ReadSources(state_root()))
            sources = await asyncio.to_thread(store.add, request.path_params['thread'], path, kind)
        else:
            sources = None
        return JSONResponse(
            {'path': path, 'sources': sources}, headers={'Cache-Control': 'no-store'}
        )
    except (OSError, TypeError, ValueError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def read_sources(request):
    try:
        store = await asyncio.to_thread(lambda: ReadSources(state_root()))
        thread = request.path_params['thread']
        if request.method == 'POST':
            trusted_ui(request)
            value = await request.json()
            if not isinstance(value, dict) or set(value) != {'sources'}:
                raise ValueError('只接受 sources 列表')
            sources = await asyncio.to_thread(store.set, thread, value['sources'])
        else:
            sources = await asyncio.to_thread(store.get, thread)
        return JSONResponse({'sources': sources}, headers={'Cache-Control': 'no-store'})
    except (OSError, TypeError, KeyError, ValueError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def backup(request):
    store = await asyncio.to_thread(lambda: DesktopStore(state_root()))
    thread, identity = request.path_params['thread'], request.path_params['identity']
    record = await asyncio.to_thread(store.get, thread, identity)
    if not record or record['kind'] != 'file_operation' or not record.get('backup'):
        return JSONResponse({'error': '找不到恢复副本'}, status_code=404)
    expected = (
        store.root / 'file-operation-backups' / store.store.thread(thread) / (identity + '.bin')
    )
    path = Path(record['backup'])
    valid = await asyncio.to_thread(
        lambda: path == expected
        and path.is_file()
        and fingerprint(path) == record['before'][record['path']]['sha256']
    )
    if not valid:
        return JSONResponse({'error': '恢复副本已变化'}, status_code=409)
    return FileResponse(
        path,
        filename=Path(record['path']).name,
        media_type='application/octet-stream',
        headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'},
    )


routes = [
    Route('/threads/{thread}/permission', permission, methods=['GET', 'POST']),
    Route('/threads/{thread}/opens', opens, methods=['GET', 'POST']),
    Route('/threads/{thread}/opens/{identity}/backup', backup, methods=['GET']),
    Route('/threads/{thread}/preview/{identity}', preview, methods=['GET']),
    Route('/threads/{thread}/choose-file', choose_file, methods=['POST']),
    Route('/threads/{thread}/read-sources', read_sources, methods=['GET', 'POST']),
]
