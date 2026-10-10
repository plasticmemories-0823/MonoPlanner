"""Trusted UI endpoints; preview frames cannot perform control requests."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route

from agent.execution.desktop import DesktopStore, fingerprint
from agent.storage.preferences import state_root
from agent.storage.permissions import Permissions
from agent.settings import PROJECT_ROOT


def trusted_ui(request):
    origin = request.headers.get('origin', '')
    parsed = urlsplit(origin)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', 'monoplanner.localhost'}
            or parsed.port != int(os.getenv('MONOPLANER_UI_PORT', '3000'))):
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
            if set(value) != {'id', 'target'}:
                raise ValueError('只接受打开请求编号及原目标')
            result = await asyncio.to_thread(store.approve, thread, value['id'], value['target'])
        else:
            result = {'items': await asyncio.to_thread(store.list, thread)}
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except (OSError, TypeError, KeyError, ValueError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def preview(request):
    store = await asyncio.to_thread(lambda: DesktopStore(state_root()))
    record = await asyncio.to_thread(store.get, request.path_params['thread'], request.path_params['identity'])
    if not record or record['kind'] != 'preview':
        return JSONResponse({'error': '找不到预览'}, status_code=404)
    path = Path(record['record'])
    valid = await asyncio.to_thread(lambda: path.resolve().parent == store.root/'previews'
                                   and fingerprint(path) == record['sha256'])
    if not valid:
        return JSONResponse({'error': '预览原件已变化'}, status_code=409)
    return FileResponse(path, media_type=record['mime'], headers={
        'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; frame-ancestors http://127.0.0.1:3000 http://localhost:3000 http://monoplanner.localhost:3000",
        'Referrer-Policy': 'no-referrer'})


async def choose_file(request):
    try:
        trusted_ui(request)
        if await request.json() != {}:
            raise ValueError('文件选择不接受模型指定参数')
        helper = PROJECT_ROOT/'launcher/runtime/MonoPlanner.FilePicker.exe'
        if not helper.is_file():
            raise ValueError('本机文件选择组件不可用')
        application = request.url.path.endswith('/choose-application')
        result = await asyncio.to_thread(subprocess.run, [str(helper)]+(['application'] if application else []), capture_output=True,
                                         encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise ValueError('文件选择未完成')
        path = result.stdout.strip()
        if path:
            store = await asyncio.to_thread(lambda: DesktopStore(state_root()))
            if application:
                item = await asyncio.to_thread(store.register_application, path)
                return JSONResponse({'application': item}, headers={'Cache-Control': 'no-store'})
            path = await asyncio.to_thread(store.authorize_file,
                                          request.path_params['thread'], path)
            await asyncio.to_thread(store.preview,
                                   request.path_params['thread'], path, '', [])
        return JSONResponse({'path': path}, headers={'Cache-Control': 'no-store'})
    except (OSError, TypeError, ValueError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


routes = [Route('/threads/{thread}/permission', permission, methods=['GET', 'POST']),
          Route('/threads/{thread}/opens', opens, methods=['GET', 'POST']),
          Route('/threads/{thread}/preview/{identity}', preview, methods=['GET']),
          Route('/threads/{thread}/choose-file', choose_file, methods=['POST'])]
routes.append(Route('/threads/{thread}/choose-application', choose_file, methods=['POST']))
