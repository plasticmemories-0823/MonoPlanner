"""仅验证桌面界面控制，不实际提权或启动应用程序。"""

import asyncio
from pathlib import Path
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from tests.fixtures.web_server import graph
from agent.server.app import app as application
from agent.storage.conversations import ConversationStore
from agent.storage.preferences import state_root
from agent.storage.read_sources import ReadSources
from agent.server.desktop import permission as real_permission

permissions, calls = {}, []
options = {'permission_error': False, 'read_delay': 0}
fixture_file = str(Path(__file__).resolve())


async def permission(request):
    thread = request.path_params['thread']
    value = permissions.setdefault(thread, {'requested': 'sandbox', 'effective': 'sandbox'})
    if request.method == 'POST':
        change = await request.json()
        calls.append({'kind': 'permission', 'thread': thread, **change})
        if options['permission_error']:
            return JSONResponse({'error': '离线样本：权限保存失败'}, status_code=503)
        value['requested'] = change['mode']
    else:
        await asyncio.sleep(options['read_delay'])
        session = await asyncio.to_thread(lambda: ConversationStore(state_root()).session(thread))
        if session and session['status'] == 'chat_complete':
            value['effective'] = value['requested']
    return await real_permission(request)


async def choose(request):
    value = await request.json()
    kind = value.get('kind', 'file')
    calls.append({'kind': kind, 'thread': request.path_params['thread']})
    path = str(Path(fixture_file).parent) if kind == 'directory' else fixture_file
    sources = await asyncio.to_thread(
        lambda: ReadSources(state_root()).add(request.path_params['thread'], path, kind)
    )
    return JSONResponse({'path': path, 'sources': sources})


async def checks(request):
    if request.method == 'POST':
        options.update(await request.json())
    return JSONResponse(
        {
            'calls': calls,
            'permissions': permissions,
            'options': options,
            'real_permission_changes': 0,
            'applications_started': 0,
        }
    )


app = Starlette(
    routes=[
        Route('/threads/{thread}/permission', permission, methods=['GET', 'POST']),
        Route('/threads/{thread}/choose-file', choose, methods=['POST']),
        Route('/test/desktop-checks', checks, methods=['GET', 'POST']),
        *application.routes,
    ]
)
