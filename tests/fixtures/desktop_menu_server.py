"""UI-only desktop controls: no real permission elevation or application launch."""
import asyncio
from pathlib import Path
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from tests.fixtures.web_server import graph
from agent.server.app import app as application
from agent.storage.conversations import ConversationStore
from agent.storage.preferences import state_root

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
    return JSONResponse(value, headers={'Cache-Control': 'no-store'})


async def choose(request):
    application = request.url.path.endswith('choose-application')
    calls.append({'kind': 'application' if application else 'file', 'thread': request.path_params['thread']})
    return JSONResponse({'application': {'id': 'fixture-app', 'name': '离线样本应用'}}
                        if application else {'path': fixture_file})


async def checks(request):
    if request.method == 'POST':
        options.update(await request.json())
    return JSONResponse({'calls': calls, 'permissions': permissions, 'options': options,
                         'real_permission_changes': 0, 'applications_started': 0})


app = Starlette(routes=[
    Route('/threads/{thread}/permission', permission, methods=['GET', 'POST']),
    Route('/threads/{thread}/choose-file', choose, methods=['POST']),
    Route('/threads/{thread}/choose-application', choose, methods=['POST']),
    Route('/test/desktop-checks', checks, methods=['GET', 'POST']),
    *application.routes,
])
