"""独立本机产品服务：应用调度与工作内核分离。"""

from starlette.middleware import Middleware
from agent.server.access import InstanceAccess, service_headers
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent.cli.workspace import load_saved_repo
from agent.storage.preferences import state_root
from agent.storage.preferences import Preferences
from agent.server.workspaces import directory_page
from agent.server.service import service_identity
from pathlib import Path
import asyncio
import json
import os
from urllib.parse import urlsplit
from agent.server.desktop import routes as desktop_routes
from agent.execution.projects import ProjectEnvironments, local_images, workspace_binding
from agent.application.api import routes as application_routes, lifespan, get_service


async def project_environment(request: Request):
    """仅允许经过认证的准备操作；GET 不安装依赖，也不调用模型。"""
    from agent.execution.preparation import jobs
    from agent.application.scheduler import StorageUnavailable

    try:
        if request.method == 'GET':
            result = await asyncio.to_thread(
                lambda: jobs().status(request.query_params.get('repo', ''),
                                      verify=request.query_params.get('verify') == '1')
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
            if value['action'] != 'cancel':
                get_service().ensure_dispatch_available()
            if value['action'] == 'automatic':
                await asyncio.to_thread(lambda: ProjectEnvironments().automatic(value['repo']))
                result = await asyncio.to_thread(lambda: jobs().status(value['repo']))
            else:

                def change():
                    if value['action'] != 'cancel':
                        get_service().ensure_dispatch_available()
                    service = jobs()
                    return (service.retry if value['action'] == 'prepare' else service.cancel)(
                        value['repo']
                    )

                result = await asyncio.to_thread(change)
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    except StorageUnavailable as exc:
        return JSONResponse({'error': str(exc), 'storage_error': get_service().storage_error()},
                            status_code=503, headers={'Cache-Control': 'no-store'})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse(
            {'error': str(exc)}, status_code=400, headers={'Cache-Control': 'no-store'}
        )


LOADED_SERVICE = service_identity()


async def service(request: Request):
    return JSONResponse({**LOADED_SERVICE, 'storage_error': get_service().storage_error()},
                        headers={'Cache-Control': 'no-store'})


async def service_status(request: Request):
    try:
        state = await asyncio.to_thread(get_service().status)
        active = sum(state.get(name, 0) for name in (
            'running_work', 'running_chat', 'running_progress', 'running_title', 'running_control'))
        return JSONResponse({
            **state, 'queue': {'n_running': active, 'n_pending': state.get('pending', 0)},
        }, headers={'Cache-Control': 'no-store'})
    except Exception:
        return JSONResponse({'error': 'Cannot confirm whether tasks are idle'}, status_code=503)


async def service_shutdown(request: Request):
    """认证后的桌面控制请求只启动一次停止过程，不接收任意 PID 或命令。"""
    try:
        raw = await request.body()
        if len(raw) > 8192 or json.loads(raw) != {}:
            raise ValueError('Shutdown accepts no arguments')
    except (ValueError, TypeError):
        return JSONResponse({'error': 'Shutdown accepts an empty JSON object'}, status_code=400)
    shutdown = get_service().shutdown
    shutdown.start()
    # 不把终止工作排进工作线程池；超时后返回真实未确认项目，让启动器限时收尾。
    deadline = asyncio.get_running_loop().time() + 2
    while not shutdown.status()['complete'] and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.025)
    return JSONResponse(shutdown.status(), headers={'Cache-Control': 'no-store'})


async def service_reconcile(request: Request):
    # 应用启动时核验遗留命令；不再修改另一套框架的 run 状态。
    if await request.json() != {}:
        return JSONResponse({'error': 'Service recovery accepts no arguments'}, status_code=400)
    return JSONResponse({'reconciled': 0, 'scheduler': 'application'},
                        headers={'Cache-Control': 'no-store'})


async def resources(request: Request):
    from agent.storage.resources import resource_catalog, delete_owned_docker_resource, release_project
    try:
        catalog = await asyncio.to_thread(resource_catalog, state_root())
        if request.method == 'GET':
            value = await asyncio.to_thread(catalog.usage)
        else:
            from agent.server.desktop import trusted_ui
            trusted_ui(request)
            data = await request.json()
            if request.url.path.endswith('/settings'):
                if set(data) != {'cache_limit_bytes', 'reserve_bytes'}:
                    raise ValueError('只接受缓存上限与预留空间')
                value = await asyncio.to_thread(catalog.set_limits, **data)
            elif request.url.path.endswith('/project'):
                if set(data) != {'repo', 'retain'} or type(data['retain']) is not bool:
                    raise ValueError('只接受项目目录和保留选择')
                value = await asyncio.to_thread(release_project, data['repo'],
                                               root=state_root(), retain=data['retain'])
            else:
                if set(data) - {'resource_ids'} or ('resource_ids' in data and
                    (not isinstance(data['resource_ids'], list) or
                     any(not isinstance(v, str) for v in data['resource_ids']))):
                    raise ValueError('只接受要清理的资源编号')
                value = await asyncio.to_thread(catalog.cleanup, data.get('resource_ids'),
                                               docker=delete_owned_docker_resource)
        return JSONResponse(value, headers={'Cache-Control': 'no-store'})
    except (ValueError, TypeError, OSError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


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
        *application_routes,
        *desktop_routes,
        Route('/resources', resources, methods=['GET']),
        Route('/resources/settings', resources, methods=['POST']),
        Route('/resources/cleanup', resources, methods=['POST']),
        Route('/resources/project', resources, methods=['POST']),
        Route('/service', service, methods=['GET']),
        Route('/service/status', service_status, methods=['GET']),
        Route('/service/shutdown', service_shutdown, methods=['POST']),
        Route('/service/reconcile', service_reconcile, methods=['POST']),
        Route('/execution/images', execution_images, methods=['GET']),
        Route('/workspace/execution', project_execution, methods=['GET', 'POST']),
        Route('/workspace/environment', project_environment, methods=['GET', 'POST']),
        Route('/workspace', workspace, methods=['GET', 'POST']),
        Route('/workspace/directories', directories, methods=['GET']),
        Route('/settings', settings, methods=['GET', 'POST']),
        Route('/settings/api-key', settings, methods=['DELETE']),
    ],
)
