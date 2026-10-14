"""为完整的本地 Agent Server ASGI 栈提供实例认证。"""

import hmac
import os
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

MAX_BODY = 8 * 1024 * 1024
LOCAL_HOSTS = {'127.0.0.1', 'localhost', 'monoplanner.localhost'}


def service_headers():
    token = os.getenv('MONOPLANER_SERVICE_TOKEN', '')
    if len(token) < 32:
        raise RuntimeError('Start MonoPlanner through the local launcher.')
    return {'X-MonoPlanner-Service': token}


def local_address(value, port, *, origin=False):
    try:
        parsed = urlsplit(value if origin else 'http://' + value)
        return (
            parsed.scheme == 'http'
            and parsed.hostname in LOCAL_HOSTS
            and parsed.port == port
            and not parsed.username
            and not parsed.password
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


class InstanceAccess:
    """纯 ASGI 中间件，保持响应流和取消语义不变。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] not in {'http', 'websocket'}:
            return await self.app(scope, receive, send)
        headers = {k.decode('latin1').lower(): v.decode('latin1') for k, v in scope['headers']}
        status, error = None, None
        if not local_address(
            headers.get('host', ''), int(os.getenv('MONOPLANER_BACKEND_PORT', '2024'))
        ):
            status, error = 403, 'Invalid local service host'
        elif scope['type'] == 'http' and scope['path'] == '/ok' and scope['method'] == 'GET':
            return await JSONResponse({'ok': True})(scope, receive, send)
        else:
            token = os.getenv('MONOPLANER_SERVICE_TOKEN', '')
            if len(token) < 32:
                status, error = 503, 'Local service access is not initialized'
            elif not hmac.compare_digest(headers.get('x-monoplanner-service', ''), token):
                status, error = 401, 'Local service authentication required'
            elif headers.get('origin') and not local_address(
                headers['origin'], int(os.getenv('MONOPLANER_UI_PORT', '3000')), origin=True
            ):
                status, error = 403, 'Cross-site request denied'
            elif scope['type'] == 'websocket':
                status, error = 403, 'Unsupported connection'
        # 即使框架默认行为改变，这些接口也不属于产品操作。
        path = scope['path'].rstrip('/')
        if status is None and (
            path in {'/info', '/metrics', '/docs', '/redoc', '/openapi.json'}
            or path.split('/')[1] in {'mcp', 'a2a', 'store', 'webhooks', 'studio'}
            or 'crons' in path.split('/')
        ):
            status, error = 404, 'Endpoint unavailable'
        if (
            status is None
            and scope['type'] == 'http'
            and scope['method'] in {'POST', 'PUT', 'PATCH', 'DELETE'}
        ):
            try:
                size = int(headers.get('content-length', '0'))
                if size < 0 or size > MAX_BODY:
                    status, error = 413, 'Request too large'
                elif size and not headers.get('content-type', '').lower().startswith(
                    'application/json'
                ):
                    status, error = 415, 'JSON request required'
            except ValueError:
                status, error = 400, 'Invalid request length'
            if status is None:
                # 进入任何修改接口前，也限制分块传输的请求体大小。
                body = bytearray()
                while True:
                    part = await receive()
                    if part['type'] == 'http.disconnect':
                        return
                    body.extend(part.get('body', b''))
                    if len(body) > MAX_BODY:
                        status, error = 413, 'Request too large'
                        break
                    if not part.get('more_body', False):
                        break
                if (
                    status is None
                    and body
                    and not headers.get('content-type', '').lower().startswith('application/json')
                ):
                    status, error = 415, 'JSON request required'
                delivered = False
                original_receive = receive

                async def bounded_receive():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                    return await original_receive()

                receive = bounded_receive
        if status is not None:
            if scope['type'] == 'websocket':
                return await send({'type': 'websocket.close', 'code': 1008})
            return await JSONResponse(
                {'error': error}, status_code=status, headers={'Cache-Control': 'no-store'}
            )(scope, receive, send)
        await self.app(scope, receive, send)
