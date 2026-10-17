"""攻击用例验证生产门控；不调用模型或外部服务。"""


import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient as RawClient

from agent.server.access import InstanceAccess, MAX_BODY
from agent.acceptance.runner import AcceptanceRun
from agent.acceptance.contracts import VerificationSpec

TOKEN = 'offline-instance-token-not-a-production-credential'


@pytest.fixture
def guarded():
    calls = []

    async def endpoint(request):
        calls.append(request.url.path)
        return JSONResponse({'ok': True})

    async def stream(request):
        async def chunks():
            yield b'first\n'
            yield b'second\n'

        return StreamingResponse(chunks(), media_type='text/event-stream')

    app = Starlette(
        middleware=[Middleware(InstanceAccess)],
        routes=[
            Route('/settings', endpoint, methods=['GET', 'POST']),
            Route('/threads', endpoint, methods=['POST']),
            Route('/stream', stream),
        ],
    )
    with RawClient(app, base_url='http://127.0.0.1:2024') as client:
        yield client, calls


def test_no_credentials_cannot_read_or_mutate_and_health_is_minimal(guarded):
    client, calls = guarded
    assert client.get('/settings').status_code == 401
    assert client.post('/threads', json={}).status_code == 401
    assert client.get('/ok').json() == {'ok': True}
    assert not calls


@pytest.mark.parametrize(
    'headers',
    [
        {'Origin': 'https://attacker.example'},
        {'Origin': 'null'},
        {'Host': 'attacker.example:2024'},
        {'Host': '127.0.0.1:9999'},
        {'Origin': 'http://127.0.0.1:3000/forged'},
    ],
)
def test_host_and_origin_cannot_bypass_authentication(guarded, headers):
    client, calls = guarded
    assert (
        client.post(
            '/threads', json={}, headers={'X-MonoPlanner-Service': TOKEN, **headers}
        ).status_code
        == 403
    )
    assert not calls


def test_body_type_size_and_chunked_requests_checked_before_execution(guarded):
    client, calls = guarded
    headers = {'X-MonoPlanner-Service': TOKEN}
    assert client.post('/threads', content='{}', headers=headers).status_code == 415
    assert (
        client.post(
            '/threads',
            content=(b'x' * 1024 for _ in range(MAX_BODY // 1024 + 1)),
            headers={**headers, 'Content-Type': 'application/json'},
        ).status_code
        == 413
    )
    assert not calls
    assert client.post('/threads', json={}, headers=headers).status_code == 200
    assert calls == ['/threads']


@pytest.mark.parametrize(
    'path',
    [
        '/docs',
        '/info',
        '/metrics',
        '/openapi.json',
        '/mcp',
        '/a2a',
        '/store/items',
        '/webhooks',
        '/studio',
        '/runs/crons',
        '/threads/one/runs/crons',
    ],
)
def test_unused_routes_remain_closed(guarded, path):
    client, calls = guarded
    assert client.get(path, headers={'X-MonoPlanner-Service': TOKEN}).status_code == 404
    assert not calls


def test_missing_instance_configuration_fails_closed(guarded, monkeypatch):
    monkeypatch.delenv('MONOPLANER_SERVICE_TOKEN')
    assert guarded[0].get('/settings').status_code == 503


def test_stream_is_not_buffered_or_rewritten_by_backend_gate(guarded):
    response = guarded[0].get('/stream', headers={'X-MonoPlanner-Service': TOKEN})
    assert response.status_code == 200
    assert response.text == 'first\nsecond\n'


def test_acceptance_cannot_default_to_host_execution(tmp_path):
    with pytest.raises(ValueError, match='host fallback is disabled'):
        AcceptanceRun(
            tmp_path,
            VerificationSpec(('target',), command='echo unreachable'),
            tmp_path / 'records',
        )
