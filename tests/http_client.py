import os
from starlette.testclient import TestClient as StarletteTestClient

TEST_TOKEN = 'offline-instance-token-not-a-production-credential'


class TestClient(StarletteTestClient):
    def __init__(self, app, **kwargs):
        os.environ.setdefault('MONOPLANER_SERVICE_TOKEN', TEST_TOKEN)
        port = os.getenv('MONOPLANER_BACKEND_PORT', '2024')
        kwargs.setdefault('base_url', 'http://127.0.0.1:' + port)
        headers = {'X-MonoPlanner-Service': os.environ['MONOPLANER_SERVICE_TOKEN']}
        headers.update(kwargs.pop('headers', None) or {})
        super().__init__(app, headers=headers, **kwargs)
