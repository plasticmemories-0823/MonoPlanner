"""离线验证代理开关、真实 HTTP 路由和 SDK 接线，不改系统代理。"""

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from threading import Thread
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage

from agent.model.client import create_model
from agent.model.network import AsyncSystemProxyTransport, SystemProxyTransport, system_proxy_for


@contextmanager
def registry_settings(settings):
    def query_value(key, name):
        if name not in settings:
            raise FileNotFoundError(name)
        return settings[name], 0

    with patch("winreg.OpenKey"), patch("winreg.QueryValueEx", side_effect=query_value):
        yield


@unittest.skipUnless(os.name == "nt", "Windows 系统代理读取")
class WindowsProxyTests(unittest.TestCase):
    def test_disabled_system_proxy_ignores_inherited_proxy_without_changing_env(self):
        settings = {"ProxyEnable": 0, "ProxyServer": "127.0.0.1:8888"}
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1",
                                     "HTTPS_PROXY": "http://127.0.0.1:1",
                                     "ALL_PROXY": "http://127.0.0.1:1"}):
            before = dict(os.environ)
            with registry_settings(settings):
                self.assertIsNone(system_proxy_for(httpx.URL("https://api.deepseek.com/models")))
            self.assertEqual(dict(os.environ), before)

    def test_enabled_system_proxy_uses_current_address_for_each_protocol(self):
        settings = {"ProxyEnable": 1, "ProxyServer": "127.0.0.1:8888",
                    "ProxyOverride": ""}
        with registry_settings(settings):
            self.assertEqual(system_proxy_for(httpx.URL("https://example.com")),
                             "http://127.0.0.1:8888")
            settings["ProxyServer"] = "http=127.0.0.1:8111;https=127.0.0.1:8222"
            self.assertEqual(system_proxy_for(httpx.URL("http://example.com")),
                             "http://127.0.0.1:8111")
            self.assertEqual(system_proxy_for(httpx.URL("https://example.com")),
                             "http://127.0.0.1:8222")

    def test_bypass_comes_from_windows_settings_not_inherited_no_proxy(self):
        settings = {"ProxyEnable": 1, "ProxyServer": "127.0.0.1:8888",
                    "ProxyOverride": "*.internal;<local>"}
        with registry_settings(settings), patch.dict(os.environ, {"NO_PROXY": "example.com"}):
            self.assertIsNone(system_proxy_for(httpx.URL("http://service.internal")))
            self.assertIsNone(system_proxy_for(httpx.URL("http://printer")))
            self.assertEqual(system_proxy_for(httpx.URL("https://example.com")),
                             "http://127.0.0.1:8888")


@contextmanager
def recording_server(label):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.server.paths.append(self.path)
            body = label.encode("ascii")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.paths = []
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server.paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class HttpRoutingTests(unittest.TestCase):
    def test_same_client_switches_direct_proxy_direct_using_real_sockets(self):
        with recording_server("origin") as (origin, direct_paths), \
                recording_server("proxy") as (proxy, proxy_paths), \
                patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1"}), \
                patch("agent.model.network.system_proxy_for", side_effect=[None, proxy, None]):
            with httpx.Client(transport=SystemProxyTransport(), trust_env=False, timeout=2) as client:
                bodies = [client.get(f"{origin}/{step}").text for step in (1, 2, 3)]
            self.assertEqual(bodies, ["origin", "proxy", "origin"])
            self.assertEqual(direct_paths, ["/1", "/3"])
            self.assertEqual(proxy_paths, [f"{origin}/2"])

    def test_proxy_failure_is_reported_without_silent_direct_retry(self):
        def unavailable(request):
            raise httpx.ConnectError("offline proxy is unavailable", request=request)

        with patch("agent.model.network.system_proxy_for", return_value="http://127.0.0.1:1"), \
                patch("agent.model.network.httpx.HTTPTransport",
                      return_value=httpx.MockTransport(unavailable)) as factory:
            with httpx.Client(transport=SystemProxyTransport(), trust_env=False) as client:
                with self.assertRaises(httpx.ConnectError):
                    client.get("https://example.com")
            factory.assert_called_once_with(proxy="http://127.0.0.1:1", trust_env=False, retries=0)

    def test_existing_route_pools_are_reused_and_closed(self):
        created = []

        class Transport(httpx.MockTransport):
            def __init__(self, **kwargs):
                super().__init__(lambda request: httpx.Response(200, text="ok"))
                self.closed = False
                created.append(self)

            def close(self):
                self.closed = True

        with patch("agent.model.network.system_proxy_for", side_effect=[None, "http://localhost:8", None]), \
                patch("agent.model.network.httpx.HTTPTransport", Transport):
            with httpx.Client(transport=SystemProxyTransport(), trust_env=False) as client:
                for _ in range(3):
                    client.get("https://example.com")
            self.assertEqual(len(created), 2)
            self.assertTrue(all(transport.closed for transport in created))

    def test_real_sdk_uses_dynamic_transport_and_keeps_authentication_and_payload(self):
        routes, requests = [], []

        def transport(**kwargs):
            routes.append(kwargs["proxy"])

            def respond(request):
                requests.append(request)
                return httpx.Response(200, json={
                    "id": "offline", "object": "chat.completion", "created": 0,
                    "model": "deepseek-flash",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "OK"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                })
            return httpx.MockTransport(respond)

        with patch("agent.model.network.system_proxy_for", side_effect=[None, "http://localhost:8"]), \
                patch("agent.model.network.httpx.HTTPTransport", side_effect=transport), \
                patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:1"}):
            model = create_model({"model_name": "deepseek-flash", "api_key": "offline-placeholder"})
            try:
                for _ in range(2):
                    answer = model.invoke([HumanMessage(content="Reply OK")])
                    self.assertEqual(answer.content, "OK")
                    self.assertEqual(answer.usage_metadata["total_tokens"], 3)
            finally:
                model.client.http_client.close()
                asyncio.run(model.client.http_async_client.aclose())
        self.assertEqual(routes, [None, "http://localhost:8"])
        self.assertTrue(all(r.url.host == "api.deepseek.com" for r in requests))
        self.assertTrue(all(r.headers["authorization"] == "Bearer offline-placeholder" for r in requests))


class AsyncHttpRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_client_also_follows_switch_without_inherited_proxy(self):
        with recording_server("origin") as (origin, direct_paths), \
                recording_server("proxy") as (proxy, proxy_paths), \
                patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1"}), \
                patch("agent.model.network.system_proxy_for", side_effect=[None, proxy, None]):
            async with httpx.AsyncClient(transport=AsyncSystemProxyTransport(),
                                         trust_env=False, timeout=2) as client:
                bodies = [(await client.get(f"{origin}/{step}")).text for step in (1, 2, 3)]
            self.assertEqual(bodies, ["origin", "proxy", "origin"])
            self.assertEqual(direct_paths, ["/1", "/3"])
            self.assertEqual(proxy_paths, [f"{origin}/2"])


if __name__ == "__main__":
    unittest.main()
