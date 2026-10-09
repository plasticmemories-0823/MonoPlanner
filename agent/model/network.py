"""模型请求跟随系统代理；Windows 不继承启动器固定的代理环境变量。"""

import os
from threading import Lock
import urllib.request

import httpx


def system_proxy_for(url: httpx.URL) -> str | None:
    """每次请求读取当前开关，并沿用系统的协议选择和绕过名单。"""
    if os.name == "nt":
        # getproxies() 会优先使用环境变量，无法反映 Windows 当前开关。
        proxies = urllib.request.getproxies_registry()
        bypass = urllib.request.proxy_bypass_registry
    else:
        proxies = urllib.request.getproxies()
        bypass = urllib.request.proxy_bypass
    proxy = proxies.get(url.scheme) or proxies.get("all")
    return None if not proxy or bypass(url.netloc.decode("ascii")) else proxy


class _ProxyPools:
    """每条路线复用连接池；开关变化只影响随后发出的请求。"""

    def __init__(self, transport_class):
        self._transport_class = transport_class
        self._pools = {}
        self._lock = Lock()

    def for_request(self, request):
        proxy = system_proxy_for(request.url)
        with self._lock:
            if proxy not in self._pools:
                self._pools[proxy] = self._transport_class(
                    proxy=proxy, trust_env=False, retries=0,
                )
            return self._pools[proxy]

    def pop_all(self):
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
            return pools


class SystemProxyTransport(httpx.BaseTransport):
    def __init__(self):
        self._routes = _ProxyPools(httpx.HTTPTransport)

    def handle_request(self, request):
        return self._routes.for_request(request).handle_request(request)

    def close(self):
        for transport in self._routes.pop_all():
            transport.close()


class AsyncSystemProxyTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self._routes = _ProxyPools(httpx.AsyncHTTPTransport)

    async def handle_async_request(self, request):
        return await self._routes.for_request(request).handle_async_request(request)

    async def aclose(self):
        for transport in self._routes.pop_all():
            await transport.aclose()


def model_http_clients(timeout):
    """隔离本项目的 HTTP 客户端，不改全局环境或系统设置。"""
    return {
        "http_client": httpx.Client(
            transport=SystemProxyTransport(), trust_env=False, timeout=timeout,
        ),
        "http_async_client": httpx.AsyncClient(
            transport=AsyncSystemProxyTransport(), trust_env=False, timeout=timeout,
        ),
    }
