"""读取公开网页，并将地址校验结果绑定到真实 TCP 或代理隧道目标。"""

from contextlib import contextmanager
import ipaddress
import socket
from threading import Lock
from urllib.parse import urlsplit

import httpcore
import httpx

from agent.model.network import system_proxy_for


def resolve_public(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {'http', 'https'}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError('url must be a public HTTP(S) URL without credentials')
    addresses = socket.getaddrinfo(
        parsed.hostname,
        parsed.port or (443 if parsed.scheme == 'https' else 80),
        type=socket.SOCK_STREAM,
    )
    ips = list(dict.fromkeys(row[4][0] for row in addresses))
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError('Private, loopback and local-network pages are not permitted')
    return parsed, ips


class _OriginTLSStream(httpcore.NetworkStream):
    """CONNECT 固定公网 IP；仅目标 TLS 使用原域名，代理自己的 TLS 不受影响。"""

    def __init__(self, stream, address, hostname, proxy_tls=False):
        self.stream = stream
        self.address = address
        self.hostname = hostname
        self.proxy_tls = proxy_tls

    def read(self, max_bytes, timeout=None):
        return self.stream.read(max_bytes, timeout)

    def write(self, buffer, timeout=None):
        return self.stream.write(buffer, timeout)

    def close(self):
        self.stream.close()

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        hostname = server_hostname
        if not self.proxy_tls and server_hostname == self.address:
            hostname = self.hostname
        self.stream = self.stream.start_tls(ssl_context, hostname, timeout)
        self.proxy_tls = False
        return self

    def get_extra_info(self, info):
        return self.stream.get_extra_info(info)


class _OriginTLSBackend(httpcore.NetworkBackend):
    def __init__(self, address, hostname, proxy_tls=False):
        self.backend = httpcore.SyncBackend()
        self.address = address
        self.hostname = hostname
        self.proxy_tls = proxy_tls

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        stream = self.backend.connect_tcp(
            host, port, timeout=timeout,
            local_address=local_address, socket_options=socket_options,
        )
        return _OriginTLSStream(stream, self.address, self.hostname, self.proxy_tls)


@contextmanager
def _http_errors():
    # 不将代理地址或用户名密码拼进错误；请求和流读取都使用 HTTPX 的异常层。
    try:
        yield
    except httpcore.TimeoutException as exc:
        raise httpx.TimeoutException('Public page connection timed out') from exc
    except httpcore.ProxyError as exc:
        raise httpx.ProxyError('Public page proxy connection failed') from exc
    except (httpcore.NetworkError, httpcore.ProtocolError) as exc:
        raise httpx.TransportError('Public page connection failed') from exc


class _ProxyResponseStream(httpx.SyncByteStream):
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        with _http_errors():
            yield from self.stream

    def close(self):
        self.stream.close()


class _PinnedProxyTransport(httpx.BaseTransport):
    def __init__(self, proxy, address, hostname):
        route = httpx.Proxy(proxy)
        self.pool = httpcore.HTTPProxy(
            proxy_url=httpcore.URL(
                scheme=route.url.raw_scheme,
                host=route.url.raw_host,
                port=route.url.port,
                target=route.url.raw_path,
            ),
            proxy_auth=route.raw_auth,
            network_backend=_OriginTLSBackend(address, hostname, route.url.scheme == 'https'),
            retries=0,
        )

    def handle_request(self, request):
        # httpcore 的代理连接自身也有 TLS；目标域名由专用 backend 还原。
        extensions = dict(request.extensions)
        extensions.pop('sni_hostname', None)
        with _http_errors():
            response = self.pool.handle_request(httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
                extensions=extensions,
            ))
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_ProxyResponseStream(response.stream),
            extensions=response.extensions,
        )

    def close(self):
        self.pool.close()


class PublicTransport(httpx.BaseTransport):
    def __init__(self, transport=None):
        self.transport = transport or httpx.HTTPTransport(retries=0, trust_env=False)
        self._injected = transport is not None
        self._proxy_transports = {}
        self._lock = Lock()

    def handle_request(self, request):
        parsed, ips = resolve_public(str(request.url))
        original = request.url
        pinned = original.copy_with(host=ips[0])
        headers = httpx.Headers(request.headers)
        headers['Host'] = original.netloc.decode('ascii')
        extensions = {**request.extensions, 'sni_hostname': parsed.hostname}
        transport = self.transport
        # 代理规则针对原 URL 选择；连接目标仍是已校验 IP，不能交给代理重新解析。
        proxy = None if self._injected else system_proxy_for(original)
        if proxy:
            key = (proxy, ips[0], parsed.hostname)
            with self._lock:
                if key not in self._proxy_transports:
                    self._proxy_transports[key] = _PinnedProxyTransport(*key)
                transport = self._proxy_transports[key]
        return transport.handle_request(httpx.Request(
            request.method,
            pinned,
            headers=headers,
            stream=request.stream,
            extensions=extensions,
        ))

    def close(self):
        self.transport.close()
        with self._lock:
            transports = list(self._proxy_transports.values())
            self._proxy_transports.clear()
        for transport in transports:
            transport.close()


def public_client(timeout):
    return httpx.Client(transport=PublicTransport(), trust_env=False, timeout=timeout)
