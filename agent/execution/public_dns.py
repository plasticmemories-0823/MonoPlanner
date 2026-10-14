"""代理路线使用加密公网 DNS，避免本机污染答复破坏公网目标校验。"""

from collections import OrderedDict
import hashlib
import ipaddress
import json
import socket
import ssl
from threading import BoundedSemaphore, Lock
import time
from urllib.parse import urlencode

import httpcore
import httpx
from httpcore._backends.sync import SyncStream, TLSinTLSStream

from agent.execution.managed_proxy import validate_public_host


# 固定解析器 IP 不再需要先查一次本机 DNS；TLS 仍校验该 IP 的证书。
DNS_ENDPOINT = 'https://1.1.1.1/dns-query'
DNS_TIMEOUT = 20.0
MAX_RESPONSE = 64 * 1024
MAX_WIRE = 128 * 1024
MAX_CACHE = 256
MAX_TTL = 300
_cache = OrderedDict()
_cache_lock = Lock()
_queries = BoundedSemaphore(8)


def _remaining(deadline, cancelled):
    if cancelled and cancelled():
        raise OSError('Public DNS query cancelled')
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OSError('Public DNS query timed out')
    return remaining


class _DeadlineSocket:
    """内层 TLS 每次收发都重新核对总时限，不能靠零碎数据无限延长。"""

    def __init__(self, raw, deadline, cancelled):
        self.raw, self.deadline, self.cancelled = raw, deadline, cancelled

    def recv(self, size):
        self.raw.settimeout(_remaining(self.deadline, self.cancelled))
        return self.raw.recv(size)

    def sendall(self, data):
        self.raw.settimeout(_remaining(self.deadline, self.cancelled))
        return self.raw.sendall(data)

    def settimeout(self, timeout):
        self.raw.settimeout(_remaining(self.deadline, self.cancelled))

    def __getattr__(self, name):
        return getattr(self.raw, name)


class _ObservedStream(httpcore.NetworkStream):
    """注册握手前后的 socket，让关闭准备服务能够打断当前 DNS 请求。"""

    def __init__(self, stream, deadline, cancelled, on_socket):
        self.stream = stream
        self.deadline = deadline
        self.cancelled = cancelled
        self.on_socket = on_socket
        self.received = 0

    def read(self, max_bytes, timeout=None):
        timeout = _remaining(self.deadline, self.cancelled)
        data = self.stream.read(min(max_bytes, MAX_WIRE + 1), timeout)
        self.received += len(data)
        if self.received > MAX_WIRE:
            raise OSError('Public DNS response too large')
        _remaining(self.deadline, self.cancelled)
        return data

    def write(self, buffer, timeout=None):
        self.stream.write(buffer, _remaining(self.deadline, self.cancelled))
        _remaining(self.deadline, self.cancelled)

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        try:
            timeout = _remaining(self.deadline, self.cancelled)
            raw = self.stream.get_extra_info('socket')
            if isinstance(raw, ssl.SSLSocket):
                # httpcore 的 MemoryBIO 实现保留 HTTPS 代理外层 TLS。
                self.stream = TLSinTLSStream(
                    _DeadlineSocket(raw, self.deadline, self.cancelled),
                    ssl_context, server_hostname, timeout,
                )
            else:
                wrapped = ssl_context.wrap_socket(
                    raw, server_hostname=server_hostname, do_handshake_on_connect=False,
                )
                self.stream = SyncStream(wrapped)
                if self.on_socket:
                    self.on_socket(wrapped)
                wrapped.settimeout(_remaining(self.deadline, self.cancelled))
                wrapped.do_handshake()
            _remaining(self.deadline, self.cancelled)
            return self
        except BaseException:
            # TLS 转换会转移原 socket 所有权，失败时关闭当前流，不能等准备任务结束。
            self.close()
            raise

    def get_extra_info(self, info):
        return self.stream.get_extra_info(info)

    def close(self):
        self.stream.close()


class _ObservedBackend(httpcore.NetworkBackend):
    def __init__(self, deadline, cancelled, on_socket):
        self.deadline = deadline
        self.cancelled = cancelled
        self.on_socket = on_socket

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        _remaining(self.deadline, self.cancelled)
        # 此处只连接已选代理。目标域名和解析器本身均不交给系统 DNS。
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            family, kind, protocol, _, target = socket.getaddrinfo(
                host, port, type=socket.SOCK_STREAM,
            )[0]
        else:
            family = socket.AF_INET if address.version == 4 else socket.AF_INET6
            kind, protocol = socket.SOCK_STREAM, socket.IPPROTO_TCP
            target = (str(address), port) if address.version == 4 else (str(address), port, 0, 0)
        raw = socket.socket(family, kind, protocol)
        try:
            if self.on_socket:
                self.on_socket(raw)
            raw.settimeout(_remaining(self.deadline, self.cancelled))
            raw.connect(target)
            _remaining(self.deadline, self.cancelled)
            return _ObservedStream(
                SyncStream(raw), self.deadline, self.cancelled, self.on_socket,
            )
        except BaseException:
            raw.close()
            raise


def _query(host, record_type, proxy, deadline, cancelled, on_socket):
    route = httpx.Proxy(proxy)
    if route.url.scheme not in {'http', 'https'}:
        raise ValueError('Public DNS requires the selected HTTP(S) proxy')
    backend = _ObservedBackend(deadline, cancelled, on_socket)
    with httpcore.HTTPProxy(
        proxy_url=httpcore.URL(
            scheme=route.url.raw_scheme, host=route.url.raw_host,
            port=route.url.port, target=route.url.raw_path,
        ),
        proxy_auth=route.raw_auth,
        network_backend=backend,
        retries=0,
        max_connections=1,
        max_keepalive_connections=0,
    ) as pool:
        # 请求只含主机名和记录类型；原始 URL、用户请求头和令牌不进入 DNS 请求。
        url = DNS_ENDPOINT + '?' + urlencode({'name': host, 'type': record_type})
        remaining = _remaining(deadline, cancelled)
        with pool.stream(
            'GET', url,
            headers={'Accept': 'application/dns-json', 'Accept-Encoding': 'identity'},
            extensions={'timeout': dict.fromkeys(('connect', 'read', 'write', 'pool'), remaining)},
        ) as response:
            headers = {key.lower(): value for key, value in response.headers}
            if (
                response.status != 200
                or headers.get(b'content-type', b'').split(b';')[0].strip()
                != b'application/dns-json'
                or headers.get(b'content-encoding', b'identity') != b'identity'
            ):
                raise ValueError('Invalid public DNS response')
            data = bytearray()
            for chunk in response.iter_stream():
                _remaining(deadline, cancelled)
                data.extend(chunk)
                if len(data) > MAX_RESPONSE:
                    raise ValueError('Public DNS response too large')
            _remaining(deadline, cancelled)
            return json.loads(data)


def _answer(payload, host, record_type):
    expected = 1 if record_type == 'A' else 28
    if not isinstance(payload, dict) or type(payload.get('Status')) is not int:
        raise ValueError('Invalid public DNS response')
    question = payload.get('Question')
    if (
        payload['Status'] != 0 or payload.get('TC', False) is not False
        or not isinstance(question, list) or len(question) != 1
        or not isinstance(question[0], dict)
        or type(question[0].get('type')) is not int
        or question[0].get('type') != expected
        or validate_public_host(question[0].get('name')) != host
    ):
        raise ValueError('Public DNS query did not resolve the requested host')
    answers = payload.get('Answer', [])
    if not isinstance(answers, list) or len(answers) > 128:
        raise ValueError('Invalid public DNS records')
    aliases, addresses, ttl = {}, [], MAX_TTL
    for record in answers:
        if not isinstance(record, dict) or type(record.get('type')) is not int:
            raise ValueError('Invalid public DNS record')
        name = validate_public_host(record.get('name'))
        record_ttl = record.get('TTL')
        if type(record_ttl) is not int or record_ttl < 0:
            raise ValueError('Invalid public DNS TTL')
        ttl = min(ttl, record_ttl)
        kind, value = record['type'], record.get('data')
        if kind == 5:
            target = validate_public_host(value)
            if name in aliases and aliases[name] != target:
                raise ValueError('Ambiguous public DNS alias')
            aliases[name] = target
        elif kind in (1, 28):
            # 连无关答复也不能夹带私网地址；绝不把它们当作已验证目标。
            if not isinstance(value, str):
                raise ValueError('Invalid public DNS address')
            address = ipaddress.ip_address(value)
            validate_public_host(str(address))
            if address.version != (4 if kind == 1 else 6):
                raise ValueError('Invalid public DNS address type')
            addresses.append((name, kind, str(address)))
        else:
            raise ValueError('Unexpected public DNS record type')
    current, visited = host, set()
    while current in aliases:
        if current in visited or len(visited) >= 16:
            raise ValueError('Invalid public DNS alias chain')
        visited.add(current)
        current = aliases[current]
    if any(name in aliases for name, _, _ in addresses):
        raise ValueError('Ambiguous public DNS alias address')
    return [value for name, kind, value in addresses if name == current and kind == expected], ttl


def _socket_addresses(ips, port):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, port))
        if ipaddress.ip_address(ip).version == 4 else
        (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, port, 0, 0))
        for ip in ips
    ]


def proxy_public_addresses(host, port, proxy, *, cancelled=None, on_socket=None):
    """经指定代理解析公网主机；失败不改走本机 DNS 或其他路线。"""
    host = validate_public_host(host)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid public destination port')
    deadline = time.monotonic() + DNS_TIMEOUT
    _remaining(deadline, cancelled)
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return _socket_addresses([str(literal)], port)

    # 代理凭据不作为明文缓存键，也不出现在异常与日志里。
    key = (hashlib.sha256(proxy.encode('utf-8')).digest(), host)
    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached[0] > time.monotonic():
            _cache.move_to_end(key)
            return _socket_addresses(cached[1], port)
        _cache.pop(key, None)
    if not _queries.acquire(blocking=False):
        raise OSError('Public DNS query capacity exceeded')
    try:
        ips, ttl = [], MAX_TTL
        for kind in ('A', 'AAAA'):
            _remaining(deadline, cancelled)
            payload = _query(host, kind, proxy, deadline, cancelled, on_socket)
            values, answer_ttl = _answer(payload, host, kind)
            ips.extend(values)
            ttl = min(ttl, answer_ttl)
        _remaining(deadline, cancelled)
        ips = tuple(dict.fromkeys(ips))
        if not ips:
            raise ValueError('Public DNS returned no public address')
        if ttl > 0:
            with _cache_lock:
                _cache[key] = (time.monotonic() + ttl, ips)
                _cache.move_to_end(key)
                while len(_cache) > MAX_CACHE:
                    _cache.popitem(last=False)
        return _socket_addresses(ips, port)
    except (
        OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError,
        httpcore.NetworkError, httpcore.TimeoutException, httpcore.ProtocolError,
        httpcore.ProxyError, httpcore.UnsupportedProtocol,
    ):
        # 上游异常可能包含代理地址或凭据，诊断只保留固定受控文字。
        raise OSError('Public DNS resolution failed or was cancelled') from None
    finally:
        _queries.release()
