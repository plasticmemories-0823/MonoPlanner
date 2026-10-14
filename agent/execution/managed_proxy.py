"""只在隔离的环境准备辅助容器中运行公开下载代理。

安装容器仅接入 Docker 内部网络。代理负责解析 DNS，并使用已校验的
数值 IP 地址建立真实连接；不转发代理认证、本机凭据或任意请求头。
"""

import http.server
import base64
import ipaddress
import os
import select
import socket
import socketserver
import time
from urllib.parse import unquote, urlsplit

TIMEOUT = 30
MAX_TUNNEL_SECONDS = 600
MAX_TRANSFER = 512 * 1024 * 1024


def public_addresses(host, port):
    if not host or host.lower().rstrip('.').endswith(('.localhost', '.local', '.internal')):
        raise ValueError('private host')
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(info[4][0]).is_global
        or ipaddress.ip_address(info[4][0]).is_multicast
        or ipaddress.ip_address(info[4][0]).is_reserved
        for info in addresses
    ):
        raise ValueError('private address')
    return addresses


def tunnel(proxy, host, port, *, target_url=None, on_socket=None, cancelled=None):
    """连接受信任代理，但只将固定后的目标地址交给 CONNECT。"""
    import ssl

    parsed = urlsplit(proxy)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        raise ValueError('unsupported proxy')
    addresses = socket.getaddrinfo(
        parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80),
        type=socket.SOCK_STREAM,
    )
    if cancelled and cancelled():
        raise OSError('preparation relay closed')
    family, kind, protocol, _, address = addresses[0]
    connection = socket.socket(family, kind, protocol)
    try:
        if on_socket:
            on_socket(connection)
        connection.settimeout(TIMEOUT)
        connection.connect(address)
        if parsed.scheme == 'https':
            connection = ssl.create_default_context().wrap_socket(
                connection, server_hostname=parsed.hostname, do_handshake_on_connect=False
            )
            if on_socket:
                on_socket(connection)
            connection.do_handshake()
        authority = f'[{host}]:{port}' if ':' in host else f'{host}:{port}'
        request = f'CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n'
        if parsed.username is not None:
            credentials = unquote(parsed.username) + ':' + unquote(parsed.password or '')
            encoded = base64.b64encode(credentials.encode('utf-8')).decode('ascii')
            request += 'Proxy-Authorization: Basic ' + encoded + '\r\n'
        if target_url:
            if any(char in target_url for char in '\r\n'):
                raise ValueError('invalid target')
            request += 'X-MonoPlanner-Download-URL: ' + target_url + '\r\n'
        connection.sendall((request + '\r\n').encode('ascii'))
        # 不让有缓冲的 HTTP 解析器提前吞掉随后交给客户端的 TLS 数据。
        response = bytearray()
        deadline = time.monotonic() + TIMEOUT
        while not response.endswith(b'\r\n\r\n') and len(response) < 16384:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or (cancelled and cancelled()):
                raise OSError('download proxy handshake interrupted')
            connection.settimeout(remaining)
            chunk = connection.recv(1)
            if not chunk:
                break
            response.extend(chunk)
        status = bytes(response).split(b'\r\n', 1)[0].split(b' ')
        if not response.endswith(b'\r\n\r\n') or len(status) < 2 or status[1] != b'200':
            raise OSError('download proxy connection failed')
        connection.settimeout(TIMEOUT)
        return connection
    except BaseException:
        connection.close()
        raise


def connect_public(host, port, target_url=None):
    addresses = public_addresses(host, port)
    relay = os.getenv('MONOPLANNER_DOWNLOAD_RELAY')
    if relay:
        # 安装容器只接触一次性中继口令；系统代理及凭据留在宿主进程。
        return tunnel(relay, host, port, target_url=target_url)
    last = None
    for family, kind, proto, _, address in addresses:
        connection = socket.socket(family, kind, proto)
        try:
            connection.settimeout(TIMEOUT)
            connection.connect(address)
            return connection
        except OSError as exc:
            last = exc
            connection.close()
    raise last or OSError('connection failed')


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    timeout = TIMEOUT

    def log_message(self, *args):
        # URL 可能包含依赖名称或签名令牌，不能写入日志。
        pass

    def open_public(self, host, port, target_url):
        connector = getattr(self.server, 'connect_public', connect_public)
        return connector(host, port, target_url)

    def do_CONNECT(self):
        upstream = None
        try:
            parsed = urlsplit('//' + self.path)
            ports = getattr(self.server, 'tunnel_ports', (443,))
            if parsed.username or parsed.password or parsed.port not in ports or parsed.path:
                raise ValueError('only public HTTPS downloads are permitted')
            scheme = 'https' if parsed.port == 443 else 'http'
            target_url = scheme + '://' + self.path + '/'
            supplied_url = self.headers.get('X-MonoPlanner-Download-URL')
            if supplied_url and getattr(self.server, 'accept_target_url', False):
                supplied = urlsplit(supplied_url)
                if (supplied.scheme != scheme or supplied.hostname != parsed.hostname
                        or (supplied.port or (443 if scheme == 'https' else 80)) != parsed.port
                        or supplied.username or supplied.password):
                    raise ValueError('target identity mismatch')
                target_url = supplied_url
            upstream = self.open_public(parsed.hostname, parsed.port, target_url)
            self.send_response(200, 'Connection Established')
            self.end_headers()
            self.wfile.flush()
            sockets = [self.connection, upstream]
            deadline, transferred = time.monotonic() + MAX_TUNNEL_SECONDS, 0
            while time.monotonic() < deadline and transferred < MAX_TRANSFER:
                readable = [source for source in sockets
                            if getattr(source, 'pending', lambda: 0)()]
                if not readable:
                    readable, _, _ = select.select(sockets, [], [], TIMEOUT)
                if not readable:
                    break
                for source in readable:
                    chunk = source.recv(65536)
                    if not chunk:
                        return
                    transferred += len(chunk)
                    (upstream if source is self.connection else self.connection).sendall(chunk)
        except (OSError, ValueError):
            if upstream is None:
                self.send_error(403, 'Public download destination required')
        finally:
            self.close_connection = True
            if upstream is not None:
                upstream.close()

    def do_GET(self):
        upstream = None
        try:
            parsed = urlsplit(self.path)
            if (
                parsed.scheme != 'http'
                or parsed.port not in (None, 80)
                or parsed.username
                or parsed.password
                or self.headers.get('Transfer-Encoding')
                or self.headers.get('Content-Length') not in (None, '0')
            ):
                raise ValueError('only public HTTP downloads are permitted')
            upstream = self.open_public(parsed.hostname, 80, self.path)
            target = (parsed.path or '/') + ('?' + parsed.query if parsed.query else '')
            request = f'{self.command} {target} HTTP/1.1\r\nHost: {parsed.hostname}\r\nConnection: close\r\n'
            for key in ('Accept', 'User-Agent', 'Range', 'If-None-Match', 'If-Modified-Since'):
                value = self.headers.get(key)
                if value:
                    request += key + ': ' + value + '\r\n'
            upstream.sendall((request + '\r\n').encode('latin-1'))
            transferred, deadline = 0, time.monotonic() + MAX_TUNNEL_SECONDS
            while transferred < MAX_TRANSFER and time.monotonic() < deadline:
                chunk = upstream.recv(65536)
                if not chunk:
                    break
                transferred += len(chunk)
                self.connection.sendall(chunk)
        except (OSError, ValueError):
            if upstream is None:
                self.send_error(403, 'Public download destination required')
        finally:
            self.close_connection = True
            if upstream is not None:
                upstream.close()

    do_HEAD = do_GET


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def server_bind(self):
        # 代理不需要主机反向 DNS；默认 HTTPServer 的 getfqdn 会拖住离线启动。
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


if __name__ == '__main__':
    Server(('0.0.0.0', 3128), Handler).serve_forever()
