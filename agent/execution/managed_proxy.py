"""Public-download proxy run only inside the isolated preparation sidecar.

The setup container has only an internal Docker network. DNS is resolved by
this proxy and the validated numeric address is used for the actual connection.
No proxy authentication, host credentials, or arbitrary forwarding headers exist.
"""
import http.server
import ipaddress
import select
import socket
import socketserver
import time
from urllib.parse import urlsplit

TIMEOUT = 30
MAX_TUNNEL_SECONDS = 600
MAX_TRANSFER = 512 * 1024 * 1024


def connect_public(host, port):
    if not host or host.lower().rstrip('.').endswith(('.localhost', '.local', '.internal')):
        raise ValueError('private host')
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(info[4][0]).is_global
                            or ipaddress.ip_address(info[4][0]).is_multicast
                            or ipaddress.ip_address(info[4][0]).is_reserved for info in addresses):
        raise ValueError('private address')
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
        # URLs may contain dependency names or signed tokens. Never log them.
        pass

    def do_CONNECT(self):
        upstream = None
        try:
            parsed = urlsplit('//' + self.path)
            if parsed.username or parsed.password or parsed.port != 443 or parsed.path:
                raise ValueError('only public HTTPS downloads are permitted')
            upstream = connect_public(parsed.hostname, 443)
            self.send_response(200, 'Connection Established')
            self.end_headers()
            self.wfile.flush()
            sockets = [self.connection, upstream]
            deadline, transferred = time.monotonic() + MAX_TUNNEL_SECONDS, 0
            while time.monotonic() < deadline and transferred < MAX_TRANSFER:
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
            if (parsed.scheme != 'http' or parsed.port not in (None, 80)
                    or parsed.username or parsed.password or self.headers.get('Transfer-Encoding')
                    or self.headers.get('Content-Length') not in (None, '0')):
                raise ValueError('only public HTTP downloads are permitted')
            upstream = connect_public(parsed.hostname, 80)
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


if __name__ == '__main__':
    Server(('0.0.0.0', 3128), Handler).serve_forever()
