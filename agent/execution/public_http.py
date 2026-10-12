"""Public-page transport with validation bound to the actual TCP destination."""
import ipaddress
import socket
from urllib.parse import urlsplit

import httpx


def resolve_public(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username or parsed.password):
        raise ValueError('url must be a public HTTP(S) URL without credentials')
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80),
                                   type=socket.SOCK_STREAM)
    ips = list(dict.fromkeys(row[4][0] for row in addresses))
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError('Private, loopback and local-network pages are not permitted')
    return parsed, ips


class PublicTransport(httpx.BaseTransport):
    def __init__(self, transport=None):
        # No ambient proxy: a proxy that resolves the hostname would undo IP pinning.
        self.transport = transport or httpx.HTTPTransport(retries=0, trust_env=False)

    def handle_request(self, request):
        parsed, ips = resolve_public(str(request.url))
        original = request.url
        pinned = original.copy_with(host=ips[0])
        headers = httpx.Headers(request.headers)
        headers['Host'] = original.netloc.decode('ascii')
        # httpcore uses this extension for TLS SNI and certificate hostname checks.
        extensions = {**request.extensions, 'sni_hostname': parsed.hostname}
        response = self.transport.handle_request(httpx.Request(
            request.method, pinned, headers=headers, stream=request.stream, extensions=extensions))
        return response

    def close(self):
        self.transport.close()


def public_client(timeout):
    return httpx.Client(transport=PublicTransport(), trust_env=False, timeout=timeout)
