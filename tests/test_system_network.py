"""默认网络、原生 PAC 与网页代理的离线回归；不修改系统设置或访问模型。"""

import asyncio
from contextlib import contextmanager
import ctypes
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import socket
import ssl
from threading import Event, Thread
import time
from unittest.mock import Mock, patch

import httpcore
import httpx
import pytest

from agent.execution.public_http import PublicTransport
from agent.model import network


@contextmanager
def local_server(handler):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_windows_pac_is_resolved_for_each_original_url_and_never_uses_old_environment():
    settings = {'auto_detect': False, 'config_url': 'http://pac.test/rules',
                'proxy': 'ignored:80', 'bypass': ''}
    urls = [httpx.URL('https://api.deepseek.com/chat/completions'),
            httpx.URL('https://example.com/page')]
    with patch.object(network, '_windows_settings', return_value=settings), patch.object(
        network, '_windows_auto_proxy', side_effect=[('127.0.0.1:8123', ''), ('', '')]
    ) as resolver, patch.dict(os.environ, {'HTTPS_PROXY': 'http://old:9000'}):
        assert [network._windows_proxy_for(url) for url in urls] == ['http://127.0.0.1:8123', None]
    assert [call.args[0] for call in resolver.call_args_list] == urls


def test_pac_failure_does_not_construct_transport_or_retry_direct():
    with patch.object(network, 'system_proxy_for', side_effect=httpx.ProxyError('PAC failed')), \
         patch.object(network.httpx, 'HTTPTransport') as factory:
        with httpx.Client(transport=network.SystemProxyTransport(), trust_env=False) as client:
            with pytest.raises(httpx.ProxyError, match='PAC failed'):
                client.get('https://api.deepseek.com/chat/completions')
        factory.assert_not_called()


def test_pac_wait_is_bounded_and_late_resolution_does_not_send_request():
    release, entered = Event(), Event()

    def blocked(*args):
        entered.set()
        release.wait(2)
        return 'proxy.invalid:80', ''

    try:
        with patch.object(network, '_PAC_TIMEOUT', 0.02), patch.object(
            network, '_windows_resolve_auto_proxy', side_effect=blocked
        ), patch.object(network, '_windows_settings', return_value={
            'auto_detect': False, 'config_url': 'http://pac.invalid', 'proxy': '', 'bypass': ''
        }), patch.object(network.httpx, 'HTTPTransport') as factory:
            start = time.monotonic()
            with pytest.raises(httpx.ProxyError, match='timed out'):
                network._windows_proxy_for(httpx.URL('https://example.com'))
            assert time.monotonic() - start < 0.5
            assert entered.is_set()
            factory.assert_not_called()
    finally:
        release.set()


def test_async_cancellation_during_route_lookup_never_dispatches_afterwards():
    async def check():
        entered, release = Event(), Event()

        def blocked(url):
            entered.set()
            release.wait(2)
            return None

        with patch.object(network, 'system_proxy_for', side_effect=blocked), patch.object(
            network.httpx, 'AsyncHTTPTransport'
        ) as factory:
            async with httpx.AsyncClient(transport=network.AsyncSystemProxyTransport()) as client:
                task = asyncio.create_task(client.get('https://example.com'))
                assert await asyncio.to_thread(entered.wait, 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                release.set()
                await asyncio.sleep(0.02)
                factory.assert_not_called()

    asyncio.run(check())

@pytest.mark.parametrize('proxy', ['socks5://127.0.0.1:1080', 'http://proxy/a', 'http://'])
def test_unsupported_proxy_has_explicit_error_without_echoing_credentials(proxy):
    with pytest.raises(httpx.ProxyError):
        network._validated_proxy(proxy)


@pytest.mark.skipif(os.name != 'nt', reason='Windows WinHTTP 原生调用')
def test_native_pac_selects_proxy_or_direct_without_changing_system_settings():
    class PAC(BaseHTTPRequestHandler):
        def do_GET(self):
            body = (b'function FindProxyForURL(url, host) {'
                    b' if (host == "api.deepseek.com") return "PROXY 127.0.0.1:8123";'
                    b' return "DIRECT"; }')
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ns-proxy-autoconfig')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with local_server(PAC) as server:
        settings = {'config_url': f'http://127.0.0.1:{server.server_port}/rules.pac',
                    'auto_detect': False, 'proxy': '', 'bypass': ''}
        route, _ = network._windows_auto_proxy(httpx.URL('https://api.deepseek.com'), settings)
        assert route == '127.0.0.1:8123'
        assert network._windows_auto_proxy(httpx.URL('https://example.com'), settings) == ('', '')


@pytest.mark.skipif(os.name != 'nt', reason='Windows WinHTTP 原生结构')
def test_winhttp_owned_strings_and_session_are_released_and_credentials_not_sent():
    buffers = [ctypes.create_unicode_buffer(value) for value in (
        'http://pac.example/rules', 'proxy.example:8080', '*.internal', 'selected:8888', ''
    )]
    pointers = [ctypes.addressof(buffer) for buffer in buffers]
    winhttp, kernel = Mock(), Mock()

    def user_config(pointer):
        value = pointer._obj
        value.auto_detect = False
        value.config_url, value.proxy, value.bypass = pointers[:3]
        return True

    def proxy_for_url(session, url, options_pointer, info_pointer):
        assert options_pointer._obj.auto_logon == 0
        assert options_pointer._obj.flags == 2
        info_pointer._obj.access_type = 3
        info_pointer._obj.proxy, info_pointer._obj.bypass = pointers[3:]
        return True

    winhttp.WinHttpGetIEProxyConfigForCurrentUser.side_effect = user_config
    winhttp.WinHttpOpen.return_value = 123
    winhttp.WinHttpSetTimeouts.return_value = True
    winhttp.WinHttpGetProxyForUrl.side_effect = proxy_for_url
    with patch.object(network, '_windows_api', return_value=(winhttp, kernel)):
        settings = network._windows_settings()
        assert network._windows_resolve_auto_proxy(
            httpx.URL('https://example.com'), settings
        ) == ('selected:8888', '')
    assert [call.args[0] for call in kernel.GlobalFree.call_args_list] == pointers
    winhttp.WinHttpCloseHandle.assert_called_once_with(123)


@pytest.mark.skipif(os.name != 'nt', reason='Windows WinHTTP 错误码')
@pytest.mark.parametrize('error,explicit,allowed', [(12180, False, True), (12180, True, False),
                                                 (12167, False, False), (12166, True, False)])
def test_only_absent_wpad_may_use_current_manual_route(error, explicit, allowed):
    winhttp, kernel = Mock(), Mock()
    winhttp.WinHttpOpen.return_value = 123
    winhttp.WinHttpSetTimeouts.return_value = True
    winhttp.WinHttpGetProxyForUrl.return_value = False
    settings = {'config_url': 'http://pac.example' if explicit else '',
                'auto_detect': True, 'proxy': 'manual:80', 'bypass': '*.internal'}
    with patch.object(network, '_windows_api', return_value=(winhttp, kernel)), patch.object(
        ctypes, 'get_last_error', return_value=error
    ):
        if allowed:
            assert network._windows_resolve_auto_proxy(
                httpx.URL('https://example.com'), settings
            ) == ('manual:80', '*.internal')
        else:
            with pytest.raises(httpx.ProxyError):
                network._windows_resolve_auto_proxy(httpx.URL('https://example.com'), settings)
    winhttp.WinHttpCloseHandle.assert_called_once_with(123)


def test_real_proxy_connect_pins_public_ip_but_tls_and_http_keep_original_host(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'example.com')])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName('example.com')]), False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / 'test-cert.pem', tmp_path / 'test-key.pem'
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                         serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
    server_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_tls.load_cert_chain(cert_path, key_path)
    observed = {'connect': [], 'sni': [], 'request': []}
    server_tls.set_servername_callback(lambda sock, name, context: observed['sni'].append(name))
    client_tls = ssl.create_default_context(cafile=str(cert_path))

    class Proxy(BaseHTTPRequestHandler):
        def do_CONNECT(self):
            observed['connect'].append((self.path, dict(self.headers)))
            self.send_response(200)
            self.end_headers()
            with server_tls.wrap_socket(self.connection, server_side=True) as connection:
                body = b''
                while b'\r\n\r\n' not in body:
                    body += connection.recv(4096)
                observed['request'].append(body)
                connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 4\r\n'
                                   b'Connection: close\r\n\r\npage')

        def log_message(self, *args):
            pass

    native_proxy = httpcore.HTTPProxy
    dns = socket.getaddrinfo

    def pinned_dns(host, port, *args, **kwargs):
        if host == 'example.com':
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '',
                     ('93.184.216.34', port))]
        return dns(host, port, *args, **kwargs)

    with local_server(Proxy) as server, patch(
        'agent.execution.public_http.socket.getaddrinfo', side_effect=pinned_dns
    ), patch('agent.execution.public_http.system_proxy_for', return_value=(
        f'http://offline-user:offline-password@127.0.0.1:{server.server_port}'
    )) as route, patch('agent.execution.public_http.httpcore.HTTPProxy', side_effect=(
        lambda **kwargs: native_proxy(**kwargs, ssl_context=client_tls)
    )):
        with httpx.Client(transport=PublicTransport(), trust_env=False, timeout=2) as client:
            assert client.get('https://example.com/test').text == 'page'
    assert str(route.call_args.args[0]) == 'https://example.com/test'
    assert observed['connect'][0][0] == '93.184.216.34:443'
    assert 'Proxy-Authorization' in observed['connect'][0][1]
    assert observed['sni'] == ['example.com']
    assert b'GET /test HTTP/1.1\r\n' in observed['request'][0]
    assert b'Host: example.com\r\n' in observed['request'][0]
    assert b'Proxy-Authorization' not in observed['request'][0]
    assert b'Authorization' not in observed['request'][0]


def test_page_redirect_is_revalidated_before_proxy_gets_private_target():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={'Location': 'http://localhost/private'})

    def dns(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '',
                 ('93.184.216.34' if host == 'example.com' else '127.0.0.1', port))]

    with patch('agent.execution.public_http.socket.getaddrinfo', side_effect=dns), patch(
        'agent.execution.public_http.system_proxy_for', return_value='http://127.0.0.1:8123'
    ), patch('agent.execution.public_http._PinnedProxyTransport',
             return_value=httpx.MockTransport(respond)):
        with httpx.Client(transport=PublicTransport(), follow_redirects=True) as client:
            with pytest.raises(ValueError, match='Private'):
                client.get('https://example.com')
    assert len(requests) == 1
    assert requests[0].url.host == '93.184.216.34'


def test_https_proxy_identity_is_preserved_even_when_its_ip_matches_target():
    from agent.execution.public_http import _OriginTLSStream

    raw = Mock()
    raw.start_tls.return_value = raw
    stream = _OriginTLSStream(raw, '93.184.216.34', 'example.com', proxy_tls=True)
    context = ssl.create_default_context()
    stream.start_tls(context, '93.184.216.34', 2)
    stream.start_tls(context, '93.184.216.34', 2)
    assert [call.args[1] for call in raw.start_tls.call_args_list] == [
        '93.184.216.34', 'example.com'
    ]
