"""公网代理 DNS 的离线边界、取消和真实 TLS 传输验证。"""

from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import socket
import ssl
from threading import Event, Thread
import time
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock, patch

import httpcore
import pytest

from agent.execution import public_dns as dns


@pytest.fixture(autouse=True)
def clear_cache():
    with dns._cache_lock:
        dns._cache.clear()
    yield
    with dns._cache_lock:
        dns._cache.clear()


def answer(host='example.com', kind='A', addresses=None, ttl=60):
    number = 1 if kind == 'A' else 28
    if addresses is None:
        addresses = ['93.184.216.34'] if kind == 'A' else ['2606:4700:4700::1111']
    return {
        'Status': 0, 'TC': False,
        'Question': [{'name': host + '.', 'type': number}],
        'Answer': [{'name': host + '.', 'type': number, 'TTL': ttl, 'data': ip}
                   for ip in addresses],
    }


def fake_query(host, kind, *args):
    return answer(host, kind)


def test_proxy_resolution_never_uses_polluted_operating_system_dns_and_returns_pinned_ips():
    with patch.object(dns, '_query', side_effect=fake_query) as query, patch.object(
        socket, 'getaddrinfo', side_effect=AssertionError('polluted DNS must not be consulted'),
    ):
        result = dns.proxy_public_addresses('Example.com.', 443, 'http://proxy.test:80')
    assert [row[4][0] for row in result] == ['93.184.216.34', '2606:4700:4700::1111']
    assert [row[4][1] for row in result] == [443, 443]
    assert [call.args[1] for call in query.call_args_list] == ['A', 'AAAA']


@pytest.mark.parametrize('bad_ip', [
    '127.0.0.1', '10.0.0.1', '169.254.169.254', '100.64.0.1', '192.0.2.1',
    '224.0.0.1', '240.0.0.1', '::1', 'fe80::1', '2001::1', 'ff02::1',
])
def test_private_reserved_or_mixed_answer_is_rejected(bad_ip):
    kind = 'AAAA' if ':' in bad_ip else 'A'
    payload = answer(kind=kind, addresses=[bad_ip])
    payload['Answer'].append(answer()['Answer'][0])
    with pytest.raises(ValueError):
        dns._answer(payload, 'example.com', kind)


def test_cname_chain_only_returns_related_final_records_and_checks_unrelated_addresses():
    payload = answer('edge.example.com')
    payload['Question'] = [{'name': 'example.com.', 'type': 1}]
    payload['Answer'].insert(0, {
        'name': 'example.com.', 'type': 5, 'TTL': 20, 'data': 'edge.example.com.',
    })
    payload['Answer'].append(answer('unrelated.example.com')['Answer'][0])
    assert dns._answer(payload, 'example.com', 'A') == (['93.184.216.34'], 20)
    payload['Answer'][-1]['data'] = '127.0.0.1'
    with pytest.raises(ValueError):
        dns._answer(payload, 'example.com', 'A')


@pytest.mark.parametrize('replacement', [
    'localhost.', 'private.local.', '127.0.0.1', 'example.com.',
])
def test_invalid_aliases_are_rejected(replacement):
    payload = answer(addresses=[])
    payload['Answer'] = [{'name': 'example.com.', 'type': 5, 'TTL': 60, 'data': replacement}]
    with pytest.raises(ValueError):
        dns._answer(payload, 'example.com', 'A')


def test_unrelated_records_cannot_supply_the_requested_destination():
    payload = answer('other.example.com')
    payload['Question'] = [{'name': 'example.com.', 'type': 1}]
    assert dns._answer(payload, 'example.com', 'A')[0] == []


@pytest.mark.parametrize('field,value', [
    ('Status', 3), ('Status', True), ('TC', True), ('Question', []),
    ('Question', [{'name': 'other.example.com.', 'type': 1}]),
    ('Question', [{'name': 'example.com.', 'type': True}]), ('Answer', 'invalid'),
])
def test_malformed_truncated_and_mismatched_dns_responses_are_rejected(field, value):
    payload = answer()
    payload[field] = value
    with pytest.raises(ValueError):
        dns._answer(payload, 'example.com', 'A')


@pytest.mark.parametrize('error', [OSError('http://user:secret@proxy.test'), TimeoutError(),
                                 ValueError(), httpcore.ProxyError('secret')])
def test_failure_is_bounded_sanitized_and_not_retried(error):
    with patch.object(dns, '_query', side_effect=error) as query:
        with pytest.raises(OSError, match='Public DNS resolution failed') as caught:
            dns.proxy_public_addresses('example.com', 443, 'http://user:secret@proxy.test')
    assert 'secret' not in str(caught.value)
    query.assert_called_once()
    assert not dns._cache


def test_cancel_after_first_query_blocks_second_query_and_cache():
    stopped = Event()

    def query(host, kind, *args):
        stopped.set()
        return answer(host, kind)

    with patch.object(dns, '_query', side_effect=query) as mocked:
        with pytest.raises(OSError):
            dns.proxy_public_addresses('example.com', 443, 'http://proxy.test',
                                       cancelled=stopped.is_set)
    mocked.assert_called_once()
    assert not dns._cache


def test_literals_bypass_dns_but_private_literals_are_still_rejected():
    with patch.object(dns, '_query') as query:
        assert dns.proxy_public_addresses('1.1.1.1', 80, 'http://proxy.test')[0][4] == (
            '1.1.1.1', 80,
        )
        with pytest.raises(ValueError):
            dns.proxy_public_addresses('127.0.0.1', 80, 'http://proxy.test')
    query.assert_not_called()


def test_cache_is_bounded_route_specific_and_ttl_capped():
    with patch.object(dns, '_query', side_effect=fake_query) as query, patch.object(
        dns, 'MAX_CACHE', 3,
    ), patch.object(dns.time, 'monotonic', return_value=10):
        for i in range(5):
            dns.proxy_public_addresses(f'host{i}.example.com', 443, 'http://user:secret@proxy')
        assert len(dns._cache) == 3
        dns.proxy_public_addresses('host4.example.com', 80, 'http://user:secret@proxy')
        assert query.call_count == 10
        dns.proxy_public_addresses('host4.example.com', 80, 'http://different-proxy')
        assert query.call_count == 12
        assert 'secret' not in repr(dns._cache)
    with patch.object(dns, '_query', side_effect=lambda h, k, *a: answer(h, k, ttl=10000)), (
        patch.object(dns.time, 'monotonic', return_value=20)
    ):
        dns.proxy_public_addresses('ttl.example.com', 80, 'http://proxy')
        assert list(dns._cache.values())[-1][0] == 320


def test_zero_ttl_and_expired_answers_are_not_reused():
    with patch.object(dns, '_query', side_effect=lambda h, k, *a: answer(h, k, ttl=0)) as query:
        dns.proxy_public_addresses('example.com', 443, 'http://proxy')
        dns.proxy_public_addresses('example.com', 443, 'http://proxy')
        assert query.call_count == 4
        assert not dns._cache
    with patch.object(dns, '_query', side_effect=fake_query) as query:
        with patch.object(dns.time, 'monotonic', return_value=10):
            dns.proxy_public_addresses('example.com', 443, 'http://proxy')
        with patch.object(dns.time, 'monotonic', return_value=100):
            dns.proxy_public_addresses('example.com', 443, 'http://proxy')
        assert query.call_count == 4


def test_total_deadline_and_capacity_do_not_create_background_workers():
    with pytest.raises(OSError, match='timed out'):
        dns._remaining(time.monotonic() - 1, None)
    with patch.object(dns, '_queries') as slots, patch.object(dns, '_query') as query:
        slots.acquire.return_value = False
        with pytest.raises(OSError, match='capacity'):
            dns.proxy_public_addresses('example.com', 443, 'http://proxy')
        query.assert_not_called()


@pytest.mark.parametrize('failure', ['wrap', 'handshake', 'register', 'nested', 'cancelled'])
def test_tls_setup_failure_closes_current_connection_immediately(failure):
    raw = Mock(spec=ssl.SSLSocket if failure == 'nested' else socket.socket)
    original = Mock()
    original.get_extra_info.return_value = raw
    wrapped, context = Mock(spec=ssl.SSLSocket), Mock()
    context.wrap_socket.return_value = wrapped
    register = Mock()
    error = OSError('controlled TLS failure')
    if failure == 'wrap':
        context.wrap_socket.side_effect = error
    elif failure == 'handshake':
        wrapped.do_handshake.side_effect = error
    elif failure == 'register':
        register.side_effect = error
    stream = dns._ObservedStream(
        original, time.monotonic() + 2, lambda: failure == 'cancelled', register,
    )
    with patch.object(dns, 'TLSinTLSStream', side_effect=error) as nested:
        with pytest.raises(OSError):
            stream.start_tls(context, '1.1.1.1')
    if failure in {'handshake', 'register'}:
        wrapped.close.assert_called_once()
        original.close.assert_not_called()
    else:
        original.close.assert_called_once()
    if failure == 'nested':
        nested.assert_called_once()
        context.wrap_socket.assert_not_called()
    if failure == 'register':
        wrapped.do_handshake.assert_not_called()


@contextmanager
def dns_proxy(tmp_path, *, stalled=False, large=False, secure=False, wrong_certificate=False):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, '1.1.1.1')])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.ip_address(
                    '1.1.1.2' if wrong_certificate else '1.1.1.1'
                )),
                x509.IPAddress(ipaddress.ip_address('127.0.0.1')),
            ]), False).sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                         serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
    server_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_tls.load_cert_chain(cert_path, key_path)
    client_tls = ssl.create_default_context(cafile=str(cert_path))
    observed = {'connect': [], 'requests': [], 'ready': Event(), 'release': Event()}

    class InnerTLS:
        def __init__(self, raw):
            self.raw, self.incoming, self.outgoing = raw, ssl.MemoryBIO(), ssl.MemoryBIO()
            self.tls = server_tls.wrap_bio(self.incoming, self.outgoing, server_side=True)
            self.perform(self.tls.do_handshake)

        def perform(self, operation):
            while True:
                try:
                    result = operation()
                except ssl.SSLWantReadError:
                    self.raw.sendall(self.outgoing.read())
                    data = self.raw.recv(16384)
                    if data:
                        self.incoming.write(data)
                    else:
                        self.incoming.write_eof()
                else:
                    self.raw.sendall(self.outgoing.read())
                    return result

        def recv(self, size):
            return self.perform(lambda: self.tls.read(size))

        def sendall(self, data):
            while data:
                written = self.perform(lambda: self.tls.write(data))
                data = data[written:]

        def close(self):
            self.raw.close()

    class Proxy(BaseHTTPRequestHandler):
        def do_CONNECT(self):
            observed['connect'].append((self.path, dict(self.headers)))
            self.send_response(200)
            self.end_headers()
            try:
                wrapped = (InnerTLS(self.connection) if secure else
                           server_tls.wrap_socket(self.connection, server_side=True))
                with closing(wrapped) as connection:
                    request = b''
                    while b'\r\n\r\n' not in request:
                        chunk = connection.recv(4096)
                        if not chunk:
                            return
                        request += chunk
                    observed['requests'].append(request)
                    observed['ready'].set()
                    if stalled:
                        observed['release'].wait(3)
                        return
                    target = request.split(b' ')[1].decode('ascii')
                    params = parse_qs(urlsplit(target).query)
                    body = json.dumps(answer(params['name'][0], params['type'][0])).encode()
                    if large:
                        body = b' ' * (dns.MAX_RESPONSE + 1)
                    connection.sendall(
                        b'HTTP/1.1 200 OK\r\nContent-Type: application/dns-json\r\n'
                        b'Content-Length: ' + str(len(body)).encode() +
                        b'\r\nConnection: close\r\n\r\n' + body
                    )
            except (OSError, ssl.SSLError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    if secure:
        server.socket = server_tls.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    native_pool = httpcore.HTTPProxy
    try:
        with patch.object(dns.httpcore, 'HTTPProxy', side_effect=lambda **kwargs: native_pool(
            **kwargs, ssl_context=client_tls, proxy_ssl_context=client_tls if secure else None,
        )):
            scheme = 'https' if secure else 'http'
            yield f'{scheme}://offline-user:offline-secret@127.0.0.1:{server.server_port}', observed
    finally:
        observed['release'].set()
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize('secure', [False, True])
def test_real_proxy_pins_dns_ip_verifies_certificate_and_keeps_auth_out_of_dns(tmp_path, secure):
    registered = []
    with dns_proxy(tmp_path, secure=secure) as (proxy, observed), patch.object(
        socket, 'getaddrinfo', side_effect=AssertionError('OS DNS forbidden'),
    ):
        result = dns.proxy_public_addresses('example.com', 443, proxy, on_socket=registered.append)
    assert len(result) == 2
    assert len(registered) == 4
    assert all(row[0] == '1.1.1.1:443' for row in observed['connect'])
    assert all('Proxy-Authorization' in row[1] for row in observed['connect'])
    assert all(b'Authorization' not in value and b'offline-secret' not in value
               for value in observed['requests'])
    assert b'GET /dns-query?name=example.com&type=A HTTP/1.1' in observed['requests'][0]
    assert all(item.fileno() == -1 for item in registered)


def test_real_connection_is_closed_by_owner_during_stalled_response(tmp_path):
    registered, result, cancelled = [], [], Event()
    with dns_proxy(tmp_path, stalled=True) as (proxy, observed):
        def resolve():
            try:
                dns.proxy_public_addresses('example.com', 443, proxy,
                                           on_socket=registered.append,
                                           cancelled=cancelled.is_set)
            except OSError as exc:
                result.append(str(exc))

        worker = Thread(target=resolve, daemon=True)
        worker.start()
        assert observed['ready'].wait(2)
        cancelled.set()
        started = time.monotonic()
        for connection in registered:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        worker.join(1)
        assert not worker.is_alive()
        assert time.monotonic() - started < 1
    assert len(observed['connect']) == 1
    assert result and not dns._cache


def test_real_oversized_response_is_rejected_without_retry(tmp_path):
    with dns_proxy(tmp_path, large=True) as (proxy, observed):
        with pytest.raises(OSError):
            dns.proxy_public_addresses('example.com', 443, proxy)
    assert len(observed['connect']) == 1


def test_real_stalled_response_obeys_total_deadline(tmp_path):
    with dns_proxy(tmp_path, stalled=True) as (proxy, observed), patch.object(
        dns, 'DNS_TIMEOUT', 0.2,
    ):
        started = time.monotonic()
        with pytest.raises(OSError):
            dns.proxy_public_addresses('example.com', 443, proxy)
        assert time.monotonic() - started < 1
    assert len(observed['connect']) == 1


def test_dns_tls_still_rejects_wrong_ip_certificate(tmp_path):
    with dns_proxy(tmp_path, wrong_certificate=True) as (proxy, observed):
        with pytest.raises(OSError):
            dns.proxy_public_addresses('example.com', 443, proxy)
    assert len(observed['connect']) == 1
    assert not observed['requests']
