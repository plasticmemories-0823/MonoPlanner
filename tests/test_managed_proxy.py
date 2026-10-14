"""容器下载转发的地址边界与故障分类；不访问外网。"""

from contextlib import contextmanager
import os
import socket
import threading
import unittest
from unittest.mock import Mock, patch

from agent.execution import managed_proxy


@contextmanager
def proxy_server(failure):
    server = managed_proxy.Server(('127.0.0.1', 0), managed_proxy.Handler)
    server.connect_public = Mock(side_effect=failure)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={'poll_interval': 0.01},
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, line):
    with socket.create_connection(server.server_address, timeout=2) as connection:
        connection.sendall(line + b' HTTP/1.1\r\nHost: packages.example\r\n\r\n')
        chunks = []
        while data := connection.recv(16384):
            chunks.append(data)
        return b''.join(chunks)


class ManagedProxyBoundaryTests(unittest.TestCase):
    def test_container_relay_does_not_resolve_target_with_container_dns(self):
        relay = 'http://prepare:temporary@host.docker.internal:12345'
        with patch.dict(os.environ, {'MONOPLANNER_DOWNLOAD_RELAY': relay}):
            with patch.object(socket, 'getaddrinfo', side_effect=AssertionError('poisoned DNS')):
                with patch.object(managed_proxy, 'tunnel') as tunnel:
                    result = managed_proxy.connect_public(
                        'packages.example', 443, 'https://packages.example/artifact'
                    )
        self.assertIs(result, tunnel.return_value)
        tunnel.assert_called_once_with(
            relay, 'packages.example', 443, target_url='https://packages.example/artifact'
        )

    def test_relay_rejects_local_literals_reserved_names_and_header_injection_without_dns(self):
        hosts = (
            '127.0.0.1', '10.0.0.1', '169.254.169.254', '::1', 'fe80::1%1',
            '::ffff:127.0.0.1', '198.18.0.1', '224.0.0.1', '2001::c73b:960c',
            'localhost', 'LOCALHOST.', 'service.local', 'service.internal', 'service.lan',
            'service.home', 'service.home.arpa', 'service.localhost', 'service.localdomain',
            '127.1', '2130706433', '0x7f000001', '', 'singlelabel',
            'packages.example\r\nX-Injected: yes', 'packages.example/path',
            'packages.example:443', 'user@packages.example', 'a..example', '-a.example',
        )
        with patch.dict(os.environ, {'MONOPLANNER_DOWNLOAD_RELAY': 'http://relay.internal:1234'}):
            for host in hosts:
                with self.subTest(host=host):
                    with patch.object(socket, 'getaddrinfo') as resolve:
                        with patch.object(managed_proxy, 'tunnel') as tunnel:
                            with self.assertRaises(ValueError):
                                managed_proxy.connect_public(host, 443)
                    resolve.assert_not_called()
                    tunnel.assert_not_called()

    def test_relay_rejects_url_scheme_identity_credentials_and_injected_headers(self):
        targets = (
            'file://packages.example/file',
            'http://packages.example/file',
            'https://different.example/file',
            'https://packages.example:444/file',
            'https://user:secret@packages.example/file',
            'https://packages.example/file\r\nX-Injected: yes',
        )
        with patch.dict(os.environ, {'MONOPLANNER_DOWNLOAD_RELAY': 'http://relay.internal:1234'}):
            for target in targets:
                with self.subTest(target=target):
                    with patch.object(socket, 'getaddrinfo') as resolve:
                        with patch.object(managed_proxy, 'tunnel') as tunnel:
                            with self.assertRaises(ValueError):
                                managed_proxy.connect_public('packages.example', 443, target)
                    resolve.assert_not_called()
                    tunnel.assert_not_called()

    def test_direct_destination_still_validates_all_dns_answers_before_connecting(self):
        addresses = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2001::c73b:960c', 443)),
        ]
        with patch.dict(os.environ, {'MONOPLANNER_DOWNLOAD_RELAY': ''}):
            with patch.object(socket, 'getaddrinfo', return_value=addresses):
                with patch.object(socket, 'socket') as make:
                    with self.assertRaises(ValueError):
                        managed_proxy.connect_public('packages.example', 443)
        make.assert_not_called()

    def test_failure_responses_distinguish_blocked_network_and_timeout_without_details(self):
        failures = (
            (ValueError('blocked https://user:secret@example.invalid/?token=secret'), 403),
            (OSError('proxy credentials secret and token=secret'), 502),
            (TimeoutError('https://example.invalid/?token=secret'), 504),
        )
        for line in (
            b'CONNECT packages.example:443',
            b'GET http://packages.example/artifact',
            b'HEAD http://packages.example/artifact',
        ):
            for failure, status in failures:
                with self.subTest(method=line.split()[0], status=status):
                    with proxy_server(failure) as server:
                        result = request(server, line)
                    self.assertIn(f' {status} '.encode(), result.split(b'\r\n')[0])
                    self.assertNotIn(b'secret', result)
                    self.assertNotIn(b'token=', result)
                    self.assertNotIn(b'example.invalid', result)


if __name__ == '__main__':
    unittest.main()
