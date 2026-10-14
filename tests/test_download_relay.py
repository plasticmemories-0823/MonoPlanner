"""下载中继用真实本机套接字验证，目标与系统代理均由测试替身提供。"""

import base64
from contextlib import contextmanager
import http.server
import json
import os
from pathlib import Path
import select
import socket
import socketserver
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from agent.execution import download_relay, managed_proxy


@contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class Echo(socketserver.BaseRequestHandler):
    def handle(self):
        while data := self.request.recv(1024):
            self.request.sendall(data)


class HoldingProxy(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.recv(16384)
        self.server.accepted.set()
        while self.request.recv(1024):
            pass


class Origin(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(dict(self.headers))
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'ok')


class UpstreamProxy(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_CONNECT(self):
        self.close_connection = True
        self.server.requests.append((self.path, dict(self.headers)))
        if self.server.reject:
            self.send_error(502)
            return
        with socket.create_connection(self.server.destination, timeout=2) as upstream:
            self.send_response(200, 'Connection Established')
            self.end_headers()
            self.wfile.flush()
            self.connection.settimeout(2)
            for _ in range(16):
                ready, _, _ = select.select([self.connection, upstream], [], [], 2)
                if not ready:
                    return
                for source in ready:
                    data = source.recv(16384)
                    if not data:
                        return
                    (upstream if source is self.connection else self.connection).sendall(data)


def proxy(destination):
    server = managed_proxy.Server(('127.0.0.1', 0), UpstreamProxy)
    server.destination, server.requests, server.reject = destination, [], False
    return server


def public_address(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', args[1]))]


def response(connection):
    data = bytearray()
    while chunk := connection.recv(16384):
        data.extend(chunk)
    return bytes(data)


class DownloadRelayTests(unittest.TestCase):
    def test_start_does_not_wait_for_reverse_dns(self):
        with patch.object(socket, 'getfqdn', side_effect=AssertionError('reverse DNS not needed')):
            with download_relay.DownloadRelay() as relay:
                self.assertGreater(relay.server.server_port, 0)

    def test_system_proxy_receives_pinned_ip_and_no_relay_credentials(self):
        with running(socketserver.ThreadingTCPServer(('127.0.0.1', 0), Echo)) as origin:
            with running(proxy(origin.server_address)) as upstream:
                route = f'http://system:secret@127.0.0.1:{upstream.server_port}'
                with patch.object(download_relay, 'system_proxy_for', return_value=route) as resolve:
                    with patch.object(managed_proxy, 'public_addresses', side_effect=public_address):
                        with download_relay.DownloadRelay() as relay:
                            with managed_proxy.tunnel(relay.url(), 'download.example', 443) as client:
                                client.sendall(b'example tls bytes')
                                self.assertEqual(client.recv(1024), b'example tls bytes')
                self.assertEqual(upstream.requests[0][0], '93.184.216.34:443')
                self.assertEqual(upstream.requests[0][1]['Proxy-Authorization'],
                                 'Basic ' + base64.b64encode(b'system:secret').decode())
                self.assertNotIn(relay.token, repr(upstream.requests))
                self.assertEqual(resolve.call_args.args[0].host, 'download.example')

    def test_container_forwarder_and_host_relay_preserve_http_url_and_strip_credentials(self):
        origin = managed_proxy.Server(('127.0.0.1', 0), Origin)
        origin.requests = []
        with running(origin), running(proxy(origin.server_address)) as upstream:
            route = f'http://127.0.0.1:{upstream.server_port}'
            with patch.object(download_relay, 'system_proxy_for', return_value=route) as resolve:
                with patch.object(managed_proxy, 'public_addresses', side_effect=public_address):
                    with download_relay.DownloadRelay() as relay:
                        with patch.dict(os.environ, {'MONOPLANNER_DOWNLOAD_RELAY': relay.url()}):
                            with running(managed_proxy.Server(('127.0.0.1', 0), managed_proxy.Handler)) as inner:
                                with socket.create_connection(inner.server_address, timeout=2) as client:
                                    client.sendall(b'GET http://packages.example/files/a.whl HTTP/1.1\r\n'
                                                  b'Host: packages.example\r\n'
                                                  b'Proxy-Authorization: Basic do-not-forward\r\n\r\n')
                                    result = response(client)
            self.assertIn(b'200 OK', result)
            self.assertTrue(result.endswith(b'ok'))
            self.assertEqual(str(resolve.call_args.args[0]), 'http://packages.example/files/a.whl')
            self.assertEqual(origin.requests[0]['Host'], 'packages.example')
            self.assertNotIn('Proxy-Authorization', origin.requests[0])
            self.assertNotIn('X-MonoPlanner-Download-URL', origin.requests[0])

    def test_next_request_observes_pac_direct_without_restarting_relay(self):
        origin = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Echo)
        with running(origin), running(proxy(origin.server_address)) as upstream:
            addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', origin.server_address)]
            routes = [f'http://127.0.0.1:{upstream.server_port}', None]
            with patch.object(download_relay, 'system_proxy_for', side_effect=routes) as resolve:
                with patch.object(managed_proxy, 'public_addresses', return_value=addresses):
                    with download_relay.DownloadRelay() as relay:
                        for text in (b'proxy', b'direct'):
                            with managed_proxy.tunnel(relay.url(), 'packages.example', 443) as client:
                                client.sendall(text)
                                self.assertEqual(client.recv(1024), text)
            self.assertEqual(resolve.call_count, 2)
            self.assertEqual(len(upstream.requests), 1)

    def test_unauthenticated_relay_never_resolves_or_connects(self):
        with patch.object(download_relay, 'connect_default') as connect:
            with download_relay.DownloadRelay() as relay:
                with socket.create_connection(('127.0.0.1', relay.server.server_port), timeout=2) as client:
                    client.sendall(b'CONNECT public.example:443 HTTP/1.1\r\n\r\n')
                    self.assertIn(b'407', response(client))
        connect.assert_not_called()

    def test_saturated_relay_does_not_accumulate_rejected_sockets(self):
        clients = []
        with download_relay.DownloadRelay() as relay:
            address = ('127.0.0.1', relay.server.server_port)
            try:
                for _ in range(16):
                    client = socket.create_connection(address, timeout=2)
                    client.sendall(b'CONNECT ')
                    clients.append(client)
                deadline = time.monotonic() + 2
                while len(relay.server._connections) < 16 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(len(relay.server._connections), 16)
                for _ in range(40):
                    with socket.create_connection(address, timeout=2) as rejected:
                        self.assertEqual(rejected.recv(1), b'')
                self.assertEqual(len(relay.server._connections), 16)
            finally:
                for client in clients:
                    client.close()

    def test_private_destination_is_rejected_even_with_valid_preparation_token(self):
        with patch.object(download_relay, 'system_proxy_for') as route:
            with download_relay.DownloadRelay() as relay:
                with self.assertRaises(OSError):
                    managed_proxy.tunnel(relay.url(), '127.0.0.1', 443)
        route.assert_not_called()

    def test_failed_system_proxy_does_not_retry_direct(self):
        with patch.object(managed_proxy, 'public_addresses', side_effect=public_address):
            with patch.object(download_relay, 'system_proxy_for', side_effect=httpx.ProxyError('PAC failed')):
                with patch.object(socket, 'socket') as direct:
                    with self.assertRaises(OSError):
                        download_relay.connect_default('public.example', 443, 'https://public.example/')
                direct.assert_not_called()

    def test_close_cancels_open_tunnel_and_releases_listener(self):
        with running(socketserver.ThreadingTCPServer(('127.0.0.1', 0), Echo)) as origin:
            addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', origin.server_address)]
            with patch.object(download_relay, 'system_proxy_for', return_value=None):
                with patch.object(managed_proxy, 'public_addresses', return_value=addresses):
                    relay = download_relay.DownloadRelay()
                    relay.__enter__()
                    client = managed_proxy.tunnel(relay.url(), 'packages.example', 443)
                    client.sendall(b'ready')
                    self.assertEqual(client.recv(1024), b'ready')
                    started, port = time.monotonic(), relay.server.server_port
                    relay.__exit__(None, None, None)
                    self.assertLess(time.monotonic() - started, 2)
                    try:
                        self.assertEqual(client.recv(1024), b'')
                    except ConnectionResetError:
                        pass
                    finally:
                        client.close()
                    with self.assertRaises(OSError):
                        socket.create_connection(('127.0.0.1', port), timeout=0.2)

    def test_cancel_during_pac_cannot_start_late_outbound_connection(self):
        waiting, release, finished = threading.Event(), threading.Event(), threading.Event()
        failures = []

        def resolve(url):
            waiting.set()
            release.wait(2)
            return None

        def connect(relay):
            try:
                relay.server.connect_public('public.example', 443, 'https://public.example/')
            except OSError:
                failures.append('cancelled')
            finally:
                finished.set()

        with patch.object(managed_proxy, 'public_addresses', side_effect=public_address):
            with patch.object(download_relay, 'system_proxy_for', side_effect=resolve):
                relay = download_relay.DownloadRelay()
                relay.__enter__()
                thread = threading.Thread(target=connect, args=(relay,), daemon=True)
                thread.start()
                self.assertTrue(waiting.wait(2))
                relay.__exit__(None, None, None)
                with patch.object(socket, 'socket') as late_connection:
                    release.set()
                    self.assertTrue(finished.wait(2))
                late_connection.assert_not_called()
                self.assertEqual(failures, ['cancelled'])

    def test_cancel_closes_socket_while_upstream_connect_handshake_is_pending(self):
        upstream = socketserver.ThreadingTCPServer(('127.0.0.1', 0), HoldingProxy)
        upstream.accepted = threading.Event()
        finished, failures = threading.Event(), []

        def connect(relay):
            try:
                relay.server.connect_public('public.example', 443, 'https://public.example/')
            except OSError:
                failures.append('cancelled')
            finally:
                finished.set()

        with running(upstream):
            route = f'http://127.0.0.1:{upstream.server_address[1]}'
            with patch.object(managed_proxy, 'public_addresses', side_effect=public_address):
                with patch.object(download_relay, 'system_proxy_for', return_value=route):
                    relay = download_relay.DownloadRelay()
                    relay.__enter__()
                    thread = threading.Thread(target=connect, args=(relay,), daemon=True)
                    thread.start()
                    self.assertTrue(upstream.accepted.wait(2))
                    relay.__exit__(None, None, None)
                    self.assertTrue(finished.wait(2))
                    self.assertEqual(failures, ['cancelled'])

    def test_slow_connect_response_has_absolute_deadline(self):
        clock = [0]

        def receive(size):
            clock[0] += 11
            return b'x'

        with patch.object(socket, 'getaddrinfo', side_effect=public_address):
            with patch.object(socket, 'socket') as make:
                make.return_value.recv.side_effect = receive
                with patch.object(managed_proxy.time, 'monotonic', side_effect=lambda: clock[0]):
                    with self.assertRaises(OSError):
                        managed_proxy.tunnel('http://127.0.0.1:8080', '93.184.216.34', 443)
        self.assertEqual(make.return_value.recv.call_count, 3)
        make.return_value.close.assert_called_once()

    def test_cli_environment_replaces_stale_proxy_and_does_not_mutate_process(self):
        with patch.dict(os.environ, {'HTTP_PROXY': 'http://stale.invalid', 'NO_PROXY': '*'}):
            with download_relay.DownloadRelay() as relay:
                environment = relay.environment()
                self.assertEqual(environment['HTTP_PROXY'], relay.url())
                self.assertEqual(environment['NO_PROXY'], '')
                self.assertEqual(os.environ['HTTP_PROXY'], 'http://stale.invalid')
                self.assertIn('@host.docker.internal:', relay.url(container=True))

    def test_download_logs_hide_ephemeral_token_even_on_failure_and_chunk_boundary(self):
        from agent.execution.managed import _default_downloads, _preparation_log

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'download.log'
            with _default_downloads() as relay:
                data = b'x' * 65520 + relay.token.encode() + b' error ' + relay.url().encode()
                with self.assertRaises(RuntimeError):
                    with _preparation_log(path) as output:
                        output.write(data)
                        raise RuntimeError('download failed')
                saved = path.read_bytes()
                self.assertEqual(len(data), len(saved))
                self.assertNotIn(relay.token.encode(), saved)
                self.assertIn(b' error ', saved)

    def test_container_inspection_output_omits_environment_and_command(self):
        from agent.execution.managed import ManagedEnvironments
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as temporary:
            manager = ManagedEnvironments(Path(temporary))
            process = Mock(returncode=0)
            process.poll.return_value = 0
            with patch('agent.execution.managed.docker_executable', return_value='docker'):
                with patch('agent.execution.managed.subprocess.Popen', return_value=process) as start:
                    manager._run(['container', 'inspect', 'owned-container'],
                                 log=Path(temporary) / 'inspect.log')
            command = start.call_args.args[0]
            template = command[command.index('--format') + 1]
            self.assertIn('.Config.Labels', template)
            self.assertIn('.State', template)
            self.assertNotIn('.Config.Env', template)
            self.assertNotIn('.Config.Cmd', template)

    def test_docker_modes_report_only_modes_not_addresses(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'Docker/settings-store.json'
            path.parent.mkdir()
            path.write_text(json.dumps({'ProxyHTTPMode': 'system', 'ContainersProxyMode': 'manual',
                                        'ProxyHTTP': 'http://secret:password@private:80'}))
            with patch.dict(os.environ, {'APPDATA': temporary}), patch.object(os, 'name', 'nt'):
                result = download_relay.docker_proxy_modes()
        self.assertEqual(result, {'desktop': 'system', 'containers': 'manual'})

    def test_download_relay_is_part_of_frozen_source_identity(self):
        from agent.storage.recovery import code_key

        baseline = code_key()
        original = Path.read_bytes

        def altered(path):
            data = original(path)
            return data + b'\n# changed\n' if path.name == 'download_relay.py' else data

        with patch.object(Path, 'read_bytes', altered):
            self.assertNotEqual(code_key(), baseline)


if __name__ == '__main__':
    unittest.main()
