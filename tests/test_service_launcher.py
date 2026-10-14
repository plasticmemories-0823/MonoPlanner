"""本地生命周期检查使用临时服务器，不运行产品任务或模型。"""

import json
import ctypes
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from tests.http_client import TestClient
from unittest.mock import patch
from agent.server.app import app
from agent.server.service import service_identity


class ServiceIdentityTests(unittest.TestCase):
    def test_named_local_origin_can_update_settings_but_other_names_cannot(self):
        with patch('agent.server.app.Preferences') as preferences, TestClient(app) as client:
            preferences.return_value.update.return_value = {'configured': True}
            allowed = client.post(
                '/settings', json={}, headers={'Origin': 'http://monoplanner.localhost:3000'}
            )
            self.assertEqual(allowed.status_code, 200)
            denied = client.post(
                '/settings', json={}, headers={'Origin': 'http://monoplanner:3000'}
            )
            self.assertEqual(denied.status_code, 403)
            preferences.return_value.update.assert_called_once_with({})

    def test_loaded_identity_is_public_and_unchanged_by_disk_changes(self):
        with TestClient(app) as client:
            value = client.get('/service')
            self.assertEqual(value.status_code, 200)
            self.assertEqual(value.headers['cache-control'], 'no-store')
            self.assertEqual(set(value.json()), {
                'product', 'project_id', 'version', 'source_hash', 'storage_error',
            })
            self.assertIsNone(value.json()['storage_error'])
            self.assertEqual(value.json()['product'], 'MonoPlanner')
            self.assertEqual(len(value.json()['source_hash']), 64)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'agent').mkdir()
            (root / 'agent' / 'app.py').write_text('old source')
            original = service_identity(root)
            (root / '.env').write_text('SECRET=not exposed')
            self.assertEqual(service_identity(root), original)
            (root / 'agent' / 'app.py').write_text('new source')
            self.assertNotEqual(service_identity(root), original)


SERVER = '''import http.server, json, sys
config, port = sys.argv[1], int(sys.argv[2])
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  with open(config) as f: state = json.load(f)
  self.send_response(state.get('status', 200)); self.end_headers()
  self.wfile.write(json.dumps({'queue': state.get('queue')}).encode())
 def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''


@unittest.skipUnless(os.name == 'nt', 'Windows service launcher')
class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='monoplanner-launcher-')
        self.addCleanup(self.cleanup_fixture)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.config = self.root / 'queue.json'
        self.config.write_text(json.dumps({'queue': {'n_running': 0, 'n_pending': 0}}))
        self.script = Path(__file__).resolve().parents[1] / 'scripts/start-app.ps1'

    def cleanup_fixture(self):
        # WebView2 可能在宿主程序退出后才完成临时配置目录的写入。
        root = Path(self.temp.name).resolve()
        self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(root.name.startswith('monoplanner-launcher-'))
        deadline = time.monotonic() + 10
        while True:
            try:
                self.temp.cleanup()
                return
            except OSError as error:
                if error.winerror not in {5, 32, 145} or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)

    @staticmethod
    def unused_port():
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]

    def server(self, owned=True):
        port = self.unused_port()
        file = (self.project if owned else self.root) / 'fixture.py'
        file.write_text(SERVER)
        child = subprocess.Popen(
            [sys.executable, '-B', str(file), str(self.config), str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def cleanup():
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)

        self.addCleanup(cleanup)
        for _ in range(100):
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=0.1):
                    return port, child
            except OSError:
                time.sleep(0.03)
        self.fail('Fixture server did not start')

    def stop(self, port):
        shell = Path(os.environ['WINDIR']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        return subprocess.run(
            [
                str(shell),
                '-NoProfile',
                '-ExecutionPolicy',
                'Bypass',
                '-File',
                str(self.script),
                '-Action',
                'Stop',
                '-NoBrowser',
                '-ProjectRoot',
                str(self.project),
                '-BackendPort',
                str(port),
                '-FrontendPort',
                str(self.unused_port()),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

    def test_idle_owned_service_stops_and_foreign_service_is_untouched(self):
        port, child = self.server()
        result = self.stop(port)
        self.assertEqual(result.returncode, 0, result.stderr)
        child.wait(timeout=5)
        self.assertTrue(
            json.loads(
                (self.project / 'validation/launcher-logs/latest.json').read_text(
                    encoding='utf-8-sig'
                )
            )['stopped']
        )
        port, foreign = self.server(owned=False)
        self.assertNotEqual(self.stop(port).returncode, 0)
        self.assertIsNone(foreign.poll())

    def test_running_queued_and_unknown_requests_prevent_shutdown(self):
        port, child = self.server()
        for state in [
            {'queue': {'n_running': 1, 'n_pending': 0}},
            {'queue': {'n_running': 0, 'n_pending': 1}},
            {'queue': None},
            {'status': 500},
        ]:
            with self.subTest(state=state):
                self.config.write_text(json.dumps(state))
                self.assertNotEqual(self.stop(port).returncode, 0)
                self.assertIsNone(child.poll())

    def test_desktop_is_single_instance_and_close_stops_services(self):
        executable = self.project / 'MonoPlanner.exe'
        shutil.copyfile(self.script.parents[1] / executable.name, executable)
        shutil.copytree(
            self.script.parents[1] / 'launcher/runtime', self.project / 'launcher/runtime'
        )
        (self.project / 'scripts').mkdir()
        (self.project / 'scripts/start-app.ps1').write_text(
            "param([string]$Action, [switch]$NoBrowser)\n"
            "Add-Content -LiteralPath (Join-Path $PSScriptRoot '../actions.txt') -Value ($Action + ':' + $NoBrowser.IsPresent)\n",
            encoding='utf-8-sig',
        )
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        environment = {**os.environ, 'WEBVIEW2_USER_DATA_FOLDER': str(self.root / 'webview-cache')}
        first = subprocess.Popen([str(executable)], startupinfo=startup, env=environment)
        self.addCleanup(lambda: first.poll() is None and first.terminate())
        actions = self.project / 'actions.txt'
        for _ in range(100):
            if actions.exists():
                break
            time.sleep(0.05)
        self.assertTrue(actions.exists(), 'Desktop did not start its service controller')
        time.sleep(1.5)
        second = subprocess.Popen([str(executable)], startupinfo=startup)
        self.assertEqual(second.wait(timeout=5), 0)
        self.assertEqual(actions.read_text().splitlines(), ['Start:True'])
        handles = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def visit(handle, _):
            owner = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(
                ctypes.c_void_p(handle), ctypes.byref(owner)
            )
            if owner.value == first.pid:
                handles.append(handle)
            return True

        ctypes.windll.user32.EnumWindows(callback_type(visit), 0)
        self.assertTrue(handles)
        for handle in handles:
            ctypes.windll.user32.PostMessageW(ctypes.c_void_p(handle), 0x0010, 0, 0)
        self.assertEqual(first.wait(timeout=10), 0)
        self.assertEqual(actions.read_text().splitlines(), ['Start:True', 'Stop:True'])
