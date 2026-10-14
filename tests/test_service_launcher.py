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


SERVER = '''import http.server, json, pathlib, subprocess, sys, time
config, port, record, child_file = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
if child_file:
 child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
 pathlib.Path(child_file).write_text(str(child.pid))
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  with open(config) as f: state = json.load(f)
  self.send_response(state.get('status', 200)); self.end_headers()
  self.wfile.write(json.dumps({'queue': state.get('queue')}).encode())
 def do_POST(self):
  body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
  with open(record, 'a') as f:
   f.write(json.dumps({'path': self.path, 'body': body.decode(),
    'authenticated': bool(self.headers.get('X-MonoPlanner-Service'))}) + '\\n')
  with open(config) as f: state = json.load(f)
  time.sleep(state.get('shutdown_delay', 0))
  self.send_response(state.get('shutdown_status', 200)); self.end_headers()
  self.wfile.write(json.dumps({'shutdown': True, 'dispatch_suspended': True,
   'complete': True, 'termination_confirmed': True, 'pending_operations': [],
   'errors': []}).encode())
 def log_message(self, *args): pass
http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
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

    def server(self, owned=True, worker=False):
        port = self.unused_port()
        file = (self.project if owned else self.root) / f'fixture-{port}.py'
        file.write_text(SERVER)
        record = self.root / f'requests-{port}.jsonl'
        child_file = self.root / f'worker-{port}.txt' if worker else None
        child = subprocess.Popen(
            [sys.executable, '-B', str(file), str(self.config), str(port), str(record),
             str(child_file) if child_file else ''],
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
                    if child_file:
                        self.watch_worker(int(child_file.read_text()))
                    return port, child
            except OSError:
                time.sleep(0.03)
        self.fail('Fixture server did not start')

    def watch_worker(self, process_id):
        # 保留原进程句柄，清理和断言不受 PID 重用影响。
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.WaitForSingleObject.restype = ctypes.c_ulong
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100001, False, process_id)
        self.assertTrue(handle, 'Fixture worker must still be alive')
        self.worker_handles = getattr(self, 'worker_handles', [])
        self.worker_handles.append((kernel, handle))

        def cleanup():
            try:
                if kernel.WaitForSingleObject(handle, 0) == 258:
                    kernel.TerminateProcess(handle, 1)
                    kernel.WaitForSingleObject(handle, 5000)
            finally:
                kernel.CloseHandle(handle)

        self.addCleanup(cleanup)

    def stop(self, port, frontend_port=None):
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
                str(frontend_port or self.unused_port()),
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

    def test_stop_cancels_running_queued_and_unknown_requests(self):
        for state in [
            {'queue': {'n_running': 1, 'n_pending': 0}},
            {'queue': {'n_running': 0, 'n_pending': 1}},
            {'queue': None},
            {'status': 500},
        ]:
            with self.subTest(state=state):
                self.config.write_text(json.dumps(state))
                port, child = self.server()
                result = self.stop(port)
                self.assertEqual(result.returncode, 0, result.stderr)
                child.wait(timeout=5)
                requests = [json.loads(line) for line in
                            (self.root / f'requests-{port}.jsonl').read_text().splitlines()]
                shutdown = [entry for entry in requests if entry['path'] == '/service/shutdown']
                self.assertEqual(len(shutdown), 1)
                self.assertEqual(json.loads(shutdown[0]['body']), {})
                self.assertTrue(shutdown[0]['authenticated'])

    def test_stop_terminates_both_owned_process_trees_and_preserves_foreign_service(self):
        backend_port, backend = self.server(worker=True)
        frontend_port, frontend = self.server(worker=True)
        foreign_port, foreign = self.server(owned=False)
        started = time.monotonic()
        result = self.stop(backend_port, frontend_port)
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, 0, result.stderr)
        backend.wait(timeout=5)
        frontend.wait(timeout=5)
        for kernel, handle in self.worker_handles:
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
        self.assertIsNone(foreign.poll())
        with socket.create_connection(('127.0.0.1', foreign_port), timeout=1):
            pass
        self.assertLess(elapsed, 15, 'Exit must have a bounded wait')

    def test_unresponsive_shutdown_still_stops_owned_host_services_within_deadline(self):
        self.config.write_text(json.dumps({'shutdown_delay': 60}))
        port, backend = self.server(worker=True)
        started = time.monotonic()
        result = self.stop(port)
        self.assertEqual(result.returncode, 0, result.stderr)
        backend.wait(timeout=5)
        for kernel, handle in self.worker_handles:
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
        self.assertLess(time.monotonic() - started, 15)

    def test_missing_access_credential_stops_owned_hosts_without_claiming_confirmation(self):
        self.config.write_text(json.dumps({'shutdown_status': 401}))
        port, backend = self.server(worker=True)
        with patch.dict(os.environ, {'MONOPLANER_SERVICE_TOKEN': ''}):
            result = self.stop(port)
        self.assertEqual(result.returncode, 0, result.stderr)
        backend.wait(timeout=5)
        for kernel, handle in self.worker_handles:
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
        requests = [json.loads(line) for line in
                    (self.root / f'requests-{port}.jsonl').read_text().splitlines()]
        self.assertEqual(len(requests), 1)
        self.assertFalse(requests[0]['authenticated'])
        receipt = json.loads((self.project / 'validation/launcher-logs/latest.json').read_text(
            encoding='utf-8-sig'))
        self.assertTrue(receipt['stopped'])
        self.assertFalse(receipt['shutdown']['termination_confirmed'])
        self.assertTrue(receipt['shutdown']['requires_review'])

    def desktop(self, start_delay=0, stop_delay=0, worker=False):
        # 窗口控制仅需桌面 SDK，不复制 Python、Node、Git 和完整备用浏览器。
        executable = self.project / 'MonoPlanner.exe'
        shutil.copyfile(self.script.parents[1] / executable.name, executable)
        runtime = self.project / 'launcher/runtime'
        runtime.mkdir(parents=True)
        for name in ('Microsoft.Web.WebView2.Core.dll', 'Microsoft.Web.WebView2.WinForms.dll',
                     'WebView2Loader.dll', 'LICENSE.txt', 'version.txt'):
            shutil.copyfile(self.script.parents[1] / 'launcher/runtime' / name, runtime / name)
        (self.project / 'scripts').mkdir()
        spawn = ''
        if worker:
            (self.project / 'worker.py').write_text('import time\ntime.sleep(120)\n')
            python = sys.executable.replace("'", "''")
            spawn = (
                "if ($Action -eq 'Start') {\n"
                f"  $worker = Start-Process -FilePath '{python}' "
                "-ArgumentList @('-B', ('\"' + (Join-Path $PSScriptRoot '../worker.py') + '\"')) "
                "-WindowStyle Hidden -PassThru\n"
                "  [IO.File]::WriteAllText((Join-Path $PSScriptRoot '../worker.pid'), "
                "[string]$worker.Id)\n"
                "}\n"
            )
        (self.project / 'scripts/start-app.ps1').write_text(
            "param([string]$Action, [switch]$NoBrowser, "
            "[Parameter(ValueFromRemainingArguments=$true)][string[]]$Other)\n"
            "Add-Content -LiteralPath (Join-Path $PSScriptRoot '../actions.txt') "
            "-Value ($Action + ':' + $NoBrowser.IsPresent)\n"
            + spawn +
            f"if ($Action -eq 'Start') {{ Start-Sleep -Seconds {start_delay} }}\n"
            f"if ($Action -eq 'Stop') {{ Start-Sleep -Seconds {stop_delay} }}\n",
            encoding='utf-8-sig',
        )
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        environment = {
            **os.environ,
            'WEBVIEW2_USER_DATA_FOLDER': str(self.root / 'webview-cache'),
            'MONOPLANER_BACKEND_PORT': str(self.unused_port()),
            'MONOPLANER_UI_PORT': str(self.unused_port()),
            'MONOPLANER_STATE_DIR': str(self.root / 'state'),
        }
        first = subprocess.Popen([str(executable)], startupinfo=startup, env=environment)

        def cleanup():
            if first.poll() is None:
                first.terminate()
            first.wait(timeout=5)

        self.addCleanup(cleanup)
        actions = self.project / 'actions.txt'
        for _ in range(200):
            if actions.exists():
                break
            time.sleep(0.05)
        self.assertTrue(actions.exists(), 'Desktop did not start its service controller')
        return first, actions, environment, startup

    def close_window(self, process):
        handles = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def visit(handle, _):
            owner = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(
                ctypes.c_void_p(handle), ctypes.byref(owner)
            )
            if owner.value == process.pid:
                handles.append(handle)
            return True

        ctypes.windll.user32.EnumWindows(callback_type(visit), 0)
        self.assertTrue(handles)
        for handle in handles:
            ctypes.windll.user32.PostMessageW(ctypes.c_void_p(handle), 0x0010, 0, 0)

    def test_desktop_is_single_instance_and_close_stops_services(self):
        first, actions, environment, startup = self.desktop()
        time.sleep(1.5)
        second = subprocess.Popen([str(self.project / 'MonoPlanner.exe')],
                                  startupinfo=startup, env=environment)
        self.assertEqual(second.wait(timeout=5), 0)
        self.assertEqual(actions.read_text().splitlines(), ['Start:True'])
        self.close_window(first)
        self.assertEqual(first.wait(timeout=10), 0)
        self.assertEqual(actions.read_text().splitlines(), ['Start:True', 'Stop:True'])

    def test_close_during_start_cancels_start_and_never_leaves_background_mode(self):
        first, actions, _, _ = self.desktop(start_delay=60)
        started = time.monotonic()
        self.close_window(first)
        self.assertEqual(first.wait(timeout=15), 0)
        self.assertLess(time.monotonic() - started, 15)
        self.assertEqual(actions.read_text().splitlines(), ['Start:True', 'Stop:True'])

    def test_hung_stop_controller_does_not_leave_the_desktop_open_indefinitely(self):
        first, actions, _, _ = self.desktop(stop_delay=60)
        time.sleep(1.5)
        started = time.monotonic()
        self.close_window(first)
        self.assertEqual(first.wait(timeout=15), 0)
        self.assertLess(time.monotonic() - started, 15)
        self.assertEqual(actions.read_text().splitlines(), ['Start:True', 'Stop:True'])

    def test_abnormal_desktop_exit_releases_owned_worker_but_not_foreign_process(self):
        foreign_port, foreign = self.server(owned=False)
        first, _, _, _ = self.desktop(worker=True)
        pid_file = self.project / 'worker.pid'
        for _ in range(100):
            if pid_file.exists():
                break
            time.sleep(0.05)
        self.assertTrue(pid_file.exists(), 'Desktop fixture did not launch its owned worker')
        self.watch_worker(int(pid_file.read_text()))
        first.terminate()
        first.wait(timeout=5)
        for kernel, handle in self.worker_handles:
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
        self.assertIsNone(foreign.poll())
        with socket.create_connection(('127.0.0.1', foreign_port), timeout=1):
            pass
