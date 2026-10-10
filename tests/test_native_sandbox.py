"""Real Windows isolation tests, not simulated permission checks."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from agent.execution.native import NativeEnvironment
from agent.settings import PROJECT_ROOT


@unittest.skipUnless(os.name == 'nt' and (PROJECT_ROOT/'launcher/runtime/MonoPlanner.Sandbox.exe').is_file(),
                     'Windows native helper not built')
class NativeSandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='monoplanner-native-test-')
        self.root = Path(self.temp.name)
        self.repo = self.root/'repo'; self.repo.mkdir()
        (self.repo/'original.txt').write_text('original', encoding='utf-8')
        self.outside = self.root/'private.txt'; self.outside.write_text('private', encoding='utf-8')
        self.backend = NativeEnvironment(self.repo, artifact_root=self.root/'runtime')

    def tearDown(self):
        self.temp.cleanup()

    def run_code(self, code, timeout=10):
        report = self.backend.run_process([self.backend.python_executable, '-c', code],
            stdout_path=self.root/'stdout.log', stderr_path=self.root/'stderr.log', timeout=timeout)
        return report, (self.root/'stdout.log').read_text(encoding='utf-8')

    def test_unicode_publication_and_literal_code(self):
        report, text = self.run_code('from pathlib import Path;Path("中文.txt").write_text("引号\\\"反斜杠\\\\\\n",encoding="utf8");print("正常")')
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('正常', text)
        self.assertEqual((self.repo/'中文.txt').read_text(encoding='utf8'), '引号"反斜杠\\\n')

    def test_external_operations_and_private_reads_denied(self):
        code = '''from pathlib import Path
p=Path(%r)
for op in [lambda:p.read_text(),lambda:p.write_text('escaped'),lambda:p.unlink(),lambda:p.rename(p.with_name('renamed.txt'))]:
 try:op();print('ESCAPED')
 except PermissionError:print('DENIED')
''' % str(self.outside)
        report, text = self.run_code(code)
        self.assertEqual(report['exit_code'], 0)
        self.assertEqual(text.count('DENIED'), 4)
        self.assertNotIn('ESCAPED', text)
        self.assertEqual(self.outside.read_text(), 'private')

    def test_network_and_child_inheritance(self):
        code = '''import socket,subprocess,sys
try:socket.create_connection(('127.0.0.1',2024),1);print('ESCAPED')
except OSError as e:print('network',e.winerror)
subprocess.run([sys.executable,'-c',%r],check=True)
''' % ('from pathlib import Path\ntry:Path(%r).write_text("escaped");print("ESCAPED")\nexcept PermissionError:print("child_denied")' % str(self.outside))
        report, text = self.run_code(code)
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('network 10013', text)
        self.assertIn('child_denied', text)
        self.assertNotIn('ESCAPED', text)

    def test_timeout_discards_writes_and_kills_children(self):
        code = 'import subprocess,sys,time;from pathlib import Path;Path("original.txt").write_text("partial");p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);print(p.pid,flush=True);time.sleep(60)'
        report, text = self.run_code(code, timeout=1)
        self.assertTrue(report['timed_out'])
        self.assertEqual((self.repo/'original.txt').read_text(), 'original')
        pid = int(text.strip())
        query = subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'], capture_output=True)
        self.assertNotIn(str(pid).encode(), query.stdout)

    def test_read_only_probe_cannot_modify_source(self):
        self.backend.read_only = True
        report, text = self.run_code('from pathlib import Path\ntry:Path("original.txt").write_text("bad");print("ESCAPED")\nexcept PermissionError:print("DENIED")')
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('DENIED', text)
        self.assertEqual((self.repo/'original.txt').read_text(), 'original')

    def test_shell_body_is_preserved(self):
        report = self.backend.run_process('echo shell-ok', stdout_path=self.root/'stdout.log',
                                         stderr_path=self.root/'stderr.log', shell=True)
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('shell-ok', (self.root/'stdout.log').read_text())
        body = next((self.root/'runtime/native').glob('*/inputs/operation.cmd'))
        self.assertEqual(body.read_bytes(), b'echo shell-ok')

    def test_no_automatic_permission_fallback(self):
        self.backend.helper = self.root/'missing.exe'
        with self.assertRaises(OSError):
            self.run_code('raise RuntimeError("must not execute")')

    def test_acl_changes_refused(self):
        code = 'import subprocess,os;print(subprocess.call(["C:\\Windows\\System32\\icacls.exe","original.txt","/grant","*S-1-1-0:(F)"],stdout=subprocess.PIPE,stderr=subprocess.PIPE))'
        report, text = self.run_code(code)
        self.assertEqual(report['exit_code'], 0)
        self.assertNotEqual(text.strip(), '0')

    def test_reparse_escape_not_published(self):
        code = 'import subprocess;print(subprocess.call(["C:\\Windows\\System32\\cmd.exe","/d","/c",%r],stdout=subprocess.PIPE,stderr=subprocess.PIPE))' % ('mklink /J escape "'+str(self.root)+'"')
        try:
            report, text = self.run_code(code)
        except ValueError as error:
            self.assertIn('链接', str(error))
        else:
            self.assertNotEqual(text.strip(), '0')
        self.assertFalse((self.repo/'escape').exists())

    def test_explicit_full_mode(self):
        self.backend.access_mode = 'full'
        report, _ = self.run_code('from pathlib import Path;Path(%r).write_text("explicit")' % str(self.outside))
        self.assertEqual(report['exit_code'], 0)
        self.assertEqual(self.outside.read_text(), 'explicit')
