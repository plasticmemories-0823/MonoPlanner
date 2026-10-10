"""Real Windows isolation tests, not simulated permission checks."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import socket
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

    def test_embedded_python_imports_workspace_for_code_module_and_script(self):
        (self.repo/'project_sample.py').write_text('VALUE = "workspace-import-ok"\nprint(VALUE)\n')
        scripts = self.repo/'scripts'; scripts.mkdir()
        (scripts/'sibling.py').write_text('VALUE = "script-sibling-ok"\n')
        (scripts/'main.py').write_text('import sibling, project_sample; print(sibling.VALUE)\n')
        self.backend.read_only = True
        cases = [(['-c', 'import project_sample'], 'workspace-import-ok'),
                 (['-m', 'project_sample'], 'workspace-import-ok'),
                 ([str(scripts/'main.py')], 'script-sibling-ok')]
        for args, expected in cases:
            with self.subTest(args=args):
                result = self.backend.run_process([self.backend.python_executable, *args],
                    stdout_path=self.root/'stdout.log', stderr_path=self.root/'stderr.log')
                self.assertEqual(result['exit_code'], 0, (self.root/'stderr.log').read_text())
                self.assertIn(expected, (self.root/'stdout.log').read_text())
                self.assertEqual(result['isolation'], 'lpac')
                self.assertEqual(result['changed_files'], [])

    def test_executor_workspace_binding_cannot_be_overridden_by_call_parameters(self):
        (self.repo/'project_sample.py').write_text('VALUE = "sandbox-copy"\n')
        result = self.backend.run_process([self.backend.python_executable, '-c',
            'import project_sample,os;print(project_sample.VALUE);print(os.getcwd()==os.environ["MONOPLANER_WORKSPACE_DIR"])'],
            env={'MONOPLANER_WORKSPACE_DIR': str(self.root)},
            stdout_path=self.root/'stdout.log', stderr_path=self.root/'stderr.log')
        self.assertEqual(result['exit_code'], 0)
        self.assertIn('sandbox-copy\nTrue', (self.root/'stdout.log').read_text())

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

    def test_live_control_listener_and_credential_canary_are_inaccessible(self):
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(('127.0.0.1', 0)); listener.listen()
        port = listener.getsockname()[1]
        key = self.root/'backend-key.json'; key.write_text('LOCAL_SECRET_CANARY')
        report, text = self.run_code('''import socket
from pathlib import Path
try:socket.create_connection(('127.0.0.1',%r),1);print('ESCAPED')
except OSError as e:print(e.winerror)
try:print(Path(%r).read_text())
except PermissionError:print('KEY_DENIED')
''' % (port, str(key)))
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('10013', text)
        self.assertIn('KEY_DENIED', text)
        self.assertNotIn('LOCAL_SECRET_CANARY', text)

    def test_hardlinks_cannot_escape_or_be_published(self):
        report, text = self.run_code('import os\ntry:os.link(%r,"escape.txt");print("ESCAPED")\nexcept OSError:print("DENIED")' % str(self.outside))
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('DENIED', text)
        with self.assertRaisesRegex(ValueError, '链接'):
            self.run_code('import os;os.link("original.txt","duplicate.txt")')
        self.assertFalse((self.repo/'duplicate.txt').exists())

    def test_output_bound_stops_entire_job(self):
        report = self.backend.run_process([self.backend.python_executable, '-c', 'import time\nwhile True:print("x"*4096,flush=True);time.sleep(.01)'],
            stdout_path=self.root/'stdout.log', stderr_path=self.root/'stderr.log', timeout=15, max_output_bytes=2048)
        self.assertTrue(report['output_limited'])
        self.assertIsNone(report['exit_code'])
        self.assertLess(report['seconds'], 10)

    def test_standalone_shell_argv_preserves_code_and_quotes(self):
        code = 'print("中文引号\\\"\\\\")'
        command = subprocess.list2cmdline([self.backend.python_executable, '-c', code])
        report = self.backend.run_process(command, shell=True, stdout_path=self.root/'stdout.log', stderr_path=self.root/'stderr.log')
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('中文引号"\\', (self.root/'stdout.log').read_text(encoding='utf-8'))

    def test_authorized_materials_are_filtered_and_read_only(self):
        material = self.root/'materials'; material.mkdir()
        (material/'reference.txt').write_text('reference'); (material/'.env').write_text('PRIVATE_CANARY')
        self.backend.read_roots = [str(material)]
        report = self.backend.run_process([self.backend.python_executable, '-c',
            'import sys;from pathlib import Path;p=Path(sys.argv[1]);print(p.read_text());print(p.with_name(".env").exists())\ntry:p.write_text("bad")\nexcept PermissionError:print("READ_ONLY")', str(material/'reference.txt')],
            stdout_path=self.root/'stdout.log', stderr_path=self.root/'stderr.log')
        self.assertEqual(report['exit_code'], 0)
        text = (self.root/'stdout.log').read_text()
        self.assertIn('reference\nFalse\nREAD_ONLY', text)
        self.assertEqual((material/'reference.txt').read_text(), 'reference')

    def test_project_dependencies_are_available_but_read_only(self):
        dependency = self.repo/'.venv/Lib/site-packages'; dependency.mkdir(parents=True)
        (dependency/'installed_sample.py').write_text('value=123')
        report, text = self.run_code('import installed_sample;from pathlib import Path;print(installed_sample.value)\ntry:Path(installed_sample.__file__).write_text("bad")\nexcept PermissionError:print("READ_ONLY")')
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('123\nREAD_ONLY', text)
        self.assertEqual((dependency/'installed_sample.py').read_text(), 'value=123')

    @unittest.skipUnless((PROJECT_ROOT/'launcher/runtime/node/node.exe').is_file(), 'Bundled Node not prepared')
    def test_node_dependencies_are_available_but_read_only(self):
        dependency = self.repo/'node_modules/sample'; dependency.mkdir(parents=True)
        (dependency/'index.js').write_text('module.exports=123')
        command = [str(PROJECT_ROOT/'launcher/runtime/node/node.exe'), '-e',
            'const fs=require("fs");console.log(require("sample"));try{fs.writeFileSync(require.resolve("sample"),"bad");console.log("ESCAPED")}catch(e){console.log(e.code=== "EPERM"||e.code==="EACCES"?"READ_ONLY":e.code)}']
        report = self.backend.run_process(command, stdout_path=self.root/'stdout.log', stderr_path=self.root/'stderr.log')
        self.assertEqual(report['exit_code'], 0)
        self.assertIn('123\nREAD_ONLY', (self.root/'stdout.log').read_text())
        self.assertEqual((dependency/'index.js').read_text(), 'module.exports=123')
