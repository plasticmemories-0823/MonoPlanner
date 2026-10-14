"""运行时事实来自一个受控容器探测，绝不依据宿主机可用性推断。"""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.execution.docker import DockerEnvironment
from agent.execution.runtime_inventory import probe_runtime
from agent.execution.task_docker import TaskDocker
from agent.storage.records import save


class RuntimeFactsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.task_root = self.root / 'state/tasks/task'
        self.runtime = {'python': '/project-runtime/bin/python', 'conda_env': ''}
        self.report = {
            'python': '3.6.15 (container-only fixture)',
            'imports': {},
            'checkers': {'pytest': False, 'flake8': True, 'ruff': None},
            'checker_errors': {'ruff': 'PermissionError: unavailable metadata'},
        }
        self.commands = []
        self.exit_code = 0
        guard = patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root / 'state')})
        guard.start()
        self.addCleanup(guard.stop)

    def environment(self):
        return TaskDocker(self.repo, 'sha256:' + '1' * 64, self.task_root, runtime=self.runtime)

    def docker(self, environment, argv, **kwargs):
        self.commands.append(argv)
        if argv[0] == 'run':
            return subprocess.CompletedProcess(argv, 0, '2' * 64, '')
        if argv[0] == 'inspect':
            actual = {
                'Image': environment.image,
                'Name': '/' + environment.container_name,
                'State': {'Running': True},
                'HostConfig': {
                    'NetworkMode': 'none',
                    'Privileged': False,
                    'CapDrop': ['ALL'],
                    'SecurityOpt': ['no-new-privileges'],
                },
                'Mounts': [
                    {'Source': str(environment.repo_root), 'Destination': '/testbed', 'RW': True},
                    {
                        'Source': str(environment.artifact_root),
                        'Destination': '/agent-artifacts',
                        'RW': True,
                    },
                ],
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps([actual]), '')
        self.assertEqual(argv[:5], ['exec', '2' * 64, self.runtime['python'], '-I', '-c'])
        payload = json.loads(base64.b64decode(argv[-1]))
        self.assertEqual(payload['argv'][-4:-1], [self.runtime['python'], '-I', '-B'])
        self.assertTrue(payload['argv'][-1].startswith('/agent-artifacts/'))
        status = environment.artifact_root / Path(payload['status_path']).name
        save(status, {'exit_code': self.exit_code, 'timed_out': False})
        kwargs['stdout'].write((json.dumps(self.report) + '\n').encode())
        kwargs['stderr'].write(b'container stderr fixture\n')
        return subprocess.CompletedProcess(argv, self.exit_code)

    def start(self):
        environment = self.environment()
        with patch.object(DockerEnvironment, '_docker', autospec=True, side_effect=self.docker):
            environment.start()
        return environment

    def test_once_per_binding_facts_raw_result_and_config_path_survive_resume(self):
        environment = self.start()
        facts = environment.runtime_facts
        self.assertEqual(facts['python'], self.report['python'])
        self.assertEqual(facts['python_executable'], self.runtime['python'])
        self.assertFalse(facts['checkers']['pytest'])  # 宿主机 pytest 不能覆盖 Docker 证据。
        self.assertIsNone(facts['checkers']['ruff'])
        self.assertIn('PermissionError', facts['checker_errors']['ruff'])
        original = environment.runtime_facts_path.read_bytes()
        record = json.loads(original)
        self.assertEqual(record['state'], 'completed')
        self.assertEqual(json.loads(record['result']['raw']['stdout.log']), self.report)
        self.assertEqual(record['result']['raw']['stderr.log'], 'container stderr fixture\n')
        with patch.object(DockerEnvironment, '_docker', autospec=True, side_effect=self.docker):
            for _ in range(3):
                environment.start()
            resumed = self.environment().start()
            resumed.start()
        self.assertEqual(resumed.runtime_facts, facts)
        self.assertEqual(environment.runtime_facts_path.read_bytes(), original)
        self.assertEqual(sum(row[0] == 'exec' for row in self.commands), 1)
        self.assertEqual(sum(row[0] == 'run' for row in self.commands), 1)
        self.assertEqual(sum(row[0] == 'inspect' for row in self.commands), 1)

    def test_missing_pending_or_tampered_facts_cannot_repeat_probe(self):
        environment = self.start()
        original = environment.runtime_facts_path.read_bytes()
        for change in ('missing', 'pending', 'facts', 'binding', 'raw'):
            with self.subTest(change=change):
                record = json.loads(original)
                if change == 'missing':
                    environment.runtime_facts_path.unlink()
                else:
                    if change == 'pending':
                        record['state'] = 'pending'
                    elif change == 'facts':
                        record['result']['facts']['python'] = 'HOST OVERRIDE'
                    elif change == 'binding':
                        record['binding'] = 'OTHER CONTAINER'
                    else:
                        record['result']['raw']['stdout.log'] = 'CHANGED'
                    save(environment.runtime_facts_path, record)
                with patch.object(
                    DockerEnvironment, '_docker', autospec=True, side_effect=self.docker
                ):
                    with self.assertRaises(ValueError):
                        self.environment().start()
                environment.runtime_facts_path.write_bytes(original)
        self.assertEqual(sum(row[0] == 'exec' for row in self.commands), 1)

    def test_nonzero_probe_cannot_pass_or_fall_back_to_host(self):
        self.exit_code = 127
        environment = self.environment()
        with patch.object(DockerEnvironment, '_docker', autospec=True, side_effect=self.docker):
            with self.assertRaisesRegex(ValueError, '不会使用宿主机 Python'):
                environment.start()
            with self.assertRaisesRegex(ValueError, '不会自动重复探测'):
                self.environment().start()
        record = json.loads(environment.runtime_facts_path.read_text())
        self.assertEqual(record['state'], 'failed')
        self.assertFalse(record['result']['report']['ok'])
        self.assertNotIn('python', environment.runtime_facts)
        self.assertEqual(sum(row[0] == 'exec' for row in self.commands), 1)

    def test_interrupted_probe_keeps_reservation_and_never_replays(self):
        environment = self.environment()
        with patch.object(
            DockerEnvironment, '_docker', autospec=True, side_effect=self.docker
        ), patch(
            'agent.execution.runtime_inventory.probe_runtime',
            side_effect=RuntimeError('interrupted'),
        ) as probe:
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                environment.start()
            with self.assertRaisesRegex(ValueError, '不会自动重复探测'):
                self.environment().start()
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(json.loads(environment.runtime_facts_path.read_text())['state'], 'pending')

    def test_isolated_probe_never_imports_project_modules_or_shadowed_stdlib(self):
        marker = self.repo / 'IMPORTED'
        code = f'from pathlib import Path\nPath({str(marker)!r}).write_text("BAD")\nraise RuntimeError("BAD")\n'
        for name in ('json.py', 'shutil.py', 'pytest.py'):
            (self.repo / name).write_text(code)
        (self.repo / 'sample_pkg').mkdir()
        (self.repo / 'sample_pkg/__init__.py').write_text(code)
        environment = SimpleNamespace(python_executable=sys.executable)

        def run(argv, *, stdout_path, stderr_path, timeout, isolated_launcher):
            self.assertTrue(isolated_launcher)
            self.assertEqual(argv[1:3], ['-I', '-B'])
            result = subprocess.run(argv, cwd=self.repo, capture_output=True, timeout=timeout)
            Path(stdout_path).write_bytes(result.stdout)
            Path(stderr_path).write_bytes(result.stderr)
            return {'exit_code': result.returncode, 'timed_out': False}

        environment.run_process = run
        result = probe_runtime(environment, ['sample_pkg'], self.root / 'probe', metadata_only=True)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['imports'], {})
        self.assertFalse(marker.exists())
        self.assertEqual(result['python'], sys.version)

    def test_metadata_checker_inspection_error_stays_unknown(self):
        environment = SimpleNamespace(python_executable=sys.executable)

        def run(argv, *, stdout_path, stderr_path, **kwargs):
            # 在独立进程中注入元数据查询失败，
            # 不改动全局导入或实际容器、宿主机环境。
            wrapper = (
                'import importlib.util, runpy, shutil, sys\n'
                'shutil.which = lambda name: None\n'
                'def fail(name):\n    raise PermissionError("metadata denied")\n'
                'importlib.util.find_spec = fail\nrunpy.run_path(sys.argv[1])\n'
            )
            result = subprocess.run(
                [sys.executable, '-I', '-B', '-c', wrapper, argv[-1]],
                capture_output=True,
                timeout=20,
            )
            Path(stdout_path).write_bytes(result.stdout)
            Path(stderr_path).write_bytes(result.stderr)
            return {'exit_code': result.returncode, 'timed_out': False}

        environment.run_process = run
        result = probe_runtime(environment, (), self.root / 'unknown', metadata_only=True)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['checkers'], dict.fromkeys(('pytest', 'flake8', 'ruff')))
        self.assertTrue(
            all('PermissionError' in error for error in result['checker_errors'].values())
        )

    def test_legacy_eval_probe_keeps_its_original_fact_and_report_fields(self):
        from agent.execution.local import LocalEnvironment

        (self.repo / 'sample_pkg').mkdir()
        (self.repo / 'sample_pkg/__init__.py').write_text('VERSION = 1\n')
        environment = LocalEnvironment(self.repo, artifact_root=self.root / 'legacy')
        environment.runtime_facts = {'backend': 'previous metadata'}
        report = probe_runtime(environment, ['sample_pkg'], environment.artifact_root / 'probe')
        self.assertTrue(report['ok'], report)
        self.assertEqual(set(environment.runtime_facts), {'python', 'checkers', 'note'})
        self.assertNotIn('checker_errors', report)


if __name__ == '__main__':
    unittest.main()
