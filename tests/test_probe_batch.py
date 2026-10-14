"""验证计划内复用及失败即停止的清理；不发起网络或模型请求。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.execution.probe_batch import ProbeBatch


class FakeDocker:
    instances = []
    outcomes = []

    def __init__(
        self,
        repo_root,
        image='test',
        container_name='test',
        timeout=5,
        artifact_root=None,
        **options
    ):
        self.repo_root, self.image = Path(repo_root), image
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.timeout, self.python_executable = timeout, 'python'
        self.starts = self.stops = 0
        self.instances.append(self)

    def clone_options(self):
        return {}

    def start(self):
        self.starts += 1

    def close(self):
        self.stops += 1

    def run_process(self, command, *, stdout_path, stderr_path, timeout, shell):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text('original output', encoding='utf-8')
        stderr_path.write_text('', encoding='utf-8')
        outcome = self.outcomes.pop(0) if self.outcomes else {'exit_code': 0}
        if isinstance(outcome, Exception):
            raise outcome
        return dict(timed_out=False, **outcome)


class ProbeBatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        FakeDocker.instances, FakeDocker.outcomes = [], []
        self.environment = FakeDocker(self.root / 'repo', artifact_root=self.root / 'base')
        self.patch = patch('agent.execution.probe_batch.DockerEnvironment', FakeDocker)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.capture = patch(
            'agent.execution.probe_batch.SourceSnapshot.capture', side_effect=self.snapshot
        ).start()
        self.addCleanup(patch.stopall)
        self.batch = ProbeBatch(self.environment, self.root / 'sessions')
        self.addCleanup(self.batch.close)

    def snapshot(self, repo, destination, **kwargs):
        source = destination / 'source'
        source.mkdir(parents=True)
        return SimpleNamespace(root=source, digest='hash', verify_integrity=lambda: {'valid': True})

    def run_probe(self, version='v1', step='one'):
        return self.batch.run('echo observed', self.root / step, 5, version)

    def test_two_commands_share_one_snapshot_and_container(self):
        self.run_probe(step='one')
        self.run_probe(step='two')
        self.assertEqual(self.capture.call_count, 1)
        child = FakeDocker.instances[-1]
        self.assertTrue(child.repo_read_only and child.root_read_only)
        self.assertEqual(child.starts, 1)
        self.batch.close()
        self.assertEqual(child.stops, 1)
        self.assertEqual(self.batch.stats['commands'], 2)

    def test_new_source_replaces_session(self):
        self.run_probe('v1', 'one')
        child = FakeDocker.instances[-1]
        self.run_probe('v2', 'two')
        self.assertEqual(child.stops, 1)
        self.assertEqual(self.capture.call_count, 2)

    def test_nonzero_exit_prevents_reusing_dirty_runtime(self):
        FakeDocker.outcomes = [{'exit_code': 3}, {'exit_code': 0}]
        self.assertEqual(self.run_probe()['exit_code'], 3)
        self.assertIsNone(self.batch.child)
        self.run_probe(step='two')
        self.assertEqual(self.capture.call_count, 2)

    def test_transport_exception_stops_owned_container(self):
        FakeDocker.outcomes = [RuntimeError('transport failed')]
        with self.assertRaisesRegex(RuntimeError, 'transport failed'):
            self.run_probe()
        self.assertIsNone(self.batch.child)
        self.assertEqual(FakeDocker.instances[-1].stops, 1)

    def test_lazy_batch_starts_no_container(self):
        self.batch.close()
        self.assertEqual(self.capture.call_count, 0)
        self.assertEqual(len(FakeDocker.instances), 1)

    def test_closed_batch_cannot_run(self):
        self.batch.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.run_probe()

    def test_each_command_has_separate_archived_logs(self):
        self.run_probe(step='one')
        original = (self.root / 'one/stdout.log').read_bytes()
        self.run_probe(step='two')
        self.assertEqual((self.root / 'one/stdout.log').read_bytes(), original)
        self.assertEqual((self.root / 'two/stdout.log').read_bytes(), original)
        self.assertFalse(self.root.is_relative_to(self.batch.runtime))


if __name__ == '__main__':
    unittest.main()
