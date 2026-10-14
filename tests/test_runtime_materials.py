"""生成的运行时输入可在独立副本中保留；无需模型或 Docker。"""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent.execution.local import LocalEnvironment
from eval.swebench_runner import prepare_base_checkout, record_initial_workspace
from agent.execution.runtime_inventory import runtime_inventory, probe_runtime
from agent.source.snapshots import SourceSnapshot, SnapshotError
from agent.execution.task_docker import TaskDocker
from agent.storage.records import digest, save
from agent.storage.recovery import environment_key


class RuntimeMaterialsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "--quiet")
        self.git("config", "core.autocrlf", "false")
        self.write("sample_pkg/__init__.py", "from ._version import version\n")
        self.write(".gitignore", "sample_pkg/_version.py\n*.so\nprivate/\n")
        self.git("add", ".")
        self.git(
            "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"
        )
        self.base = self.git("rev-parse", "HEAD").decode().strip()
        self.write("sample_pkg/_version.py", "version = '1.0'\n")
        self.write("sample_pkg/native.so", "FROZEN_NATIVE_BYTES")
        self.write("sample_pkg/.env", "KEY=PRIVATE")
        self.write("sample_pkg/__pycache__/cached.pyc", "CACHE")
        self.write("private/unrelated.py", "PRIVATE")

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], stderr=subprocess.STDOUT
        )

    def write(self, name, text):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))

    def test_inventory_keeps_runtime_modules_without_ignored_secrets_or_caches(self):
        files = runtime_inventory(self.repo)["files"]
        self.assertEqual(
            set(files), {"sample_pkg/__init__.py", "sample_pkg/_version.py", "sample_pkg/native.so"}
        )
        before = SourceSnapshot.capture(self.repo, self.root / "plain")
        self.assertNotIn("sample_pkg/_version.py", before.files)
        after = SourceSnapshot.capture(self.repo, self.root / "complete", runtime_files=files)
        self.assertEqual((after.root / "sample_pkg/_version.py").read_text(), "version = '1.0'\n")
        self.assertIn("sample_pkg/native.so", after.files)
        self.assertTrue(after.verify_integrity()["valid"])
        self.write("sample_pkg/_version.py", "version = '2.0'\n")
        self.assertEqual((after.root / "sample_pkg/_version.py").read_text(), "version = '1.0'\n")

    def test_runtime_import_probe_proves_baseline_parity_and_keeps_generation(self):
        inventory = runtime_inventory(self.repo)
        snapshot = SourceSnapshot.capture(
            self.repo, self.root / "baseline", runtime_files=inventory["files"]
        )
        for index, repo in enumerate((self.repo, snapshot.root)):
            env = LocalEnvironment(repo, artifact_root=self.root / f"logs-{index}")
            result = probe_runtime(env, inventory["packages"], env.artifact_root / "probe")
            self.assertTrue(result["ok"], result)
            self.assertEqual(env.runtime_facts["python"], result["python"])
            self.assertEqual(env.runtime_facts["checkers"], result["checkers"])
            self.assertIn("sample_pkg", result["imports"])
            self.assertTrue(Path(result["imports"]["sample_pkg"]["file"]).is_relative_to(repo))
        self.assertTrue(snapshot.verify_integrity()["valid"])
        (snapshot.root / "sample_pkg/_version.py").unlink()
        result = probe_runtime(env, inventory["packages"], env.artifact_root / "missing")
        self.assertFalse(result["ok"])

    def test_saved_initial_fingerprints_match_for_fresh_copies_but_dirty_copy_is_rejected(self):
        task = {
            "instance_id": "repo__repo-1",
            "base_commit": self.base,
            "problem_statement": "Fix bug",
            "repo": "repo/repo",
        }
        first = record_initial_workspace(self.repo, task, "sha256:fixture", self.root / "first")
        copy = self.root / "independent"
        shutil.copytree(self.repo, copy)
        prepare_base_checkout(copy, self.base, self.root / "second")
        second = record_initial_workspace(copy, task, "sha256:fixture", self.root / "second")
        self.assertEqual(first, second)
        (copy / "sample_pkg/__init__.py").write_text("# OLD SOLUTION\n")
        with self.assertRaisesRegex(RuntimeError, "source content differs"):
            prepare_base_checkout(copy, self.base, self.root / "dirty")
        self.assertNotIn("OLD SOLUTION", (self.repo / "sample_pkg/__init__.py").read_text())

    def test_runtime_manifest_cannot_override_secret_or_path_boundaries(self):
        for n, name in enumerate(
            ("../outside.py", "sample_pkg/.env", "sample_pkg/__pycache__/x.pyc")
        ):
            with self.subTest(name=name), self.assertRaises(SnapshotError):
                SourceSnapshot.capture(self.repo, self.root / f"bad-{n}", runtime_files=[name])

    def test_clean_checkout_validation_handles_long_windows_paths(self):
        name = "sample_pkg/" + "long_directory/" * 13 + "fixture_name_with_long_suffix.txt"
        self.write(name, "UNCHANGED")
        self.git("-c", "core.longpaths=true", "add", name)
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "long fixture",
        )
        base = self.git("rev-parse", "HEAD").decode().strip()
        prepare_base_checkout(self.repo, base, self.root / "long-path-check")

    def task_environment(self):
        return TaskDocker(
            self.repo,
            'sha256:' + '1' * 64,
            self.root / 'state/tasks/owned',
            runtime={'python': '/usr/bin/python3', 'conda_env': ''},
        )

    def inspected(self, environment):
        return subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                [
                    {
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
                            {
                                'Source': str(environment.repo_root),
                                'Destination': '/testbed',
                                'RW': True,
                            },
                            {
                                'Source': str(environment.artifact_root),
                                'Destination': '/agent-artifacts',
                                'RW': True,
                            },
                        ],
                    }
                ]
            ),
            '',
        )

    def test_chat_bootstrap_probe_and_acceptance_copy_keep_ignored_runtime_files(self):
        from agent.server.runtime import ConversationRunner
        from agent.execution.docker import DockerEnvironment
        from agent.execution.probe_batch import ProbeBatch

        environment = self.task_environment()
        runner = object.__new__(ConversationRunner)
        runner.environment, runner.root = environment, environment.binding_path.parent
        runner.state = {
            'history': [],
            'request': 'Fix a normal code issue',
            'public_conversation': [],
        }
        runner.rules, runner.runtime_settings, runner.history_store = 'chat rules', {}, None
        runner.optimize_source = True
        runner.activate, runner.initial_material, runner.persist = lambda: None, lambda: {}, Mock()
        runner.bootstrap_work()
        expected = set(runtime_inventory(self.repo)['files'])
        self.assertTrue(expected <= set(runner.baseline.files))
        manifest = json.loads(runner.baseline.manifest_path.read_text())
        self.assertEqual(set(manifest['runtime_files']), expected)
        self.assertEqual(runner.state['environment_key'], environment_key(environment))
        probe = ProbeBatch(environment, self.root / 'probes', source_factory=runner.evidence.view)
        with patch.object(DockerEnvironment, 'start'), patch.object(DockerEnvironment, 'close'):
            try:
                probe._start(runner.evidence.current, 20)
                self.assertTrue(expected <= set(probe.snapshot.files))
                self.assertEqual(
                    (probe.snapshot.root / 'sample_pkg/native.so').read_bytes(),
                    b'FROZEN_NATIVE_BYTES',
                )
            finally:
                probe.close()
        candidate = SourceSnapshot.capture(
            self.repo, self.root / 'candidate', runtime_files=environment.runtime_files
        )
        self.assertTrue(expected <= set(candidate.files))
        self.assertEqual(candidate.files, dict(runner.evidence.view().files))
        for original in (runner.baseline, candidate):
            self.assertNotIn('sample_pkg/.env', original.files)
            self.assertNotIn('private/unrelated.py', original.files)
            self.assertTrue(original.verify_integrity()['valid'])

    def test_resume_reuses_frozen_inventory_while_source_versions_check_actual_bytes(self):
        from agent.source.versions import SourceVersions

        environment = self.task_environment()
        environment._container_id = '2' * 64
        save(environment.binding_path, environment._binding(environment._container_id))
        baseline = SourceSnapshot.capture(
            self.repo, self.root / 'baseline', runtime_files=environment.runtime_files
        )
        versions = SourceVersions(environment, self.root / 'versions', baseline=baseline)
        previous = versions.current
        self.write('sample_pkg/native.so', 'CHANGED_NATIVE_BYTES')
        self.write('sample_pkg/new_native.so', 'NEW_IGNORED_NATIVE')
        with patch(
            'agent.execution.runtime_inventory.runtime_inventory',
            side_effect=AssertionError('no reinventory'),
        ), patch.object(TaskDocker, '_docker', return_value=self.inspected(environment)):
            resumed = self.task_environment()
        self.assertEqual(environment_key(resumed), environment_key(environment))
        self.assertNotIn('sample_pkg/new_native.so', resumed.runtime_files)
        resumed_versions = SourceVersions(
            resumed, self.root / 'resumed-versions', baseline=baseline
        )
        self.assertNotEqual(resumed_versions.current, previous)
        self.assertTrue(baseline.verify_integrity()['valid'])

    def test_missing_or_changed_runtime_manifest_cannot_replace_frozen_binding(self):
        environment = self.task_environment()
        save(environment.binding_path, environment._binding('2' * 64))
        record = environment.binding_path.parent / 'execution-runtime-materials.json'
        original = json.loads(record.read_text())
        for change in ('hash', 'file_list', 'missing'):
            with self.subTest(change=change):
                value = json.loads(json.dumps(original))
                value['files'].pop('sample_pkg/native.so')
                if change == 'file_list':
                    value['sha256'] = digest(value['files'])
                if change == 'missing':
                    record.unlink()
                else:
                    save(record, value)
                with patch.object(TaskDocker, '_docker') as docker:
                    with self.assertRaises(ValueError):
                        self.task_environment()
                    docker.assert_not_called()
                save(record, original)


if __name__ == "__main__":
    unittest.main()
