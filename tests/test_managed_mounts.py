"""已准备的依赖保留在源码之外，恢复时保持身份不变。"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
import unittest
from unittest.mock import patch

from agent.execution.docker import (
    DockerEnvironment,
    managed_dependencies,
    dependency_cache_usable,
    PROXY_VARIABLES,
)
from agent.execution.task_docker import TaskDocker, inspected_host_path
from agent.source.snapshots import SourceSnapshot


class DockerFixture:
    def __init__(self):
        self.calls, self.volumes = [], {}
        self.fail_seed = False
        self.container = None
        self.seeds = {}

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        if args[:2] == ['volume', 'inspect']:
            value = self.volumes.get(args[2])
            return subprocess.CompletedProcess(
                args,
                0 if value else 1,
                json.dumps([value]) if value else '',
                '' if value else 'not found',
            )
        if args[:2] == ['volume', 'create']:
            labels = dict(
                args[i + 1].split('=', 1) for i, arg in enumerate(args) if arg == '--label'
            )
            self.volumes[args[-1]] = {
                'Name': args[-1],
                'Driver': 'local',
                'Options': None,
                'Labels': labels,
            }
            return subprocess.CompletedProcess(args, 0, args[-1], '')
        if args[0] == 'run':
            seed = args[args.index('--entrypoint') + 1] == '/bin/cp'
            return subprocess.CompletedProcess(
                args,
                1 if seed and self.fail_seed else 0,
                '' if seed else 'a' * 64,
                'seed failed' if seed and self.fail_seed else '',
            )
        if args[0] == 'create':
            name = args[args.index('--name') + 1]
            self.seeds[name] = {'Id': 'c' * 64, 'Name': '/' + name, 'State': {'Running': False},
                'Config': {'Labels': dict(args[i + 1].split('=', 1)
                                          for i, arg in enumerate(args) if arg == '--label')}}
            return subprocess.CompletedProcess(args, 0, 'c' * 64, '')
        if args[:2] == ['container', 'inspect']:
            return subprocess.CompletedProcess(args, 0, json.dumps([self.seeds[args[-1]]]), '')
        if args[:2] == ['container', 'rm']:
            self.seeds.pop(args[-1], None)
            return subprocess.CompletedProcess(args, 0, '', '')
        if args[:2] == ['start', '--attach']:
            return subprocess.CompletedProcess(args, int(self.fail_seed), '', 'seed failed' if self.fail_seed else '')
        if args[:2] == ['inspect', '--format']:
            return subprocess.CompletedProcess(args, 0, '0', '')
        if args[0] == 'inspect':
            return subprocess.CompletedProcess(args, 0, json.dumps([self.container]), '')
        return subprocess.CompletedProcess(args, 0, '', '')


class ManagedMountTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'main.js').write_text('export default 42;\n')
        self.image = 'sha256:' + 'b' * 64
        self.runtime = {'python': '/usr/local/bin/python', 'conda_env': ''}
        self.managed = {
            'node_modules': [
                {'relative': '.', 'image_path': '/opt/monoplanner/node/node_modules'},
                {
                    'relative': 'frontend',
                    'image_path': '/opt/monoplanner/node/frontend/node_modules',
                },
            ]
        }
        environment = patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root / 'state')})
        environment.start()
        self.addCleanup(environment.stop)
        self.docker = DockerFixture()
        space = patch.object(DockerEnvironment, '_dependency_space_budget',
                             return_value={'estimated': True, 'fixture': True})
        space.start()
        self.addCleanup(space.stop)

    def environment(self, **kwargs):
        return DockerEnvironment(
            self.repo,
            self.image,
            'managed-test',
            runtime=self.runtime,
            artifact_root=self.root / 'artifacts',
            managed=self.managed,
            **kwargs
        )

    def test_seed_is_isolated_once_and_tasks_and_probe_clones_mount_readonly(self):
        before = SourceSnapshot.capture(self.repo, self.root / 'before')
        env = self.environment()
        with patch.object(DockerEnvironment, '_docker', self.docker):
            env.start()
            env.start()
            clone = DockerEnvironment(
                self.repo,
                self.image,
                'managed-probe',
                artifact_root=self.root / 'probe',
                **env.clone_options()
            )
            clone.repo_read_only = clone.root_read_only = True
            clone.start()
        after = SourceSnapshot.capture(self.repo, self.root / 'after')
        self.assertEqual(before.files, after.files)
        self.assertEqual(list((self.repo / 'node_modules').iterdir()), [])
        self.assertEqual(list((self.repo / 'frontend/node_modules').iterdir()), [])
        seeds = [args for args in self.docker.calls if '/bin/cp' in args]
        self.assertEqual(len(seeds), 2)
        for args in seeds:
            self.assertEqual(args[args.index('--network') + 1], 'none')
            self.assertIn('--read-only', args)
            binds = [value for value in args if value.startswith('type=bind')]
            self.assertEqual(len(binds), 1)
            self.assertTrue(binds[0].endswith('target=/monoplanner-lifetime,readonly'))
            self.assertNotIn(str(self.repo), repr(args))
            variables = [args[i + 1] for i, arg in enumerate(args) if arg == '--env']
            self.assertTrue(all(name + '=' in variables for name in PROXY_VARIABLES))
        running = [args for args in self.docker.calls if 'exec sleep infinity' in args]
        self.assertEqual(len(running), 2)
        for args in running:
            variables = [args[i + 1] for i, arg in enumerate(args) if arg == '--env']
            self.assertTrue(all(name + '=' in variables for name in PROXY_VARIABLES))
            mounts = [args[i + 1] for i, arg in enumerate(args) if arg == '--mount']
            dependencies = [value for value in mounts if value.startswith('type=volume')]
            self.assertEqual(len(dependencies), 2)
            self.assertTrue(
                all(value.endswith(',readonly,volume-nocopy') for value in dependencies)
            )
            caches = [
                args[i + 1]
                for i, arg in enumerate(args)
                if arg == '--tmpfs' and '/node_modules/' in args[i + 1]
            ]
            self.assertEqual(len(caches), 6)
            self.assertTrue(
                all('size=67108864' in value and 'nosuid,nodev' in value for value in caches)
            )

    def test_unknown_seed_is_retained_and_never_replayed(self):
        self.docker.fail_seed = True
        with patch.object(DockerEnvironment, '_docker', self.docker):
            with self.assertRaisesRegex(RuntimeError, '准备失败'):
                self.environment().start()
            self.docker.fail_seed = False
            with self.assertRaisesRegex(RuntimeError, '结果不明'):
                self.environment().start()
        self.assertEqual(len([args for args in self.docker.calls if '/bin/cp' in args]), 1)
        record = next((self.root / 'state/managed-volumes').glob('*/completion.json'))
        self.assertEqual(json.loads(record.read_text())['status'], 'pending')

    def test_concurrent_new_tasks_wait_for_one_seed_instead_of_rejecting_pending(self):
        entered, release = Event(), Event()

        def command(environment, args, **kwargs):
            if '/bin/cp' in args and not entered.is_set():
                entered.set()
                self.assertTrue(release.wait(3))
            return self.docker(args, **kwargs)

        with patch.object(DockerEnvironment, '_docker', command), ThreadPoolExecutor(
            max_workers=2
        ) as pool:
            first = pool.submit(self.environment().start)
            self.assertTrue(entered.wait(2))
            second = pool.submit(self.environment().start)
            try:
                with self.assertRaises(TimeoutError):
                    second.result(timeout=0.05)
            finally:
                release.set()
            first.result(timeout=3)
            second.result(timeout=3)
        self.assertEqual(len([args for args in self.docker.calls if '/bin/cp' in args]), 2)

    def test_volume_lost_or_replaced_never_recreated_on_reuse(self):
        with patch.object(DockerEnvironment, '_docker', self.docker):
            env = self.environment()
            env.start()
            volume = next(iter(self.docker.volumes))
            self.docker.volumes[volume]['Options'] = {
                'device': '/private',
                'type': 'none',
                'o': 'bind',
            }
            with self.assertRaisesRegex(ValueError, '身份'):
                self.environment().start()
            self.docker.volumes.pop(volume)
            with self.assertRaises(RuntimeError):
                self.environment().start()
        self.assertEqual(len([args for args in self.docker.calls if '/bin/cp' in args]), 2)

    def test_only_future_tasks_rebuild_if_cached_dependency_volume_was_deleted(self):
        def inspect(name):
            return [self.docker.volumes[name]] if name in self.docker.volumes else None

        eligible = lambda: dependency_cache_usable(
            self.image, self.managed, root=self.root / 'state', inspect_volume=inspect
        )
        self.assertTrue(eligible())  # 首次执行尚未写入任何初始数据。
        with patch.object(DockerEnvironment, '_docker', self.docker):
            self.environment().start()
            self.assertTrue(eligible())
            records = {
                path: path.read_bytes()
                for path in (self.root / 'state/managed-volumes').glob('*/completion.json')
            }
            self.docker.volumes.pop(next(iter(self.docker.volumes)))
            self.assertFalse(eligible())
            self.assertEqual(records, {path: path.read_bytes() for path in records})
            with self.assertRaises(RuntimeError):
                self.environment()._prepare_dependencies(allow_create=False)
        self.assertEqual(len([args for args in self.docker.calls if '/bin/cp' in args]), 2)

    def test_pending_dependency_generation_is_not_eligible_and_not_modified(self):
        self.docker.fail_seed = True
        with patch.object(DockerEnvironment, '_docker', self.docker), self.assertRaises(
            RuntimeError
        ):
            self.environment().start()
        record = next((self.root / 'state/managed-volumes').glob('*/completion.json'))
        before = record.read_bytes()
        self.assertFalse(
            dependency_cache_usable(
                self.image,
                self.managed,
                root=self.root / 'state',
                inspect_volume=lambda name: (
                    [self.docker.volumes[name]] if name in self.docker.volumes else None
                ),
            )
        )
        self.assertEqual(record.read_bytes(), before)

    def test_bad_directories_and_image_sources_rejected_before_docker(self):
        for relative in (
            '../outside',
            '/outside',
            'a/../b',
            'a\\b',
            'a//b',
            'a,readonly',
            '.git',
            'x/node_modules',
        ):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                managed_dependencies(
                    self.image,
                    {
                        'node_modules': [
                            {
                                'relative': relative,
                                'image_path': '/opt/monoplanner/node/' + relative + '/node_modules',
                            }
                        ]
                    },
                )
        with self.assertRaises(ValueError):
            managed_dependencies(
                self.image, {'node_modules': [{'relative': '.', 'image_path': '/private'}]}
            )
        with self.assertRaises(ValueError):
            managed_dependencies('mutable:latest', self.managed)

    def test_data_profiles_do_not_adopt_each_others_volume_journals(self):
        first = managed_dependencies(self.image, self.managed, root=self.root / 'profile-one')
        second = managed_dependencies(self.image, self.managed, root=self.root / 'profile-two')
        self.assertNotEqual(first[0]['volume'], second[0]['volume'])
        self.assertEqual(
            first, managed_dependencies(self.image, self.managed, root=self.root / 'profile-one')
        )

    def test_existing_host_packages_are_unchanged_and_links_rejected(self):
        directory = self.repo / 'node_modules'
        directory.mkdir()
        marker = directory / 'keep.txt'
        marker.write_text('host install retained')
        with patch.object(DockerEnvironment, '_docker', self.docker):
            self.environment().start()
        self.assertEqual(marker.read_text(), 'host install retained')
        with patch(
            'agent.source.snapshots._link', side_effect=lambda path: path == directory
        ), patch.object(DockerEnvironment, '_docker', self.docker), self.assertRaisesRegex(
            ValueError, 'link'
        ):
            self.environment().start()

    def test_task_binding_freezes_volume_identity_and_rejects_writable_mount(self):
        task = self.root / 'state/tasks/task-one'
        with patch.object(DockerEnvironment, '_docker', self.docker), patch.object(
            TaskDocker, '_check_runtime'
        ):
            env = TaskDocker(
                self.repo, self.image, task, runtime=self.runtime, managed=self.managed
            )
            env.start()
            self.docker.container = {
                'Image': self.image,
                'Name': '/' + env.container_name,
                'HostConfig': {
                    'NetworkMode': 'none',
                    'CapDrop': ['ALL'],
                    'SecurityOpt': ['no-new-privileges'],
                    'Tmpfs': dict(env.dependency_caches),
                },
                'State': {'Running': True},
                'Mounts': [
                    {
                        'Type': 'bind',
                        'Source': str(env.repo_root),
                        'Destination': '/testbed',
                        'RW': True,
                    },
                    {
                        'Type': 'bind',
                        'Source': str(env.artifact_root),
                        'Destination': '/agent-artifacts',
                        'RW': True,
                    },
                    {'Type': 'bind', 'Source': str(env.lifetime().directory),
                     'Destination': '/monoplanner-lifetime', 'RW': False},
                    *(
                        {'Type': 'volume', 'Name': volume, 'Destination': target, 'RW': False}
                        for target, volume in env.dependency_mounts.items()
                    ),
                ],
            }
            restored = TaskDocker(
                self.repo, self.image, task, runtime=self.runtime, managed=self.managed
            )
            restored.start()
            self.assertEqual(restored.dependency_mounts, env.dependency_mounts)
            if os.name == 'nt':
                for mount in self.docker.container['Mounts']:
                    if mount['Type'] == 'bind':
                        source = Path(mount['Source'])
                        mount['Source'] = (
                            '/run/desktop/mnt/host/'
                            + source.drive[0].lower()
                            + '/'
                            + source.as_posix()[3:]
                        )
                translated = TaskDocker(
                    self.repo, self.image, task, runtime=self.runtime, managed=self.managed
                )
                self.assertEqual(translated._container_id, env._container_id)
            self.docker.container['Config'] = {
                'Env': ['ALL_PROXY=http://private-user:private-password@private-proxy']
            }
            with self.assertRaisesRegex(ValueError, 'isolation'):
                TaskDocker(self.repo, self.image, task, runtime=self.runtime, managed=self.managed)
            self.docker.container['Config'] = {'Env': []}
            cache = next(iter(self.docker.container['HostConfig']['Tmpfs']))
            self.docker.container['HostConfig']['Tmpfs'][cache] = 'rw,size=1073741824'
            with self.assertRaisesRegex(ValueError, 'isolation'):
                TaskDocker(self.repo, self.image, task, runtime=self.runtime, managed=self.managed)
            self.docker.container['HostConfig']['Tmpfs'] = dict(env.dependency_caches)
            self.docker.container['Mounts'][-1]['RW'] = True
            with self.assertRaisesRegex(ValueError, 'isolation'):
                TaskDocker(self.repo, self.image, task, runtime=self.runtime, managed=self.managed)
            with self.assertRaisesRegex(ValueError, 'binding'):
                TaskDocker(self.repo, self.image, task, runtime=self.runtime)
        self.assertEqual(len([args for args in self.docker.calls if '/bin/cp' in args]), 2)

    @unittest.skipUnless(os.name == 'nt', 'Docker Desktop Windows path normalization')
    def test_docker_desktop_mount_translation_rejects_escape_and_foreign_paths(self):
        self.assertEqual(
            inspected_host_path('/host_mnt/c/Users/example'), Path('C:/Users/example').resolve()
        )
        for path in ('/run/desktop/mnt/host/c/../secret', '/host_mnt/c/a/./b', '/host_mnt/c/a\\b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                inspected_host_path(path)
        self.assertNotEqual(
            inspected_host_path('/run/desktop/mnt/host/d/Users/example'),
            Path('C:/Users/example').resolve(),
        )


if __name__ == '__main__':
    unittest.main()
