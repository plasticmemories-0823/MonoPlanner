"""项目自动准备与冻结后的求解器及任务沙盒彼此独立。"""

import concurrent.futures
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent.execution.managed import ManagedEnvironments, PreparationCancelled
from agent.execution.managed_plan import PreparationError, inspect_project, source_files, _version
from agent.execution import managed_proxy


class ManagedFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        environ = patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root / 'state')})
        environ.start()
        self.addCleanup(environ.stop)

    def write(self, relative, text):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path


class ManagedPlanTests(ManagedFixture):

    def test_empty_project_gets_general_base_without_docker_or_model(self):
        with patch(
            'subprocess.Popen', side_effect=AssertionError('no execution during inspection')
        ):
            plan = inspect_project(self.repo)
        self.assertEqual((plan.python, plan.node), ('3.12', '22'))
        self.assertEqual(plan.projects, [])
        self.assertEqual(plan.public()['mode'], 'automatic')

    def test_own_root_and_nested_constraints_locks_and_requirements_are_used(self):
        self.write(
            'backend/pyproject.toml', '[project]\nname="mine"\nrequires-python=">=3.10,<3.12"\n'
        )
        self.write('backend/uv.lock', 'version = 1\n')
        self.write('backend/requirements-dev.txt', '-r requirements/test.txt\n')
        self.write('backend/requirements/test.txt', 'pytest==8.4.2\n')
        self.write(
            'frontend/package.json',
            json.dumps({'engines': {'node': '>=20 <23'}, 'packageManager': 'pnpm@10.17.1'}),
        )
        self.write('frontend/pnpm-lock.yaml', 'lockfileVersion: 9\n')
        plan = inspect_project(self.repo)
        self.assertEqual(plan.python, '3.11')
        self.assertEqual(plan.node, '22')
        self.assertEqual({p['manager'] for p in plan.projects}, {'uv', 'pnpm'})
        self.assertIn('backend/requirements/test.txt', plan.files)
        before = plan.fingerprint
        self.write('backend/requirements/test.txt', 'pytest==8.3.5\n')
        self.assertNotEqual(inspect_project(self.repo).fingerprint, before)

    def test_ordinary_source_change_reuses_but_lock_change_invalidates(self):
        self.write('package.json', '{"dependencies":{"react":"19.0.0"}}')
        self.write('package-lock.json', '{"lockfileVersion":3}')
        before = inspect_project(self.repo).fingerprint
        self.write('index.js', 'console.log(42)')
        self.assertEqual(inspect_project(self.repo).fingerprint, before)
        self.write('package-lock.json', '{"lockfileVersion":3,"packages":{}}')
        self.assertNotEqual(inspect_project(self.repo).fingerprint, before)

    def test_requirements_local_library_is_project_scoped_and_fingerprinted(self):
        self.write('requirements.txt', '-r config/requirements.txt\n')
        self.write('config/requirements.txt', './libs/helper\n')
        self.write('libs/helper/helper.py', 'VALUE=1')
        before = inspect_project(self.repo)
        self.assertIn('libs/helper/helper.py', before.files)
        self.write('libs/helper/helper.py', 'VALUE=2')
        self.assertNotEqual(inspect_project(self.repo).fingerprint, before.fingerprint)
        self.write('config/requirements.txt', '-r ../../secret.txt')
        with self.assertRaisesRegex(PreparationError, '超出'):
            inspect_project(self.repo)

    def test_editable_source_edits_reuse_but_metadata_changes_reprepare(self):
        self.write('requirements.txt', '-e .\n-e ./libs/helper\n')
        self.write('pyproject.toml', '[project]\nname="my-project"\nversion="1.0"\n')
        self.write('libs/helper/setup.py', 'from setuptools import setup; setup(name="helper")')
        before = inspect_project(self.repo).fingerprint
        self.write('app.py', 'VALUE=2')
        self.write('libs/helper/helper.py', 'VALUE=3')
        self.assertEqual(inspect_project(self.repo).fingerprint, before)
        self.write(
            'libs/helper/setup.py',
            'from setuptools import setup; setup(name="helper", install_requires=["requests"])',
        )
        self.assertNotEqual(inspect_project(self.repo).fingerprint, before)

    def test_private_files_caches_and_keys_never_enter_preparation_copy(self):
        for relative, value in {
            '.env': 'SECRET=abc',
            '.env.production': 'SECRET=xyz',
            '.npmrc': '//registry/:_authToken=private',
            '.yarnrc.yml': 'npmAuthToken: private',
            '.git/config': 'credential helper',
            '.venv/Lib/site.py': 'host runtime',
            'node_modules/private/file': 'host deps',
            'key.txt': '-----BEGIN PRIVATE KEY-----',
            'settings.py': 'API_KEY = "a-real-looking-secret-123456789"',
            'run.py': 'print(1)',
            'config.json': '{"theme":"light"}',
        }.items():
            self.write(relative, value)
        copied = {path.as_posix() for path, _ in source_files(self.repo)}
        self.assertEqual(copied, {'run.py', 'config.json'})

    def test_credential_bearing_dependency_url_is_rejected_without_exposing_it(self):
        self.write('requirements.txt', 'https://user:secret-value@example.com/package.whl')
        with self.assertRaises(PreparationError) as caught:
            inspect_project(self.repo)
        self.assertNotIn('secret-value', str(caught.exception))
        self.write('package.json', '{"scripts":{"dev":"echo ${NORMAL_PUBLIC_SETTING}"}}')
        (self.repo / 'requirements.txt').unlink()
        self.assertEqual(inspect_project(self.repo).node, '22')

    def test_link_escape_is_rejected(self):
        outside = self.root / 'outside.txt'
        outside.write_text('private')
        linked = self.repo / 'requirements.txt'
        try:
            linked.symlink_to(outside)
        except OSError:
            self.skipTest('symlink privilege unavailable')
        with self.assertRaisesRegex(PreparationError, '链接'):
            inspect_project(self.repo)

    def test_source_bounds_stop_before_large_copy(self):
        self.write('large.txt', 'abcd')
        with patch('agent.execution.managed_plan.MAX_BYTES', 3), self.assertRaisesRegex(
            PreparationError, '过大'
        ):
            list(source_files(self.repo))

    def test_project_can_never_include_private_application_state(self):
        with self.assertRaisesRegex(PreparationError, '私有'):
            inspect_project(self.root)

    def test_explicit_version_files_and_incompatible_ranges(self):
        self.assertEqual(_version(['3.11.9'], 'python'), '3.11.9')
        self.assertEqual(_version(['^20.0.0'], 'node'), '20')
        self.write('.tool-versions', 'python 3.10\nnodejs 20\n')
        self.assertEqual(inspect_project(self.repo).python, '3.10')
        self.write('frontend/package.json', '{"engines":{"node":">=22"}}')
        with self.assertRaisesRegex(PreparationError, '互不兼容'):
            inspect_project(self.repo)

    def test_old_python_stops_before_download_without_silently_upgrading(self):
        with patch('subprocess.Popen', side_effect=AssertionError('no download')):
            for constraint in ('3.8', '3.7.17', '>=3.6,<3.9'):
                with self.subTest(constraint=constraint), self.assertRaisesRegex(
                    PreparationError, 'Python 3.9'
                ):
                    _version([constraint], 'python')

    def test_workspace_members_are_not_installed_independently(self):
        self.write('package.json', '{"workspaces":["packages/*"]}')
        self.write('packages/one/package.json', '{"name":"one"}')
        projects = inspect_project(self.repo).projects
        self.assertEqual([p['workspace_member'] for p in projects], [False, True])

    def test_private_registry_is_not_silently_replaced_with_public_registry(self):
        self.write('package.json', '{"dependencies":{"company-internal":"1.0.0"}}')
        self.write(
            '.npmrc',
            '@company:registry=https://private.example/\n//private.example/:_authToken=secret-token',
        )
        with self.assertRaisesRegex(PreparationError, '不会改用公开包源') as caught:
            inspect_project(self.repo)
        self.assertNotIn('secret-token', str(caught.exception))
        (self.repo / '.npmrc').unlink()
        self.write('pyproject.toml', '[[tool.uv.index]]\nurl="https://private.example/simple"\n')
        with self.assertRaisesRegex(PreparationError, '不会改用公开包源'):
            inspect_project(self.repo)


class FakeDockerManager(ManagedEnvironments):
    def __init__(self, root):
        super().__init__(root)
        self.calls = []
        self.images = {}
        self.failure = None
        self.installs = 0
        self.image_counter = 1
        self.containers = {}
        self.volumes, self.networks = {}, {}

    def _space_guard(self, *, initial=False):
        # 这个替身完全不连接宿主 Docker，也不依赖宿主 Docker 数据盘位置。
        return None

    def _new_image(self):
        image = 'sha256:' + format(self.image_counter, '064x')
        self.image_counter += 1
        return image

    def _stop_owned_container(self, name, metadata, owner):
        self.calls.append(['kill', metadata.get('container_id') or name])

    def _run(self, arguments, *, log, cancelled=None, timeout=60, check=True):
        from agent.execution.managed import _check_cancel

        _check_cancel(cancelled)
        self.calls.append(list(arguments))
        log.parent.mkdir(parents=True, exist_ok=True)
        stdout, code = '', 0
        if self.failure and arguments[: len(self.failure)] == self.failure:
            raise PreparationError('fixture installation failure')
        if arguments[:2] == ['image', 'inspect']:
            found = self.images.get(arguments[2])
            if found:
                stdout = json.dumps([{'Id': found, 'Os': 'linux', 'Architecture': 'amd64', 'Size': 100}])
            else:
                code = 1
        elif arguments[0] == 'pull':
            image = 'sha256:' + 'f' * 64
            self.images[arguments[-1]] = image
            self.images[image] = image
        elif arguments[0] == 'create' or arguments[:2] == ['run', '--detach']:
            name = arguments[arguments.index('--name') + 1]
            labels = dict(arguments[i + 1].split('=', 1)
                          for i, item in enumerate(arguments) if item == '--label')
            identity = format(len(self.containers) + 1, '064x')
            self.containers[identity] = {
                'Id': identity, 'Name': '/' + name,
                'Image': next(value for value in arguments if value.startswith('sha256:')),
                'Config': {'Labels': labels},
                'HostConfig': {'Memory': 2 * 1024**3, 'LogConfig': {
                    'Type': 'local', 'Config': {'max-size': '10m', 'max-file': '3'}}},
            }
            stdout = identity
        elif arguments[:2] == ['container', 'inspect']:
            value = self.containers.get(arguments[-1]) or next((value for value in self.containers.values()
                if value['Name'] == '/' + arguments[-1]), None)
            stdout, code = (json.dumps([value]), 0) if value else ('No such container', 1)
        elif arguments[:2] == ['buildx', 'rm']:
            self.containers = {key: value for key, value in self.containers.items()
                               if value['Name'] != '/buildx_buildkit_' + arguments[-1] + '0'}
            self.volumes.pop('buildx_buildkit_' + arguments[-1] + '0_state', None)
        elif arguments[:2] == ['buildx', 'build']:
            image = self._new_image()
            self.images[arguments[arguments.index('--tag') + 1]] = image
            self.images[image] = image
        elif arguments[:2] == ['network', 'create']:
            self.networks[arguments[-1]] = {
                        'Id': 'd' * 64,
                        'Labels': dict(arguments[i + 1].split('=', 1)
                                       for i, item in enumerate(arguments) if item == '--label'),
                        'Internal': True,
                        'EnableIPv6': False,
                        'Driver': 'bridge',
                        'Options': {'com.docker.network.bridge.gateway_mode_ipv4': 'isolated'},
                    }
        elif arguments[:2] == ['network', 'inspect']:
            stdout = json.dumps([self.networks[arguments[-1]]])
        elif arguments[:2] == ['volume', 'create']:
            self.volumes[arguments[-1]] = {'Labels': dict(arguments[i + 1].split('=', 1)
                                       for i, item in enumerate(arguments) if item == '--label')}
        elif arguments[:2] == ['volume', 'inspect']:
            value = self.volumes.get(arguments[-1])
            code, stdout = (0, json.dumps([value])) if value else (1, 'Error: No such volume')
        elif arguments[:2] == ['container', 'rm']:
            identity = next((key for key, value in self.containers.items()
                if key == arguments[-1] or value['Name'] == '/' + arguments[-1]), None)
            if identity:
                self.containers.pop(identity)
            else:
                code, stdout = 1, 'No such container'
        elif arguments[:2] == ['volume', 'rm']:
            self.volumes.pop(arguments[-1], None)
        elif arguments[:2] == ['start', '--attach']:
            self.installs += 1
        elif arguments[:2] == ['inspect', '--format']:
            stdout = '0\n'
        elif arguments[0] == 'cp' and arguments[1].endswith(':/monoplanner-result.json'):
            Path(arguments[2]).write_text('{"node_modules":[]}', encoding='utf-8')
        elif arguments[0] == 'commit':
            image = self._new_image()
            self.images[image] = image
            self.images[arguments[-1]] = image
            stdout = image + '\n'
        log.write_text(stdout, encoding='utf-8')
        return subprocess.CompletedProcess(arguments, code, stdout, '')


class ManagedPreparationTests(ManagedFixture):
    def setUp(self):
        super().setUp()
        self.manager = FakeDockerManager(self.root / 'state')

    def test_setup_has_no_host_mount_network_bypass_or_credential_forwarding(self):
        self.write('requirements.txt', 'requests==2.32.4')
        events = []
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'never-forward-this'}):
            binding = self.manager.prepare(self.repo, progress=events.append)
        self.assertEqual(
            binding['runtime'], {'python': '/opt/monoplanner/venv/bin/python', 'conda_env': ''}
        )
        self.assertEqual(
            [e['stage'] for e in events],
            ['checking', 'materials', 'downloading', 'installing', 'saving', 'ready'],
        )
        create = next(c for c in self.manager.calls if c[0] == 'create'
                      and c[c.index('--name') + 1].endswith('-work'))
        self.assertIn('--network', create)
        self.assertTrue(create[create.index('--network') + 1].endswith('-net'))
        self.assertIn(
            '--internal', next(c for c in self.manager.calls if c[:2] == ['network', 'create'])
        )
        self.assertNotIn('--mount', create)
        self.assertNotIn('--privileged', create)
        self.assertIn('no-new-privileges', create)
        from agent.execution.docker import PROXY_VARIABLES

        env = [create[i + 1] for i, item in enumerate(create[:-1]) if item == '--env']
        for name in PROXY_VARIABLES:
            self.assertTrue(any(item.startswith(name + '=') for item in env))
        self.assertNotIn('never-forward-this', repr(self.manager.calls))
        self.assertFalse(list((self.manager.root / 'attempts').glob('*/source')))

    def test_cache_is_per_user_project_and_manifest_changes_create_new_image(self):
        self.write('requirements.txt', 'requests==2.32.4')
        first = self.manager.prepare(self.repo)
        self.assertEqual(self.manager.prepare(self.repo), first)
        self.assertEqual(self.manager.installs, 1)
        self.write('requirements.txt', 'requests==2.32.5')
        second = self.manager.prepare(self.repo)
        self.assertNotEqual(second['image'], first['image'])
        self.assertEqual(self.manager.installs, 2)
        self.assertEqual(first['image'], 'sha256:' + '0' * 63 + '2')
        self.assertEqual(self.manager.status(self.repo)['state'], 'ready')

    def test_preparation_containers_own_deadline_when_host_controller_exits(self):
        from agent.execution.managed import SETUP_TIMEOUT, PROXY_LIFETIME

        self.manager.prepare(self.repo)
        proxy = next(c for c in self.manager.calls if c[:2] == ['run', '--detach'])
        setup = next(c for c in self.manager.calls if c[0] == 'create'
                     and c[c.index('--name') + 1].endswith('-work'))
        for command, deadline in ((proxy, PROXY_LIFETIME), (setup, SETUP_TIMEOUT)):
            self.assertIn('--init', command)
            self.assertEqual(command[command.index('--entrypoint') + 1], '/usr/bin/timeout')
            self.assertIn('--signal=TERM', command)
            self.assertIn('--kill-after=10s', command)
            self.assertIn(str(deadline) + 's', command)
        self.assertGreater(PROXY_LIFETIME, SETUP_TIMEOUT)

    def test_base_builder_is_owned_bounded_and_does_not_replace_user_default(self):
        self.manager.prepare(self.repo)
        create = next(c for c in self.manager.calls if c[:2] == ['buildx', 'create'])
        build = next(c for c in self.manager.calls if c[:2] == ['buildx', 'build'])
        self.assertNotIn('--use', create)
        self.assertIn('memory=2g', create)
        self.assertIn('cpu-quota=200000', create)
        self.assertIn('restart-policy=no', create)
        name = create[create.index('--name') + 1]
        self.assertEqual(build[build.index('--builder') + 1], name)
        self.assertIn('--load', build)
        self.assertIn(['buildx', 'rm', '--force', name], self.manager.calls)

    def test_cancel_during_base_build_kills_builder_and_preserves_its_state(self):
        def cancelled():
            return any(c[:2] == ['buildx', 'build'] for c in self.manager.calls)

        with self.assertRaises(PreparationCancelled):
            self.manager.prepare(self.repo, cancelled=cancelled)
        create = next(c for c in self.manager.calls if c[:2] == ['buildx', 'create'])
        node = create[create.index('--node') + 1]
        container = next(value for value in self.manager.containers.values()
                         if value['Name'] == '/buildx_buildkit_' + node)
        self.assertIn(['kill', container['Id']], self.manager.calls)
        self.assertFalse(any(c[:2] == ['buildx', 'rm'] for c in self.manager.calls))
        self.assertFalse(any(c[0] == 'commit' for c in self.manager.calls))

    def test_cleared_docker_cache_is_detected_and_reprepared(self):
        first = self.manager.prepare(self.repo)
        del self.manager.images[first['image']]
        next_binding = self.manager.prepare(self.repo)
        self.assertNotEqual(first['image'], next_binding['image'])
        self.assertEqual(self.manager.installs, 2)

    def test_missing_dependency_volume_rebuilds_future_binding_without_changing_old_record(self):
        first = self.manager.prepare(self.repo)
        original = self.manager._record(self.repo).read_bytes()
        with patch('agent.execution.managed.dependency_cache_usable', return_value=False):
            second = self.manager.prepare(self.repo)
        self.assertNotEqual(first['image'], second['image'])
        self.assertEqual(self.manager.installs, 2)
        self.assertEqual(first['image'], json.loads(original)['binding']['image'])

    def test_concurrent_requests_install_once(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _: self.manager.prepare(self.repo), range(2)))
        self.assertEqual(replies[0], replies[1])
        self.assertEqual(self.manager.installs, 1)

    def test_cancel_after_proxy_creation_kills_owned_container_and_preserves_unknown_scene(self):
        def cancelled():
            return any(c[:2] == ['run', '--detach'] for c in self.manager.calls)

        with self.assertRaises(PreparationCancelled):
            self.manager.prepare(self.repo, cancelled=cancelled)
        self.assertTrue(any(c[0] == 'kill' for c in self.manager.calls))
        self.assertFalse(any(c[:2] == ['rm', '--force'] for c in self.manager.calls))
        self.assertFalse(any(c[:2] == ['network', 'rm'] for c in self.manager.calls))
        self.assertFalse(any(c[0] == 'commit' for c in self.manager.calls))
        self.assertEqual(self.manager.status(self.repo)['state'], 'needs_preparation')

    def test_failed_install_does_not_replace_saved_binding(self):
        self.write('requirements.txt', 'requests==2.32.4')
        first = self.manager.prepare(self.repo)
        self.write('requirements.txt', 'requests==2.32.5')
        self.manager.failure = ['start', '--attach']
        with self.assertRaises(PreparationError):
            self.manager.prepare(self.repo)
        saved = json.loads(self.manager._record(self.repo).read_text(encoding='utf-8'))
        self.assertEqual(saved['binding'], first)
        self.assertEqual(self.manager.status(self.repo)['state'], 'needs_preparation')

    def test_display_failure_does_not_abort_preparation(self):
        def broken(event):
            raise OSError('disconnected display')

        self.assertIn('image', self.manager.prepare(self.repo, progress=broken))

    def test_manifest_change_during_install_cannot_publish_stale_ready_environment(self):
        self.write('requirements.txt', 'requests==2.32.4')

        def changed(event):
            if event['stage'] == 'installing':
                self.write('requirements.txt', 'requests==2.32.5')

        with self.assertRaisesRegex(PreparationError, '变化'):
            self.manager.prepare(self.repo, progress=changed)
        self.assertFalse(self.manager._record(self.repo).exists())


class DownloadProxyTests(unittest.TestCase):
    def test_dns_rebinding_cannot_change_validated_connect_target(self):
        result = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]
        with patch.object(socket, 'getaddrinfo', return_value=result) as resolve, patch.object(
            socket, 'socket'
        ) as make:
            connection = managed_proxy.connect_public('packages.example', 443)
        self.assertIs(connection, make.return_value)
        resolve.assert_called_once()
        connection.connect.assert_called_once_with(('93.184.216.34', 443))

    def test_any_private_dns_answer_rejects_destination_without_connection(self):
        for address in (
            '127.0.0.1',
            '10.0.0.1',
            '172.17.0.1',
            '192.168.1.1',
            '169.254.169.254',
            '::1',
            '::ffff:127.0.0.1',
            '224.0.0.1',
        ):
            with self.subTest(address=address):
                result = [
                    (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443)),
                    (socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443)),
                ]
                with patch.object(socket, 'getaddrinfo', return_value=result), patch.object(
                    socket, 'socket'
                ) as make:
                    with self.assertRaises(ValueError):
                        managed_proxy.connect_public('packages.example', 443)
                    make.assert_not_called()


class EditableSourceTests(ManagedFixture):
    def test_only_generated_import_maps_point_to_live_task_source(self):
        from agent.execution.managed_setup import bind_live_python_source

        site = self.root / 'lib/python3.12/site-packages'
        site.mkdir(parents=True)
        (site / 'editable.pth').write_text('/monoplanner-source/src\n')
        (site / '__editable___project_finder.py').write_text(
            "MAPPING={'project':'/monoplanner-source/src/project'}\n"
        )
        ordinary = site / 'ordinary.py'
        ordinary.write_text("MESSAGE='/monoplanner-source'\n")
        bind_live_python_source(self.root / 'lib')
        self.assertEqual((site / 'editable.pth').read_text(), '/testbed/src\n')
        self.assertIn('/testbed/src/project', (site / '__editable___project_finder.py').read_text())
        self.assertIn('/monoplanner-source', ordinary.read_text())


if __name__ == '__main__':
    unittest.main()
