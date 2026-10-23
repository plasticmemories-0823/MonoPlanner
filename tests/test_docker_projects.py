"""项目配置不会授予宿主机执行权限或隐式安装镜像。"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.execution.docker import DockerEnvironment
from agent.execution.projects import (
    EnvironmentRequired,
    ProjectEnvironments,
    local_images,
    require_binding,
    workspace_binding,
)
from agent.execution.task_docker import TaskDocker
from agent.execution.existing import command_for


class ProjectDockerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.image = 'sha256:' + 'a' * 64
        self.runtime = {'python': '/usr/local/bin/python', 'conda_env': ''}
        guard = patch.dict(
            os.environ,
            {
                'MONOPLANER_STATE_DIR': str(self.root / 'state'),
                'MONOPLANER_EVALUATION_WORKSPACES': '{}',
            },
        )
        guard.start()
        self.addCleanup(guard.stop)

    def configure(self):
        inspected = json.dumps([{'Id': self.image, 'Os': 'linux', 'Architecture': 'amd64'}])
        with patch('agent.execution.projects.docker_query', return_value=inspected) as query:
            value = ProjectEnvironments().save(
                {'repo': str(self.repo), 'image': self.image, 'runtime': self.runtime}
            )
        self.assertEqual(query.call_args.args[0], ['image', 'inspect', self.image])
        return value

    def test_missing_configuration_is_lazy_and_cannot_fall_back(self):
        with patch(
            'agent.execution.projects.docker_query',
            side_effect=AssertionError('no Docker for lookup'),
        ):
            self.assertEqual(workspace_binding(str(self.repo)), {'backend': 'docker'})
        with self.assertRaises(EnvironmentRequired):
            require_binding(str(self.repo))

    def test_project_configuration_roundtrip_and_pinned_task_is_independent(self):
        saved = self.configure()
        self.assertEqual(workspace_binding(str(self.repo)), saved)
        frozen = require_binding(str(self.repo))
        second = dict(saved, runtime={'python': '/usr/bin/python3', 'conda_env': ''})
        from agent.storage.preferences import atomic_write

        atomic_write(ProjectEnvironments().path, json.dumps({str(self.repo): second}).encode())
        self.assertEqual(require_binding(str(self.repo), frozen), saved)
        self.assertEqual(workspace_binding(str(self.repo)), second)

    def test_invalid_parameters_never_call_docker(self):
        for image, runtime in [
            ('latest', self.runtime),
            (self.image, {'python': 'C:\\python.exe', 'conda_env': ''}),
            (self.image, {'python': '/bin/python', 'conda_env': 'testbed; echo bad'}),
        ]:
            with self.subTest(image=image, runtime=runtime), patch(
                'agent.execution.projects.docker_query'
            ) as query:
                with self.assertRaises(ValueError):
                    ProjectEnvironments().save(
                        {'repo': str(self.repo), 'image': image, 'runtime': runtime}
                    )
                query.assert_not_called()

    def test_image_list_deduplicates_tags_without_pull(self):
        row = {'ID': self.image, 'Repository': 'project', 'Tag': 'v1', 'Size': '20MB'}
        with patch(
            'agent.execution.projects.docker_query',
            side_effect=['linux', '\n'.join(json.dumps(v) for v in [row, dict(row, Tag='v2')])],
        ) as query:
            images = local_images()['images']
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]['names'], ['project:v1', 'project:v2'])
        self.assertNotIn('pull', repr(query.call_args_list))

    def test_runtime_and_snapshot_options_preserve_plain_python(self):
        env = DockerEnvironment(
            self.repo,
            self.image,
            'plain',
            artifact_root=self.root / 'artifacts',
            runtime=self.runtime,
        )
        self.assertEqual(env.python_executable, '/usr/local/bin/python')
        self.assertNotIn('activate', repr(env._activated_argv(['python', '-V'], False)))
        self.assertEqual(env.clone_options()['runtime'], self.runtime)
        self.assertIn(
            'activate testbed',
            repr(DockerEnvironment(self.repo, self.image, 'conda')._activated_argv('ls', True)),
        )

    def test_existing_entries_use_container_python_not_host_venv(self):
        (self.repo / 'script.py').write_text('print(1)')
        (self.repo / '.venv/Scripts').mkdir(parents=True)
        (self.repo / '.venv/Scripts/python.exe').write_bytes(b'not-a-runtime')
        env = DockerEnvironment(self.repo, self.image, 'argv', runtime=self.runtime)
        command, _, _ = command_for(str(self.repo), 'script.py', ['中文\\quote"'], env)
        self.assertEqual(command, ['/usr/local/bin/python', '/testbed/script.py', '中文\\quote"'])

    def test_materials_are_filtered_copies_and_single_file_does_not_grant_parent(self):
        docs = self.root / 'docs'
        docs.mkdir()
        (docs / 'good.txt').write_text('public')
        (docs / '.env').write_text('secret')
        env = TaskDocker(
            self.repo,
            self.image,
            self.root / 'state/tasks/one',
            runtime=self.runtime,
            read_sources=[{'path': str(docs), 'kind': 'directory'}],
        )
        copy = Path(env.material_mounts[str(docs)])
        self.assertEqual((copy / 'good.txt').read_text(), 'public')
        self.assertFalse((copy / '.env').exists())
        self.assertEqual(env.host_to_container(docs / 'good.txt'), '/agent-materials/0/good.txt')
        single = TaskDocker(
            self.repo,
            self.image,
            self.root / 'state/tasks/two',
            runtime=self.runtime,
            read_sources=[{'path': str(docs / 'good.txt'), 'kind': 'file'}],
        )
        self.assertEqual(single.host_to_container(docs / 'good.txt'), '/agent-materials/0')
        with self.assertRaises(ValueError):
            single.host_to_container(docs / '.env')


    def test_private_launcher_credentials_cannot_be_mounted_even_with_custom_state(self):
        data = self.root / 'local/MonoPlanner'
        data.mkdir(parents=True)
        with patch.dict(os.environ, {'LOCALAPPDATA': str(data.parent)}):
            for repo in (data, data.parent, data / 'inside'):
                repo.mkdir(exist_ok=True)
                with self.subTest(repo=repo), self.assertRaises(ValueError):
                    TaskDocker(
                        repo, self.image, self.root / 'state/tasks/private', runtime=self.runtime
                    )


    def test_environment_routes_do_not_start_containers_or_accept_foreign_origin(self):
        from tests.http_client import TestClient
        from agent.server.app import app

        with TestClient(app) as client, patch(
            'agent.server.app.local_images', return_value={'images': []}
        ), patch.object(TaskDocker, 'start', side_effect=AssertionError('settings cannot execute')):
            self.assertEqual(client.get('/execution/images').json(), {'images': []})
            response = client.get('/workspace/execution', params={'repo': str(self.repo)})
            self.assertEqual(response.json(), {'backend': 'docker'})
            value = {'repo': str(self.repo), 'image': self.image, 'runtime': self.runtime}
            self.assertEqual(
                client.post(
                    '/workspace/execution',
                    json=value,
                    headers={'Origin': 'https://untrusted.example'},
                ).status_code,
                403,
            )
            inspected = json.dumps([{'Id': self.image, 'Os': 'linux', 'Architecture': 'amd64'}])
            with patch('agent.execution.projects.docker_query', return_value=inspected):
                response = client.post('/workspace/execution', json=value)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['runtime'], self.runtime)

    def test_cli_resume_keeps_completed_reads_and_first_delivery(self):
        from contextlib import redirect_stdout
        import io
        from agent.cli.tasks import main
        from agent.application.work_bridge import run_work
        from tests.fixtures.models import (
            FixtureDocker,
            ScriptedModel,
            investigation,
            BlockedModel,
        )

        self.configure()
        for name in ('one.py', 'two.py'):
            (self.repo / name).write_text('VALUE = 1\n')
        folder = self.root / 'cli-state'
        model = ScriptedModel([investigation(['one.py', 'two.py'])])
        selected_model = model

        def execute(*args, **kwargs):
            return run_work(*args, environment_factory=FixtureDocker,
                            model=selected_model, **kwargs)

        with patch('agent.cli.tasks.run_work', side_effect=execute), redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        '--state-dir',
                        str(folder),
                        '--repo',
                        str(self.repo),
                        '--task',
                        '直接读取',
                        '--pause-after-steps',
                        '1',
                    ]
                ),
                0,
            )
        job = json.loads((folder / 'bridge-job.json').read_text(encoding='utf-8'))
        records = folder / 'tasks' / job['task_id']
        scope = json.loads((records / 'project-scope.json').read_text(encoding='utf-8'))
        self.assertEqual(scope['root'], str(self.repo))
        self.assertEqual(scope['excluded_projects'], [])
        before = [(p, p.read_bytes()) for p in records.glob('plans/*/*/result.json')]
        self.assertEqual(len(before), 1)
        first_request = (records / 'requests/call-0001/request.json').read_bytes()
        selected_model = BlockedModel()
        with patch('agent.cli.tasks.run_work', side_effect=execute), redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--state-dir', str(folder), '--continue']), 1)
        self.assertEqual(len(list(records.glob('plans/*/*/result.json'))), 2)
        self.assertEqual(before[0][0].read_bytes(), before[0][1])
        self.assertEqual((records / 'requests/call-0001/request.json').read_bytes(), first_request)
        self.assertEqual(
            json.loads((records / 'project-scope.json').read_text(encoding='utf-8')), scope,
        )
        self.assertEqual(len(model.calls), 1)
