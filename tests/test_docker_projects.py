"""项目配置不会授予宿主机执行权限或隐式安装镜像。"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.execution.docker import DockerEnvironment, runtime_profile
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

    def test_plain_chat_requires_no_image_or_docker(self):
        from test_web_graph import ConversationTests, ScriptedModel, text_reply

        fixture = ConversationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        (fixture.root / 'runs/project-environments.json').unlink()
        with patch.object(TaskDocker, 'start', side_effect=AssertionError('no container')):
            model = ScriptedModel([text_reply()])
            result = fixture.invoke(model)
        self.assertEqual(result['status'], 'chat_complete')
        self.assertNotIn('cmd.exe', model.calls[0][0].content)
        self.assertNotIn('Unix commands and heredocs are not available', model.calls[0][0].content)

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

    def test_missing_image_preserves_first_work_response_and_waits(self):
        from test_web_graph import ConversationTests, ScriptedModel, start_reply, investigation

        fixture = ConversationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        (fixture.root / 'runs/project-environments.json').unlink()
        model = ScriptedModel([start_reply(), investigation(['first.py'])])
        result = fixture.invoke(model, '直接修改')
        self.assertEqual(result['status'], 'awaiting_user')
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(
            list((fixture.root / 'runs/tasks' / result['task_id']).glob('plans/*/*/result.json'))
        )

    def test_permission_modes_never_select_host_execution(self):
        from test_web_graph import (
            ConversationTests,
            ScriptedModel,
            start_reply,
            investigation,
            text_reply,
            FixtureDocker,
        )
        from agent.storage.permissions import Permissions

        for mode in ('sandbox', 'ask', 'full'):
            with self.subTest(mode=mode):
                fixture = ConversationTests()
                fixture.setUp()
                try:
                    Permissions(fixture.root / 'runs').request(fixture.thread, mode)
                    model = ScriptedModel(
                        [start_reply(), investigation(['first.py']), text_reply('读取完成')]
                    )
                    with patch.object(FixtureDocker, 'start', wraps=lambda: None) as started:
                        result = fixture.invoke(model, '直接读取源码')
                    self.assertTrue(result['workspace_bound'])
                    self.assertTrue(started.called)
                    self.assertEqual(
                        fixture.store.session(fixture.thread)['execution_binding']['backend'],
                        'docker',
                    )
                finally:
                    fixture.doCleanups()

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
        from test_web_graph import (
            FixtureDocker,
            ScriptedModel,
            start_reply,
            investigation,
            BlockedModel,
        )

        self.configure()
        for name in ('one.py', 'two.py'):
            (self.repo / name).write_text('VALUE = 1\n')
        folder = self.root / 'state/tasks/cli'
        model = ScriptedModel([start_reply(), investigation(['one.py', 'two.py'])])
        with patch('agent.execution.task_docker.TaskDocker', FixtureDocker), patch(
            'agent.cli.tasks.LazyModel', return_value=model
        ), redirect_stdout(io.StringIO()):
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
        before = [(p, p.read_bytes()) for p in folder.glob('plans/*/*/result.json')]
        self.assertEqual(len(before), 1)
        with patch('agent.execution.task_docker.TaskDocker', FixtureDocker), patch(
            'agent.cli.tasks.LazyModel', return_value=BlockedModel()
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--state-dir', str(folder)]), 1)
        self.assertEqual(len(list(folder.glob('plans/*/*/result.json'))), 2)
        self.assertEqual(before[0][0].read_bytes(), before[0][1])
        self.assertEqual(len(model.calls), 2)
