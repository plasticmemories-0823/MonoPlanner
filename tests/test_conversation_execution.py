"""固定评测绑定保留运行时默认设置和容器身份。"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agent.execution.task_docker import TaskDocker
from agent.execution.projects import workspace_binding
from agent.storage.records import save
from agent.storage.recovery import environment_key


class ConversationExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.task = self.root / 'tasks' / 'owned-task'
        self.task.mkdir(parents=True)
        self.image, self.cid = 'sha256:' + '1' * 64, '2' * 64

    def test_default_requires_docker_configuration_and_exact_operator_binding_matches(self):
        with patch.dict(os.environ, {'MONOPLANER_EVALUATION_WORKSPACES': '{}'}):
            self.assertEqual(workspace_binding(str(self.repo)), {'backend': 'docker'})
        binding = {str(self.repo): self.image}
        with patch.dict(os.environ, {'MONOPLANER_EVALUATION_WORKSPACES': json.dumps(binding)}):
            self.assertEqual(workspace_binding(str(self.repo))['image'], self.image)
            (self.repo / 'child').mkdir()
            self.assertEqual(workspace_binding(str(self.repo / 'child')), {'backend': 'docker'})
        with patch.dict(
            os.environ, {'MONOPLANER_EVALUATION_WORKSPACES': json.dumps({str(self.repo): 'latest'})}
        ):
            with self.assertRaisesRegex(ValueError, 'pinned'):
                workspace_binding(str(self.repo))

    def inspected(self, running=True):
        return {
            'Image': self.image,
            'Name': '/monoplanner-owned-task',
            'State': {'Running': running},
            'HostConfig': {
                'NetworkMode': 'none',
                'Privileged': False,
                'CapDrop': ['ALL'],
                'SecurityOpt': ['no-new-privileges'],
            },
            'Mounts': [
                {'Source': str(self.repo), 'Destination': '/testbed'},
                {
                    'Source': str(self.root / 'execution' / 'owned-task'),
                    'Destination': '/agent-artifacts',
                },
            ],
        }

    def test_same_container_across_nodes_and_stopped_container_never_replays(self):
        first = TaskDocker(self.repo, self.image, self.task)
        with patch.object(first, '_check_runtime'), patch.object(
            first, '_docker', return_value=subprocess.CompletedProcess([], 0, self.cid, '')
        ) as launch:
            first.start()
            first.start()
            self.assertEqual(launch.call_count, 1)
        key = environment_key(first)
        for running in (True, False):
            response = subprocess.CompletedProcess([], 0, json.dumps([self.inspected(running)]), '')
            # 此测试环境检查容器身份；运行时记录复用由
            # test_runtime_facts 中的真实探测验证。
            with patch.object(TaskDocker, '_check_runtime'), patch.object(
                TaskDocker, '_docker', return_value=response
            ) as inspect:
                restored = TaskDocker(self.repo, self.image, self.task)
                self.assertEqual(environment_key(restored), key)
                if running:
                    restored.start()
                else:
                    with self.assertRaisesRegex(RuntimeError, 'not automatically replayed'):
                        restored.start()
                self.assertEqual(inspect.call_count, 1)

    def test_tampered_binding_mount_or_network_fails_without_new_launch(self):
        initial = TaskDocker(self.repo, self.image, self.task)
        save(initial.binding_path, initial._binding(self.cid))
        for change in ('mount', 'network', 'missing'):
            with self.subTest(change=change):
                value = self.inspected()
                if change == 'mount':
                    value['Mounts'][0]['Source'] = str(self.root)
                elif change == 'network':
                    value['HostConfig']['NetworkMode'] = 'default'
                response = subprocess.CompletedProcess(
                    [], int(change == 'missing'), json.dumps([value]), ''
                )
                with patch.object(TaskDocker, '_docker', return_value=response) as inspect:
                    with self.assertRaises((ValueError, RuntimeError)):
                        TaskDocker(self.repo, self.image, self.task)
                    self.assertEqual(inspect.call_count, 1)

    def test_chat_binding_is_frozen_without_starting_execution(self):
        from tests.test_web_graph import ConversationTests, ScriptedModel, text_reply

        fixture = ConversationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with patch.dict(
            os.environ,
            {'MONOPLANER_EVALUATION_WORKSPACES': json.dumps({str(fixture.repo): self.image})},
        ), patch.object(TaskDocker, 'start') as start:
            model = ScriptedModel([text_reply()])
            result = fixture.invoke(model)
            self.assertEqual(result['status'], 'chat_complete')
            self.assertIn('Linux Docker', model.calls[0][0].content)
            self.assertNotIn('Windows; shell', model.calls[0][0].content)
            self.assertEqual(
                fixture.store.session(fixture.thread)['execution_binding']['image'], self.image
            )
            start.assert_not_called()
