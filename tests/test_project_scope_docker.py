"""根目录工作时，其他 Git 项目不可通过主容器或恢复过程重新暴露。"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agent.execution.docker import DockerEnvironment
from agent.execution.task_docker import TaskDocker
from agent.source.scope import ProjectScope
from agent.source.snapshots import SourceSnapshot
from agent.storage.records import save


class ProjectScopeDockerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'workspace'
        self.repo.mkdir()
        (self.repo / 'animation.html').write_text('<svg/>', encoding='utf-8')
        for name, marker_file in [('other', False), ('nested/worktree', True)]:
            folder = self.repo / name
            folder.mkdir(parents=True)
            if marker_file:
                (folder / '.git').write_text('gitdir: /external/git', encoding='utf-8')
            else:
                (folder / '.git').mkdir()
            (folder / 'private.txt').write_text('excluded original', encoding='utf-8')
        self.scope = ProjectScope.discover(self.repo)
        self.image = 'sha256:' + 'a' * 64
        self.cid = 'b' * 64
        self.task = self.root / 'state/tasks/scoped'
        environment = patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root / 'state')})
        environment.start()
        self.addCleanup(environment.stop)

    def environment(self, **kwargs):
        return DockerEnvironment(
            self.repo, self.image, 'scoped', artifact_root=self.root / 'evidence',
            project_scope=self.scope, **kwargs,
        )

    def task_environment(self, scope=None):
        return TaskDocker(
            self.repo, self.image, self.task,
            project_scope=self.scope.to_record() if scope is None else scope,
        )

    def inspected(self, env):
        return {
            'Id': self.cid, 'Image': self.image, 'Name': '/' + env.container_name,
            'State': {'Running': True},
            'HostConfig': {'NetworkMode': 'none', 'CapDrop': ['ALL'],
                           'SecurityOpt': ['no-new-privileges']},
            'Mounts': [
                {'Type': 'bind', 'Source': str(env.repo_root),
                 'Destination': '/testbed', 'RW': True},
                {'Type': 'bind', 'Source': str(env.artifact_root),
                 'Destination': '/agent-artifacts', 'RW': True},
                *[{'Type': 'bind', 'Source': source, 'Destination': target, 'RW': False}
                  for target, source in env.project_masks.items()],
            ],
        }

    def test_main_container_masks_git_directory_and_worktree_without_copying_them(self):
        # 不依赖聊天入口传参，CLI 和其他执行入口也使用同一目录边界。
        env = DockerEnvironment(
            self.repo, self.image, 'scoped', artifact_root=self.root / 'evidence',
        )
        with patch.object(env, '_docker', return_value=subprocess.CompletedProcess(
            [], 0, self.cid, '',
        )) as docker:
            env.start()
        args = docker.call_args.args[0]
        mounts = [args[i + 1] for i, arg in enumerate(args) if arg == '--mount']
        self.assertEqual(len(mounts), 4)
        self.assertEqual(set(env.project_masks), {'/testbed/other', '/testbed/nested/worktree'})
        for target, source in env.project_masks.items():
            self.assertIn(f'type=bind,source={source},target={target},readonly', mounts)
            self.assertEqual(list(Path(source).iterdir()), [])
            self.assertFalse(Path(source).is_relative_to(env.artifact_root))
            self.assertFalse(Path(source).is_relative_to(self.repo))
        self.assertEqual(args[args.index('--pull') + 1], 'never')
        self.assertEqual((self.repo / 'other/private.txt').read_text(), 'excluded original')
        self.assertEqual(env.host_to_container(self.repo / 'animation.html'), '/testbed/animation.html')
        with self.assertRaisesRegex(ValueError, '排除'):
            env.host_to_container(self.repo / 'other/private.txt')

    def test_new_project_mounts_only_selected_child(self):
        child = self.repo / 'pelican-bike'
        child.mkdir()
        env = DockerEnvironment(
            child, self.image, 'new-project', artifact_root=self.root / 'new-evidence',
            project_scope=ProjectScope.discover(child),
        )
        with patch.object(env, '_docker', return_value=subprocess.CompletedProcess(
            [], 0, self.cid, '',
        )) as docker:
            env.start()
        mounts = [arg for arg in docker.call_args.args[0] if arg.startswith('type=bind,')]
        self.assertEqual(len(mounts), 2)
        self.assertIn(f'type=bind,source={child},target=/testbed', mounts)
        self.assertFalse(env.project_masks)

    def test_recovery_requires_same_scope_and_exact_readonly_masks(self):
        initial = self.task_environment()
        save(initial.binding_path, initial._binding(self.cid))
        for failure in ('none', 'missing', 'writable', 'different-source'):
            with self.subTest(failure=failure):
                actual = self.inspected(initial)
                if failure == 'missing':
                    actual['Mounts'].pop()
                elif failure == 'writable':
                    actual['Mounts'][-1]['RW'] = True
                elif failure == 'different-source':
                    actual['Mounts'][-1]['Source'] = str(self.repo)
                result = subprocess.CompletedProcess([], 0, json.dumps([actual]), '')
                with patch.object(TaskDocker, '_docker', return_value=result) as docker:
                    if failure == 'none':
                        restored = self.task_environment()
                        self.assertEqual(restored._container_id, self.cid)
                    else:
                        with self.assertRaisesRegex(ValueError, 'isolation'):
                            self.task_environment()
                    self.assertEqual(docker.call_count, 1)
        with patch.object(TaskDocker, '_docker') as docker:
            binding = initial._binding(self.cid)
            binding.pop('project_scope')
            save(initial.binding_path, binding)
            with self.assertRaisesRegex(ValueError, 'binding'):
                TaskDocker(self.repo, self.image, self.task)
            docker.assert_not_called()

    def test_scope_marker_changes_stop_before_any_container_command(self):
        for change in ('new', 'removed'):
            with self.subTest(change=change):
                env = self.environment()
                if change == 'new':
                    marker = self.repo / 'new/.git'
                    marker.mkdir(parents=True)
                else:
                    marker = self.repo / 'other/.git'
                    marker.rmdir()
                with patch.object(env, '_docker') as docker:
                    with self.assertRaisesRegex(ValueError, '已变化'):
                        env.start()
                    docker.assert_not_called()
                if change == 'new':
                    marker.rmdir()
                    marker.parent.rmdir()
                else:
                    marker.mkdir()

    def test_mask_must_remain_empty_and_missing_saved_mask_is_not_recreated(self):
        env = self.task_environment()
        save(env.binding_path, env._binding(self.cid))
        mask = Path(next(iter(env.project_masks.values())))
        (mask / 'injected.txt').write_text('must not appear', encoding='utf-8')
        with patch.object(TaskDocker, '_docker') as docker:
            with self.assertRaisesRegex(ValueError, '空目录'):
                self.task_environment()
            docker.assert_not_called()
        (mask / 'injected.txt').unlink()
        mask.rmdir()
        with patch.object(TaskDocker, '_docker') as docker:
            with self.assertRaisesRegex(ValueError, '丢失'):
                self.task_environment()
            docker.assert_not_called()
        self.assertFalse(mask.exists())

    def test_output_limit_resume_checks_new_projects_before_restarting_same_container(self):
        env = self.task_environment()
        env._container_id = self.cid
        (self.repo / 'new/.git').mkdir(parents=True)
        with patch.object(env, '_docker') as docker:
            with self.assertRaisesRegex(ValueError, '范围已变化'):
                env.restart_after_output_limit({})
            docker.assert_not_called()

    def test_excluded_directory_replacement_cannot_rebind_existing_container(self):
        env = self.task_environment()
        save(env.binding_path, env._binding(self.cid))
        (self.repo / 'other').rename(self.root / 'original-other')
        (self.repo / 'other/.git').mkdir(parents=True)
        with patch.object(TaskDocker, '_docker') as docker:
            with self.assertRaisesRegex(ValueError, 'binding'):
                self.task_environment()
            docker.assert_not_called()

    def test_dependency_mount_cannot_punch_through_excluded_project(self):
        with self.assertRaisesRegex(ValueError, '排除'):
            self.environment(managed={'node_modules': [
                {'relative': 'other', 'image_path': '/opt/monoplanner/node/other/node_modules'},
            ]})

    def test_mask_original_cannot_be_exposed_through_a_second_mount(self):
        env = self.environment(material_mounts={str(self.root): str(self.root)})
        with patch.object(env, '_docker') as docker:
            with self.assertRaisesRegex(ValueError, '可访问'):
                env.start()
            docker.assert_not_called()

    def test_mask_is_registered_once_under_the_owning_task(self):
        env = self.task_environment()
        from agent.storage.resources import resource_catalog

        catalog = resource_catalog(self.root / 'state')
        with catalog._db() as db:
            row = db.execute(
                'SELECT r.kind,r.location,r.size_bytes,f.owner FROM resources r '
                'JOIN resource_refs f ON r.id=f.resource WHERE r.id=?',
                ('project-mask:' + env.container_name,),
            ).fetchone()
        self.assertEqual(row['kind'], 'temporary')
        self.assertEqual(row['size_bytes'], 0)
        self.assertEqual(row['owner'], 'task:scoped')
        self.assertEqual(row['location'], next(iter(env.project_masks.values())))
        with patch('agent.storage.resources.ResourceCatalog.register') as register:
            env._prepare_project_masks()
            register.assert_not_called()

    def test_snapshot_clone_does_not_recreate_excluded_directories_or_original_mounts(self):
        env = self.environment()
        snapshot = SourceSnapshot.capture(
            self.repo, self.root / 'snapshot', project_scope=self.scope,
        )
        clone = DockerEnvironment(
            snapshot.root, self.image, 'clone', artifact_root=self.root / 'clone-evidence',
            **env.clone_options(),
        )
        with patch.object(clone, '_docker', return_value=subprocess.CompletedProcess(
            [], 0, self.cid, '',
        )) as docker:
            clone.start()
        self.assertFalse(clone.project_masks)
        self.assertFalse((snapshot.root / 'other').exists())
        self.assertFalse((snapshot.root / 'nested/worktree').exists())
        mounts = [arg for arg in docker.call_args.args[0] if arg.startswith('type=bind,')]
        self.assertEqual(len(mounts), 2)
        self.assertTrue(any(f'source={snapshot.root},target=/testbed' in value for value in mounts))
        self.assertNotIn(f'source={self.repo},', '\n'.join(mounts))


if __name__ == '__main__':
    unittest.main()
