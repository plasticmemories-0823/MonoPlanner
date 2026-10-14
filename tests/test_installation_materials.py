"""安装材料与工作目录分离，明确引用的资料仍可使用。"""

import hashlib
import unittest
from unittest.mock import patch

from agent.execution.installation_materials import collect
from agent.execution.managed import ManagedEnvironments
from agent.execution.managed_plan import inspect_project, PreparationError
from test_managed_environment import ManagedFixture


class InstallationMaterialsTests(ManagedFixture):
    def test_no_project_never_reads_large_unrelated_file(self):
        large = self.repo / 'downloads.bin'
        with large.open('wb') as stream:
            stream.truncate(48 * 1024 * 1024)
        plan = inspect_project(self.repo)
        with patch(
            'agent.execution.installation_materials.project_files',
            side_effect=AssertionError('no scan'),
        ):
            self.assertEqual(collect(plan), {})

    def test_requirements_only_collects_recursive_manifests_not_other_projects(self):
        self.write('requirements.txt', '-r config/requirements.txt\n')
        self.write('config/requirements.txt', 'packaging==25.0\n')
        self.write('downloads/big.bin', 'irrelevant')
        self.write('other-project/package.json', '{}')
        self.assertEqual(
            set(collect(inspect_project(self.repo))),
            {'requirements.txt', 'config/requirements.txt'},
        )

    def test_package_source_uses_ignore_rules_and_nested_negation(self):
        self.write('backend/pyproject.toml', '[project]\nname="mine"\nversion="1"\n')
        self.write('.gitignore', '*.tmp\n')
        self.write('backend/.gitignore', '!keep.tmp\ncache/\n')
        self.write('backend/keep.tmp', 'needed')
        self.write('backend/drop.tmp', 'ignored')
        self.write('backend/cache/huge.bin', 'ignored')
        self.write('backend/module.py', 'VALUE=1')
        self.write('unrelated/project.dat', 'unrelated')
        files = collect(inspect_project(self.repo))
        self.assertIn('backend/module.py', files)
        self.assertIn('backend/keep.tmp', files)
        self.assertNotIn('backend/drop.tmp', files)
        self.assertNotIn('backend/cache/huge.bin', files)
        self.assertNotIn('unrelated/project.dat', files)

    def test_node_install_hook_gets_source_and_does_not_copy_sibling(self):
        self.write('frontend/package.json', '{"scripts":{"prepare":"node scripts/setup.js"}}')
        self.write('frontend/scripts/setup.js', 'console.log(1)')
        self.write('data/keep.txt', 'not needed for install')
        files = collect(inspect_project(self.repo))
        self.assertIn('frontend/scripts/setup.js', files)
        self.assertNotIn('data/keep.txt', files)

    def test_local_editable_package_is_copied_without_cache_invalidation_on_source_edit(self):
        self.write('requirements.txt', '-e ./libs/helper\n')
        self.write('libs/helper/setup.py', 'from setuptools import setup; setup(name="helper")')
        self.write('libs/helper/helper.py', 'VALUE=1')
        before = inspect_project(self.repo)
        self.assertIn('libs/helper/helper.py', collect(before))
        self.write('libs/helper/helper.py', 'VALUE=2')
        self.assertEqual(before.fingerprint, inspect_project(self.repo).fingerprint)

    def test_explicit_ignored_requirements_file_is_included(self):
        self.write('requirements.txt', '-r data/dependencies.txt\n')
        self.write('data/dependencies.txt', 'packaging==25.0\n')
        self.write('.gitignore', 'data/\n')
        self.assertIn('data/dependencies.txt', collect(inspect_project(self.repo)))

    def test_copy_detects_change_after_preflight(self):
        self.write('requirements.txt', 'packaging==25.0')
        plan = inspect_project(self.repo)
        plan.materials = collect(plan)
        self.write('requirements.txt', 'packaging==24.0')
        work = self.root / 'work'
        work.mkdir()
        with self.assertRaisesRegex(PreparationError, '变化'):
            ManagedEnvironments(self.root / 'state')._copy(plan, work, None)

    def test_size_error_includes_path_limit_and_reason_before_copy(self):
        self.write('backend/pyproject.toml', '[project]\nname="mine"\nversion="1"\n')
        self.write('backend/huge.dat', 'x' * 100)
        with patch('agent.execution.installation_materials.MAX_BYTES', 50):
            with self.assertRaisesRegex(PreparationError, 'huge.dat.*限制.*纳入原因'):
                collect(inspect_project(self.repo))

    def test_explicit_readme_is_included_even_when_ignored(self):
        self.write(
            'pyproject.toml', '[project]\nname="mine"\nversion="1"\nreadme="data/README.md"\n'
        )
        self.write('data/README.md', 'Package description')
        self.write('.gitignore', 'data/\n')
        self.assertIn('data/README.md', collect(inspect_project(self.repo)))

    def test_workspace_members_in_custom_directory_are_collected(self):
        self.write('package.json', '{"workspaces":["modules/*"]}')
        self.write('modules/helper/package.json', '{"name":"helper"}')
        self.write('unrelated/package.json', '{"name":"other"}')
        plan = inspect_project(self.repo)
        self.assertEqual({p['path'] for p in plan.projects}, {'.', 'modules/helper'})
        self.assertEqual(set(collect(plan)), {'package.json', 'modules/helper/package.json'})

    def test_install_hook_source_change_invalidates_cached_dependencies(self):
        self.write('package.json', '{"scripts":{"postinstall":"node patch.js"}}')
        self.write('patch.js', 'console.log(1)')
        before = inspect_project(self.repo).fingerprint
        self.write('patch.js', 'console.log(2)')
        self.assertNotEqual(before, inspect_project(self.repo).fingerprint)

    def test_pnpm_workspace_patterns_and_exclusions(self):
        self.write('package.json', '{"packageManager":"pnpm@10.17.1"}')
        self.write('pnpm-workspace.yaml', 'packages:\n  - modules/*\n  - "!modules/unused"\n')
        self.write('modules/helper/package.json', '{"name":"helper"}')
        self.write('modules/unused/package.json', '{"name":"unused"}')
        plan = inspect_project(self.repo)
        self.assertEqual({p['path'] for p in plan.projects}, {'.', 'modules/helper'})
        self.assertNotIn('modules/unused/package.json', collect(plan))

    def test_explicit_local_package_can_override_parent_ignore(self):
        self.write('requirements.txt', '-e ./libs/helper\n')
        self.write('libs/helper/setup.py', 'from setuptools import setup; setup(name="helper")')
        self.write('libs/helper/helper.py', 'VALUE=1')
        self.write('.gitignore', 'libs/\n')
        self.assertIn('libs/helper/helper.py', collect(inspect_project(self.repo)))

    def test_two_projects_without_dependencies_share_prepared_base(self):
        from test_managed_environment import FakeDockerManager

        manager = FakeDockerManager(self.root / 'state')
        other = self.root / 'another-project'
        other.mkdir()
        first = manager.prepare(self.repo)
        second = manager.prepare(other)
        self.assertEqual(first['image'], second['image'])
        self.assertEqual(manager.installs, 1)
        self.assertEqual(manager.status(other)['materials'], {'file_count': 0, 'total_bytes': 0})

    def test_package_can_reference_readme_inside_workspace_parent(self):
        self.write('README.md', 'Project description')
        self.write(
            'backend/pyproject.toml', '[project]\nname="mine"\nversion="1"\nreadme="../README.md"\n'
        )
        self.assertIn('README.md', collect(inspect_project(self.repo)))
        self.write(
            'backend/pyproject.toml',
            '[project]\nname="mine"\nversion="1"\nreadme="../../outside.md"\n',
        )
        with self.assertRaisesRegex(PreparationError, '超出'):
            inspect_project(self.repo)
