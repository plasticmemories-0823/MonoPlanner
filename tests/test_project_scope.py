"""任务范围只包含选定项目；其他独立项目既不读取也不成为安装输入。"""

from pathlib import Path
from threading import Event
import subprocess

import pytest

from agent.execution.installation_materials import collect, project_files
from agent.execution.local import LocalEnvironment
from agent.execution.managed_plan import inspect_project, PreparationError
from agent.execution.runtime_inventory import runtime_inventory
from agent.source.preparation import build_preparation
from agent.source.scope import ProjectScope, ProjectScopeError
from agent.source.snapshots import SourceSnapshot, SnapshotError
from agent.source.versions import SourceVersions


def write(root, name, content='VALUE = 1\n'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


def test_scope_prunes_git_directories_and_worktree_files_before_reading_children(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    write(root, 'main.py')
    (root / 'one/.git').mkdir(parents=True)
    write(root, 'one/secret.py', 'NEVER READ')
    write(root, 'nested/two/.git', 'gitdir: /not-authorized')
    write(root, 'nested/two/secret.py', 'NEVER READ')
    import agent.source.scope as module
    original = module.os.scandir

    def bounded(path):
        assert Path(path) not in {root / 'one', root / 'nested/two'}
        return original(path)

    monkeypatch.setattr(module.os, 'scandir', bounded)
    scope = ProjectScope.discover(root)
    assert scope.excluded_projects == ('nested/two', 'one')
    assert ProjectScope.from_record(scope.to_record(), root) == scope
    assert scope.excludes(Path('nested/two/secret.py'))
    assert not scope.excludes(Path('.'))
    assert not scope.excludes('nested/twenty/file.py')
    snapshot = SourceSnapshot.capture(root, tmp_path / 'capture', project_scope=scope)
    assert set(snapshot.files) == {'main.py'}


def test_frozen_scope_ignores_excluded_contents_but_detects_marker_changes(tmp_path):
    root = tmp_path / 'project'
    write(root, 'main.py')
    marker = write(root, 'other/.git', 'gitdir: elsewhere')
    scope = ProjectScope.discover(root)
    write(root, 'other/more/.git', 'independent nested')
    scope.validate(full_check=True)
    marker.unlink()
    with pytest.raises(ProjectScopeError, match='标记'):
        scope.validate()


@pytest.mark.parametrize('names', [['..'], ['.'], ['x\\y'], ['b', 'a'], ['a', 'a'], ['a', 'a/b']])
def test_scope_record_rejects_escaping_or_ambiguous_ranges(tmp_path, names):
    with pytest.raises(ProjectScopeError):
        ProjectScope.from_record({'version': 1, 'root': str(tmp_path), 'excluded_projects': names})


def test_source_versions_keep_frozen_range_and_do_not_archive_other_projects(tmp_path):
    root = tmp_path / 'project'
    write(root, 'main.py')
    write(root, 'other/.git', 'gitdir: elsewhere')
    write(root, 'other/huge.py', 'NEVER ARCHIVE')
    scope = ProjectScope.discover(root)
    baseline = SourceSnapshot.capture(root, tmp_path / 'baseline', project_scope=scope)
    environment = LocalEnvironment(root, artifact_root=tmp_path / 'artifacts')
    environment.project_scope = scope
    versions = SourceVersions(environment, tmp_path / 'versions', baseline)
    before = versions.current
    write(root, 'other/huge.py', 'CHANGED OUTSIDE THIS TASK')
    assert versions.record('outside-change') == before
    assert set(versions.view().files) == {'main.py'}
    write(root, 'new-project/.git', 'gitdir: elsewhere')
    with pytest.raises(ProjectScopeError, match='新的独立项目'):
        versions.record('new-independent-project')


def test_scoped_git_subdirectory_keeps_ancestor_and_local_ignore_without_sibling_reads(tmp_path):
    root = tmp_path / 'repository'
    write(root, '.gitignore', '*.tmp\n')
    write(root, 'child/.gitignore', 'cache/\n!keep.tmp\n')
    write(root, 'child/main.py')
    write(root, 'child/ignored.tmp')
    write(root, 'child/keep.tmp')
    write(root, 'child/cache/generated.py')
    write(root, 'sibling/never.py')
    git(root, 'init', '--quiet')
    git(root, 'add', 'child/main.py', 'sibling/never.py')
    selected = root / 'child'
    expected = {'.gitignore', 'main.py', 'keep.tmp'}
    assert set(SourceSnapshot.capture(selected, tmp_path / 'capture').files) == expected
    assert {p.as_posix() for p in project_files(selected)} == expected


def test_ignored_independent_child_uses_its_own_ignore_instead_of_empty_parent_index(tmp_path):
    root = tmp_path / 'repository'
    write(root, '.gitignore', 'child/\n')
    write(root, 'child/.gitignore', 'cache/\n*.tmp\n')
    write(root, 'child/main.py')
    write(root, 'child/cache/huge.bin')
    write(root, 'child/ignored.tmp')
    git(root, 'init', '--quiet')
    selected = root / 'child'
    expected = {'.gitignore', 'main.py'}
    assert set(SourceSnapshot.capture(selected, tmp_path / 'capture').files) == expected
    assert {p.as_posix() for p in project_files(selected)} == expected


def test_git_tracked_or_runtime_paths_cannot_restore_excluded_project(tmp_path):
    root = tmp_path / 'project'
    write(root, 'main.py')
    write(root, 'other/file.py')
    git(root, 'init', '--quiet')
    git(root, 'add', '.')
    write(root, 'other/.git', 'gitdir: elsewhere')
    assert set(SourceSnapshot.capture(root, tmp_path / 'capture').files) == {'main.py'}
    with pytest.raises(SnapshotError, match='已排除'):
        SourceSnapshot.capture(root, tmp_path / 'runtime', runtime_files=['other/file.py'])


def test_environment_discovery_does_not_adopt_a_neighboring_git_project(tmp_path):
    root = tmp_path / 'project'
    write(root, 'main.py')
    write(root, 'backend/.git', 'gitdir: elsewhere')
    write(root, 'backend/pyproject.toml', '[project]\nname="other"\nversion="1"\n')
    write(root, 'backend/requirements.txt', 'DO NOT INSTALL')
    plan = inspect_project(root)
    assert not plan.projects
    assert not plan.files
    assert not collect(plan)
    assert list(project_files(root)) == [Path('main.py')]


@pytest.mark.parametrize('manifest,content', [
    ('requirements.txt', './other\n'),
    ('requirements.txt', '-r other/requirements.txt\n'),
    ('package.json', '{"workspaces":["other"]}'),
    ('package.json', '{"dependencies":{"other":"file:other"}}'),
])
def test_explicit_dependency_into_excluded_project_fails_instead_of_omitting_it(tmp_path, manifest, content):
    root = tmp_path / 'project'
    write(root, manifest, content)
    write(root, 'other/.git', 'gitdir: elsewhere')
    write(root, 'other/requirements.txt', 'fake-package==1\n')
    with pytest.raises((ProjectScopeError, PreparationError), match='已排除'):
        inspect_project(root)


def test_runtime_inventory_and_preparation_do_not_read_independent_package(tmp_path):
    root = tmp_path / 'project'
    write(root, 'own/__init__.py', 'def ours(): pass\n')
    write(root, 'other/.git', 'gitdir: elsewhere')
    write(root, 'other/__init__.py', 'def never_index(): pass\n')
    inventory = runtime_inventory(root)
    assert inventory['packages'] == ['own']
    assert set(inventory['files']) == {'own/__init__.py'}
    baseline = SourceSnapshot.capture(root, tmp_path / 'baseline')
    environment = LocalEnvironment(root, artifact_root=tmp_path / 'artifacts')
    versions = SourceVersions(environment, tmp_path / 'versions', baseline)
    prepared = build_preparation(versions.view(), {}, Event())
    assert set(prepared['entries']) == {'own/__init__.py'}


def test_routine_validation_does_not_rediscover_entire_scope(tmp_path, monkeypatch):
    write(tmp_path, 'other/.git', 'gitdir: elsewhere')
    scope = ProjectScope.discover(tmp_path)
    monkeypatch.setattr(ProjectScope, 'discover', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('routine validation must not rediscover the tree')))
    scope.validate()
    assert scope.excludes('other/file.py')


def test_scope_discovery_is_bounded(tmp_path, monkeypatch):
    import agent.source.scope as module
    write(tmp_path, 'one/file.py')
    monkeypatch.setattr(module, 'MAX_SCOPE_ENTRIES', 1)
    with pytest.raises(ProjectScopeError, match='范围过大'):
        ProjectScope.discover(tmp_path)


def test_snapshot_integrity_cannot_hide_changes_by_adding_a_nested_git_marker(tmp_path):
    root = tmp_path / 'project'
    write(root, 'module/value.py')
    captured = SourceSnapshot.capture(root, tmp_path / 'capture')
    write(captured.root, 'module/.git', 'gitdir: elsewhere')
    write(captured.root, 'module/value.py', 'CHANGED')
    report = captured.verify_integrity()
    assert not report['valid']
    assert report['errors']


def test_recursive_workspace_pattern_stops_at_excluded_root_without_traversing_it(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    write(root, 'package.json', '{"workspaces":["**/package"]}')
    write(root, 'other/.git', 'gitdir: elsewhere')
    write(root, 'other/package/package.json', '{}')
    import agent.source.scope as module
    original = module.os.scandir

    def bounded(path):
        assert not Path(path).is_relative_to(root / 'other')
        return original(path)

    monkeypatch.setattr(module.os, 'scandir', bounded)
    with pytest.raises(ProjectScopeError, match='已排除'):
        inspect_project(root)


def test_workspace_negation_can_exclude_an_independent_project(tmp_path):
    root = tmp_path / 'project'
    write(root, 'package.json', '{"workspaces":["*", "!other"]}')
    write(root, 'app/package.json', '{}')
    write(root, 'other/.git', 'gitdir: elsewhere')
    write(root, 'other/package.json', '{"dependencies":{"never":"1"}}')
    plan = inspect_project(root)
    assert {item['path'] for item in plan.projects} == {'.', 'app'}
    assert 'other/package.json' not in plan.files
