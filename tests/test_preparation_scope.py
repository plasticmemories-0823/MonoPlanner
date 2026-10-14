"""安装准备沿用任务冻结范围，边界变化必须先于材料读取和安装被拒绝。"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from agent.execution.managed import ManagedEnvironments
from agent.execution.projects import prepare_binding
from agent.execution.preparation import ensure_prepared
from agent.source.scope import ProjectScope, ProjectScopeError


def project(tmp_path):
    root = tmp_path / 'project'
    excluded = root / 'backend'
    excluded.mkdir(parents=True)
    marker = excluded / '.git'
    marker.write_text('gitdir: elsewhere')
    (excluded / 'requirements.txt').write_text('MUST_NOT_INSTALL')
    return root, marker, ProjectScope.discover(root)


def test_removed_git_marker_cannot_expand_installation_before_task_binding(tmp_path, monkeypatch):
    root, marker, scope = project(tmp_path)
    marker.unlink()
    monkeypatch.setattr('agent.execution.managed.inspect_project',
                        lambda *a, **k: pytest.fail('Invalid scope read project materials'))
    monkeypatch.setattr(ManagedEnvironments, '_run',
                        lambda *a, **k: pytest.fail('Invalid scope invoked Docker'))
    with pytest.raises(ProjectScopeError, match='标记'):
        ManagedEnvironments(tmp_path / 'state').prepare(root, project_scope=scope)
    assert (root / 'backend/requirements.txt').read_text() == 'MUST_NOT_INSTALL'


def test_waiting_for_prepare_lock_does_not_recompute_a_wider_scope(tmp_path, monkeypatch):
    root, marker, scope = project(tmp_path)

    @contextmanager
    def lock_changed(*args, **kwargs):
        marker.unlink()
        yield

    monkeypatch.setattr('agent.execution.managed._lock', lock_changed)
    monkeypatch.setattr('agent.execution.managed.inspect_project',
                        lambda *a, **k: pytest.fail('Lock wait widened frozen material scope'))
    monkeypatch.setattr(ManagedEnvironments, '_run',
                        lambda *a, **k: pytest.fail('Changed scope invoked Docker'))
    with pytest.raises(ProjectScopeError, match='标记'):
        ManagedEnvironments(tmp_path / 'state').prepare(root, project_scope=scope)


def test_binding_and_environment_queue_receive_the_same_frozen_scope(tmp_path, monkeypatch):
    root, _, scope = project(tmp_path)
    observed = []
    binding = {'backend': 'docker', 'image': 'sha256:' + 'a' * 64,
               'runtime': {'python': '/usr/local/bin/python', 'conda_env': ''}}

    def wait(repo, **kwargs):
        observed.append((repo, kwargs['project_scope']))
        return binding

    monkeypatch.setattr('agent.execution.projects.workspace_binding',
                        lambda repo: {'backend': 'docker'})
    monkeypatch.setattr('agent.execution.preparation.jobs',
                        lambda root=None: SimpleNamespace(wait=wait))
    monkeypatch.setattr('agent.execution.preparation.ensure_prepared', ensure_prepared)
    assert prepare_binding(str(root), project_scope=scope) == binding
    assert observed == [(str(root), scope)]
