"""项目位置交接与目录创建；只使用本地临时目录和假执行器。"""

import json
from pathlib import Path
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.application.chat import Chatbot
from agent.application.project_location import (
    ProjectLocationError, ensure_location, resolve_location,
)
from agent.application.scheduler import Scheduler
from agent.application.store import ApplicationStore, Conflict
from agent.application.work_bridge import run_work
from agent.execution.projects import EnvironmentRequired


def test_new_location_is_read_only_until_confirmed_work_starts(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    records = tmp_path / 'records'
    records.mkdir()
    location = resolve_location(str(workspace), 'pelican-bike', True)
    assert not (workspace / 'pelican-bike').exists()
    target = ensure_location(location, records)
    assert target == workspace / 'pelican-bike' and list(target.iterdir()) == []
    assert not (target / '.git').exists()
    (target / 'index.html').write_text('user work')
    assert ensure_location(location, records) == target
    assert (target / 'index.html').read_text() == 'user work'


@pytest.mark.parametrize('path', [
    '../outside', 'sub/../../outside', '/outside', 'C:/outside',
    'D:relative', '..\\outside', 'sub/.git', '.git', 'CON', 'name.', 'name ', 'a//b',
])
def test_invalid_project_location_cannot_escape_or_use_reserved_names(tmp_path, path):
    with pytest.raises((ProjectLocationError, FileNotFoundError)):
        resolve_location(str(tmp_path), path, True)
    assert list(tmp_path.iterdir()) == []


def test_existing_root_can_be_nonempty_and_contain_other_git_projects(tmp_path):
    for name in ('first', 'second'):
        directory = tmp_path / name
        directory.mkdir()
        (directory / '.git').write_text('gitdir: example')
    location = resolve_location(str(tmp_path), '.', False)
    assert location['repo'] == str(tmp_path) and location['path'] == '.'
    assert not location['create']
    with pytest.raises(ProjectLocationError, match='已经存在'):
        resolve_location(str(tmp_path), 'first', True)


def test_directory_creation_collision_or_unknown_receipt_never_replays(tmp_path):
    workspace, records = tmp_path / 'workspace', tmp_path / 'records'
    workspace.mkdir()
    records.mkdir()
    location = resolve_location(str(workspace), 'new', True)
    (workspace / 'new').mkdir()
    (workspace / 'new/keep.txt').write_text('keep')
    with pytest.raises(ProjectLocationError, match='已经出现'):
        ensure_location(location, records)
    assert (workspace / 'new/keep.txt').read_text() == 'keep'
    another = resolve_location(str(workspace), 'another', True)
    (records / 'project-create-intent.json').write_text(json.dumps({'location': another}))
    with pytest.raises(ProjectLocationError, match='结果未确认'):
        ensure_location(another, records)
    assert not (workspace / 'another').exists()


def test_linked_target_is_rejected(tmp_path):
    workspace, outside = tmp_path / 'workspace', tmp_path / 'outside'
    workspace.mkdir()
    outside.mkdir()
    try:
        (workspace / 'link').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Directory symlinks unavailable for this account')
    with pytest.raises(ProjectLocationError):
        resolve_location(str(workspace), 'link', False)
    assert not list(outside.iterdir())


def test_handoff_keeps_workspace_and_binds_only_confirmed_child(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(workspace))['thread_id']
    command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '按计划在pelican-bike开始'}
    store.submit(conversation, command)
    work = store.create_work(conversation, command['id'], project_path='pelican-bike', create=True)
    assert store.get(conversation)['repo'] == str(workspace)
    assert work['repo'] == str(workspace / 'pelican-bike')
    assert store.snapshot(conversation)['work']['repo'] == work['repo']
    assert not (workspace / 'pelican-bike').exists()
    assert store.create_work(conversation, command['id'], project_path='pelican-bike', create=True) == work


def test_workspace_selection_never_starts_preparation(tmp_path, monkeypatch):
    import agent.execution.preparation as preparation

    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create()['thread_id']
    monkeypatch.setattr(preparation, 'jobs', lambda *a, **k: pytest.fail('Selected parent prepared'))
    scheduler = Scheduler(store, model_factory=lambda _: pytest.fail('Selection invoked model'))
    try:
        selected = {'id': str(uuid.uuid4()), 'kind': 'workspace', 'repo': str(workspace)}
        scheduler.submit(conversation, selected)
        scheduler.tick()
        assert store.commands(conversation=conversation, states=('applied',))[0]['id'] == selected['id']
        assert not store.work_for_conversation(conversation)
    finally:
        scheduler.close()


def test_creation_precedes_environment_and_failed_preparation_is_not_retried(tmp_path, monkeypatch):
    import agent.application.work_bridge as bridge

    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(workspace))['thread_id']
    command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '确认开始'}
    store.submit(conversation, command)
    job = store.create_work(conversation, command['id'], project_path='new-project', create=True)
    seen = []

    def prepare(repo, **kwargs):
        assert Path(repo).is_dir() and Path(repo) == workspace / 'new-project'
        assert kwargs['project_scope'].root == Path(repo)
        assert kwargs['project_scope'].excluded_projects == ()
        seen.append(repo)
        raise EnvironmentRequired('离线注入：依赖准备未完成')

    monkeypatch.setattr(bridge, 'prepare_binding', prepare)
    factory = lambda *a, **k: pytest.fail('Failed preparation created a container')
    for _ in range(2):
        result = run_work(job, lambda *a: None, lambda: [],
                          model=object(), environment_factory=factory, root=store.root)
        assert result['status'] == 'preparation_failed'
    assert seen == [job['repo']]
    assert list((workspace / 'new-project').iterdir()) == []
    assert (store.root / 'tasks' / job['task_id'] / 'project-location.json').exists()


def test_start_work_rejects_removed_argumentless_entry(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(workspace))['thread_id']
    command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '讨论'}
    store.submit(conversation, command)

    class FakeModel:
        def __init__(self):
            self.requests = []

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return AIMessage(content='', tool_calls=[{'name': 'start_work', 'id': 'bad', 'args': {}}])
            return AIMessage(content='需要先确认项目位置。')

    model = FakeModel()
    Chatbot(store, lambda *a, **k: pytest.fail('Invalid location dispatched work'),
            factory=lambda _: model).run(conversation, command)
    result = next(m for m in model.requests[-1] if isinstance(m, ToolMessage))
    assert 'project_path' in result.content and 'actions_executed' in result.content
    assert not store.work_for_conversation(conversation)


def test_active_task_cannot_be_rebound_by_a_supplement(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'first').mkdir()
    (workspace / 'second').mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(workspace))['thread_id']
    scheduler = Scheduler(store)
    try:
        command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '开始修改first'}
        store.submit(conversation, command)
        original = scheduler.dispatch_work(conversation, command, project_path='first', create=False)
        supplement = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '第二个任务'}
        store.submit(conversation, supplement)
        with pytest.raises(Conflict, match='切换项目'):
            scheduler.dispatch_work(conversation, supplement, project_path='second', create=False)
        assert store.work(original['task_id'])['repo'] == str(workspace / 'first')
        assert store.get(conversation)['repo'] == str(workspace)
    finally:
        scheduler.close()



def test_existing_task_uses_confirmed_project_not_parent_workspace(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'project').mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(workspace))['thread_id']
    first = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '已确认项目'}
    store.submit(conversation, first)
    original = store.create_work(conversation, first['id'], project_path='project', create=False)
    store.work_state(original['task_id'], 'verified_success')
    incoming = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '运行这个项目已有的脚本'}
    store.submit(conversation, incoming)

    class FakeModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(content='', tool_calls=[{'name': 'run_existing', 'id': 'run',
                'args': {'entry': 'main.py', 'args': [], 'timeout': 10, 'usage_file': ''}}])

    scheduler = Scheduler(store, model_factory=lambda _: FakeModel())
    try:
        scheduler.chatbot.run(conversation, incoming)
        job = store.work(store.get(conversation)['work_id'])
        assert job['kind'] == 'existing' and job['repo'] == str(workspace / 'project')
        assert job['project_path'] == 'project' and job['task_id'] != original['task_id']
        assert store.get(conversation)['repo'] == str(workspace)
    finally:
        scheduler.close()
