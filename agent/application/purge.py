"""仅删除应用台账拥有的记录；用户项目、未知操作及其他会话引用受到保护。"""

from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import stat

from agent.application.store import Conflict
from agent.source.snapshots import _link
from agent.storage.activity import checked


def linked(path):
    try:
        return _link(path)
    except FileNotFoundError:
        return False


def directory(parent, name):
    target = parent / name
    if not target.exists() and not linked(target):
        return None
    if (linked(parent) or linked(target) or not target.is_dir()
            or target.resolve().parent != parent.resolve()
            or any(_link(path) for path in target.rglob('*'))):
        raise ValueError('任务记录含链接或超出保存范围，未删除记录。')
    return target


def task_paths(store, tasks, check_state):
    paths = []
    for task in tasks:
        if len(task) != 32 or any(char not in '0123456789abcdef' for char in task):
            raise ValueError('任务记录编号无效，未删除记录。')
        for area in ('tasks', 'execution'):
            target = directory(store.root / area, task)
            if target is None:
                continue
            paths.append(target)
            database = target / 'history.sqlite3'
            if not check_state or not database.exists():
                continue
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
                row = db.execute('SELECT path,sha256 FROM commits ORDER BY seq DESC LIMIT 1').fetchone()
            state = checked(target, *row)['state'] if row else {}
            if (any(state.get(key) for key in ('request_pending', 'running_step',
                    'verification_running', 'acceptance_running', 'assistant_running',
                    'web_running', 'web_request_pending'))
                    or (state.get('compaction_pending') or {}).get('stage') == 'committing'):
                raise Conflict('任务仍有进行中或结果不明的操作，不能删除。')
    return paths


def protect_projects(db, paths):
    for row in db.execute('SELECT data FROM app_conversations'):
        repo = json.loads(row[0]).get('repo')
        if repo:
            repo = Path(repo).resolve()
            if any(repo.is_relative_to(path) or path.is_relative_to(repo) for path in paths):
                raise ValueError('记录目录与工作仓库重叠，未进行删除。')


def validate_deletion(store, db, conversation, tasks):
    thread = conversation['thread_id']
    if db.execute("SELECT 1 FROM app_commands WHERE conversation=? AND state IN ('pending','running')",
                  (thread,)).fetchone():
        raise Conflict('当前仍有未处理的命令，请先等待操作结束。')
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'desktop_requests' in tables and db.execute(
            "SELECT 1 FROM desktop_requests WHERE thread=? AND status IN ('executing','launching','unknown')",
            (thread,)).fetchone():
        raise Conflict('外部操作仍在执行或结果不明，不能删除。')
    identities = {task['task_id'] for task in tasks}
    for row in db.execute('SELECT data FROM app_conversations WHERE id<>?', (thread,)):
        if json.loads(row[0]).get('work_id') in identities:
            raise Conflict('任务记录被其他对话引用，不能删除。')
    paths = task_paths(store, identities, conversation['visibility'] == 'deleted')
    paths.extend(path for area in ('application-records', 'file-operation-backups')
                 if (path := directory(store.root / area, thread)) is not None)
    protect_projects(db, paths)


def remove_directory(target):
    def readonly(function, name, error):
        path = Path(name)
        if (not isinstance(error, PermissionError) or _link(path)
                or not path.resolve().is_relative_to(target)):
            raise error
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
        function(name)

    shutil.rmtree(target, onexc=readonly)


def cleanup(store, conversation, tasks, retain_environment=False):
    from agent.storage.resources import (
        delete_owned_docker_resource, release_project, release_task_resources,
    )

    paths = task_paths(store, tasks, check_state=False)
    paths.extend(path for area in ('application-records', 'file-operation-backups')
                 if (path := directory(store.root / area, conversation)) is not None)
    with store.connection() as db:
        protect_projects(db, paths)
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        previews = []
        if 'desktop_requests' in tables:
            for row in db.execute('SELECT data FROM desktop_requests WHERE thread=?', (conversation,)):
                record = json.loads(row[0])
                if record.get('kind') != 'preview':
                    continue
                path = Path(record['record'])
                parent = store.root / 'previews'
                if (path.parent != parent or path.resolve().parent != parent.resolve()
                        or linked(path) or linked(parent)):
                    raise ValueError('预览路径超出保存范围，未进行删除。')
                previews.append(path)
    for task in tasks:
        result = release_task_resources('task:' + task, completed=True, root=store.root,
                                        docker=delete_owned_docker_resource)
        if result.get('errors'):
            raise OSError('任务资源清理未确认，记录保留供明确重试。')
    repo = store.get(conversation)['repo']
    if not retain_environment and repo:
        result = release_project(repo, root=store.root, retain=False)
        if result.get('errors'):
            raise OSError('项目环境清理未确认，记录保留供明确重试。')
    for path in previews:
        path.unlink(missing_ok=True)
    for path in paths:
        if path.exists():
            remove_directory(path)
    with store.connection() as db:
        for table in ('desktop_requests', 'desktop_files', 'permissions', 'read_sources'):
            if table in tables:
                db.execute('DELETE FROM ' + table + ' WHERE thread=?', (conversation,))
