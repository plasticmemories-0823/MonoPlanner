"""Remove one discarded conversation; never traverse workspace or evaluation paths."""
import json
import shutil
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

from agent.source.snapshots import _link
from agent.storage.activity import checked


def task_ids(session):
    identities = {session['task_id']} if session.get('task_id') else set()
    identities.update(item['id'] for item in session.get('activity_tasks', []))
    if any(not isinstance(value, str) or len(value) != 32
           or any(c not in '0123456789abcdef' for c in value) for value in identities):
        raise ValueError('任务记录编号无效，未删除任何任务目录。')
    return identities


def archive_paths(store, identities, *, check_state=True):
    task_ids({'activity_tasks': [{'id': identity} for identity in identities]})
    root = store.path.parent.resolve()
    paths = []
    for area in ('tasks', 'execution'):
        base = root/area
        if base.exists() and _link(base):
            raise ValueError('任务目录不能是链接，未进行删除。')
        for identity in sorted(identities):
            path = base/identity
            if not path.exists() and not path.is_symlink():
                continue
            if _link(path) or not path.is_dir() or path.resolve().parent != base:
                raise ValueError('任务路径超出保存范围，未进行删除。')
            paths.append(path)
    for path in paths:
        # Reject reparse points before recursive removal, including nested ones.
        if any(_link(item) for item in path.rglob('*')):
            raise ValueError('任务记录含有链接，未进行删除。')
        database = path/'history.sqlite3'
        if check_state and database.exists():
            with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as db:
                row = db.execute('SELECT path,sha256 FROM commits ORDER BY seq DESC LIMIT 1').fetchone()
            state = checked(path, *row)['state'] if row else {}
            unknown = ('request_pending', 'running_step', 'verification_running', 'acceptance_running',
                       'assistant_running', 'web_running', 'web_request_pending')
            if (any(state.get(key) for key in unknown)
                    or (state.get('compaction_pending') or {}).get('stage') == 'committing'):
                raise ValueError('任务仍有进行中或结果不明的操作，请先核对运行记录。')
    return paths


def protect_workspaces(db, paths):
    for row in db.execute('SELECT data FROM sessions'):
        repo = json.loads(row['data']).get('repo')
        if repo:
            repo = Path(repo).resolve()
            if any(repo.is_relative_to(path) or path.is_relative_to(repo) for path in paths):
                raise ValueError('记录目录与工作仓库重叠，未进行删除。')


def begin_purge(store, thread):
    thread = store.thread(thread)
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT visibility FROM presentation WHERE thread=?', (thread,)).fetchone()
        if not row or row['visibility'] not in {'deleted', 'purging', 'purged'}:
            raise ValueError('请先将对话移入回收站，再彻底删除。')
        visibility = row['visibility']
        row = db.execute('SELECT data FROM sessions WHERE thread=?', (thread,)).fetchone()
        session = json.loads(row['data']) if row else {}
        pending = db.execute('SELECT 1 FROM inputs WHERE thread=? AND applied=0 LIMIT 1', (thread,)).fetchone()
        if session.get('status') in {'needs_plan', 'executing'} or pending:
            raise ValueError('请先暂停任务，并等待当前请求或操作结束。')
        identities = task_ids(session)
        for other in db.execute('SELECT data FROM sessions WHERE thread<>?', (thread,)):
            if identities & task_ids(json.loads(other['data'])):
                raise ValueError('任务记录被其他对话引用，未进行删除。')
        protect_workspaces(db, archive_paths(store, identities, check_state=visibility == 'deleted'))
        if visibility != 'purged':
            db.execute('UPDATE presentation SET visibility=? WHERE thread=?', ('purging', thread))
    return identities


def finish_purge(store, thread, identities):
    with store.connection() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT visibility FROM presentation WHERE thread=?', (thread,)).fetchone()
        if not row or row['visibility'] not in {'purging', 'purged'}:
            raise ValueError('删除状态已改变，未继续清理。')
        # Deletion already owns these records. A partially removed checkpoint
        # is expected on retry; path/link/workspace checks still always run.
        paths = archive_paths(store, identities, check_state=False)
        protect_workspaces(db, paths)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'desktop_requests' in tables:
            preview_root = store.path.parent/'previews'
            for row in db.execute('SELECT data FROM desktop_requests WHERE thread=?', (thread,)):
                item = json.loads(row[0])
                if item.get('kind') == 'preview':
                    path = Path(item['record'])
                    if path.parent != preview_root or _link(path) or _link(preview_root):
                        raise ValueError('预览路径超出保存范围，未进行删除。')
                    path.unlink(missing_ok=True)
        for table in ('desktop_requests', 'desktop_files', 'permissions'):
            if table in tables:
                db.execute('DELETE FROM '+table+' WHERE thread=?', (thread,))
        for path in paths:
            def remove_readonly(function, name, error):
                target = Path(name)
                if not isinstance(error, PermissionError) or _link(target) or not target.resolve().is_relative_to(path):
                    raise error
                target.chmod(target.stat().st_mode | stat.S_IWRITE)
                function(name)
            shutil.rmtree(path, onexc=remove_readonly)
        db.execute('DELETE FROM inputs WHERE thread=?', (thread,))
        db.execute('DELETE FROM sessions WHERE thread=?', (thread,))
        # Keep only the UUID tombstone so stale tabs cannot recreate the chat.
        db.execute('UPDATE presentation SET visibility=? WHERE thread=?', ('purged', thread))
    return {'thread_id': thread, 'deleted': True, 'task_records_deleted': len(identities)}
