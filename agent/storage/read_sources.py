"""用户选择的只读文件与目录，供查询和预览共用。"""

import json
from agent.storage.product import ProductDatabase
from agent.execution.assistant_files import authorized_sources


class ReadSources:
    def __init__(self, root):
        self.store = ProductDatabase(root)
        with self.store.connection() as db:
            db.execute(
                'CREATE TABLE IF NOT EXISTS read_sources (thread TEXT PRIMARY KEY, data TEXT NOT NULL)'
            )

    def get(self, thread, roots=()):
        thread = self.store.thread(thread)
        with self.store.connection() as db:
            row = db.execute('SELECT data FROM read_sources WHERE thread=?', (thread,)).fetchone()
            if row:
                return json.loads(row['data'])
            values = [{'path': path, 'kind': 'directory'} for path in roots]
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='desktop_files'").fetchone():
                values.extend(
                    {'path': r['path'], 'kind': 'file'}
                    for r in db.execute('SELECT path FROM desktop_files WHERE thread=?', (thread,))
                )
        return authorized_sources(values)

    def set(self, thread, values):
        values = authorized_sources(values)
        thread = self.store.thread(thread)
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self.store.require_available(db, thread)
            session = self.store.session(thread, db) or {}
            if session.get('status') in {'queued', 'running', 'needs_plan', 'executing', 'pausing'}:
                raise ValueError('请先暂停当前任务，再更改可读资料')
            db.execute(
                'INSERT INTO read_sources VALUES (?,?) ON CONFLICT(thread) DO UPDATE SET data=excluded.data',
                (thread, json.dumps(values, ensure_ascii=False)),
            )
        return values

    def add(self, thread, path, kind):
        return self.set(thread, [*self.get(thread), {'path': str(path), 'kind': kind}])
