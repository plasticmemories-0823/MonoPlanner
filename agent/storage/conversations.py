"""Durable, idempotent input mailbox; task checkpoints still own execution."""
from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import uuid

from agent.storage.activity import checked


class ConversationStore:
    def __init__(self, root):
        self.path = Path(root)/'conversations.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    thread TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS inputs (
                    seq INTEGER PRIMARY KEY, thread TEXT NOT NULL, id TEXT NOT NULL,
                    kind TEXT NOT NULL, text TEXT NOT NULL, applied INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(thread, id));
                CREATE TABLE IF NOT EXISTS presentation (
                    thread TEXT PRIMARY KEY, visibility TEXT NOT NULL DEFAULT 'active');
            ''')

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            with db:
                db.row_factory = sqlite3.Row
                db.execute('PRAGMA synchronous=FULL')
                yield db

    @staticmethod
    def thread(value):
        return str(uuid.UUID(str(value)))

    def session(self, thread):
        with self.connection() as db:
            row = db.execute('SELECT data FROM sessions WHERE thread=?', (self.thread(thread),)).fetchone()
            return json.loads(row['data']) if row else None

    def error_is_settled(self, task):
        """Read the failed checkpoint without restoring or changing execution."""
        if not isinstance(task, str) or len(task) != 32 or any(c not in '0123456789abcdef' for c in task):
            return False
        root = self.path.parent/'tasks'/task
        try:
            with closing(sqlite3.connect((root/'history.sqlite3').as_uri()+'?mode=ro', uri=True)) as db:
                row = db.execute('SELECT path,sha256 FROM commits ORDER BY seq DESC LIMIT 1').fetchone()
            state = checked(root, *row)['state'] if row else {}
            unknown = ('request_pending', 'running_step', 'verification_running', 'acceptance_running',
                       'assistant_running', 'web_running', 'web_request_pending')
            return (state.get('status') == 'runtime_error'
                    and not any(state.get(key) for key in unknown)
                    and not state.get('budget', {}).get('unknown_usage_steps', 0)
                    and (state.get('compaction_pending') or {}).get('stage') != 'committing')
        except (OSError, ValueError, sqlite3.Error, KeyError):
            return False

    def save_session(self, thread, value):
        with self.connection() as db:
            db.execute('INSERT INTO sessions VALUES (?,?) ON CONFLICT(thread) DO UPDATE SET data=excluded.data',
                       (self.thread(thread), json.dumps(value, ensure_ascii=False)))

    def enqueue(self, thread, identity, kind, text=''):
        thread = self.thread(thread)
        if kind not in {'message', 'pause', 'resume'} or not isinstance(identity, str) or not identity:
            raise ValueError('Invalid control kind or message ID')
        if not isinstance(text, str) or (kind == 'message' and not text.strip()):
            raise ValueError('A message requires non-empty text')
        with self.connection() as db:
            old = db.execute('SELECT * FROM inputs WHERE thread=? AND id=?', (thread, identity)).fetchone()
            if old and (old['kind'], old['text']) != (kind, text):
                raise ValueError('Message ID was already used for different content')
            db.execute('INSERT OR IGNORE INTO inputs(thread,id,kind,text) VALUES (?,?,?,?)',
                       (thread, identity, kind, text))
            return dict(db.execute('SELECT * FROM inputs WHERE thread=? AND id=?', (thread, identity)).fetchone())

    def pending(self, thread):
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                'SELECT * FROM inputs WHERE thread=? AND applied=0 ORDER BY seq', (self.thread(thread),))]

    def acknowledge(self, thread, identities):
        with self.connection() as db:
            db.executemany('UPDATE inputs SET applied=1 WHERE thread=? AND id=?',
                           [(self.thread(thread), identity) for identity in identities])

    def visibility(self):
        with self.connection() as db:
            return {r['thread']: r['visibility'] for r in db.execute('SELECT * FROM presentation')}

    def set_visibility(self, thread, visibility):
        thread = self.thread(thread)
        if visibility not in {'active', 'archived', 'deleted'}:
            raise ValueError('visibility must be active, archived or deleted')
        with self.connection() as db:
            row = db.execute('SELECT data FROM sessions WHERE thread=?', (thread,)).fetchone()
            session = json.loads(row['data']) if row else {}
            pending = db.execute('SELECT 1 FROM inputs WHERE thread=? AND applied=0 LIMIT 1', (thread,)).fetchone()
            if visibility != 'active' and (session.get('status') in {'needs_plan', 'executing'} or pending):
                raise ValueError('请先暂停任务，并等待当前请求或操作结束。')
            db.execute('INSERT INTO presentation VALUES (?,?) ON CONFLICT(thread) DO UPDATE SET visibility=excluded.visibility',
                       (thread, visibility))
        return {'thread_id': thread, 'visibility': visibility}
