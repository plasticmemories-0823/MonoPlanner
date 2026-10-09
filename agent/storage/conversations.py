"""Durable, idempotent input mailbox; task checkpoints still own execution."""
from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import uuid


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
