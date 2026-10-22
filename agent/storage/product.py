"""产品安全数据入口；与新会话共用数据库，不保存第二份会话状态。"""

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import uuid


class ProductDatabase:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'application.sqlite3'
        from agent.storage.schema import initialize

        initialize(self.path)
        if self.path.is_symlink():
            raise ValueError('Product database cannot be a link')

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=15)) as db:
            from agent.storage.schema import require_version

            require_version(db)
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA synchronous=FULL')
            db.execute('PRAGMA busy_timeout=15000')
            with db:
                yield db

    @staticmethod
    def thread(value):
        return str(uuid.UUID(str(value)))

    @staticmethod
    def table(db, name):
        return db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    def session(self, thread, db=None):
        """读取当前会话及工作状态，禁止为了旧接口复制或写回另一份状态。"""
        if db is None:
            with self.connection() as connection:
                return self.session(thread, connection)
        if not self.table(db, 'app_conversations'):
            return None
        row = db.execute(
            'SELECT data FROM app_conversations WHERE id=?', (self.thread(thread),)
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row['data'])
        return value

    def require_available(self, db, thread):
        value = self.session(thread, db)
        if value and value['visibility'] in {'purging', 'purged'}:
            raise ValueError('Conversation is being deleted')


