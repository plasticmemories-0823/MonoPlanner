"""应用数据的版本边界；迁移只改应用表，不转换任务原生检查点。"""

from contextlib import closing
from pathlib import Path
import sqlite3
import uuid


DATA_VERSION = 1
MIGRATIONS = {}


class DataVersionError(ValueError):
    """不认识的数据必须停止写入，不能猜测格式或自动清空。"""


def version(db):
    return db.execute('PRAGMA user_version').fetchone()[0]


def require_version(db, expected=DATA_VERSION):
    if version(db) != expected:
        raise DataVersionError('Application data version is unsupported; no data was changed')


def initialize(path, *, target=DATA_VERSION, migrations=None):
    """空库建立基准；未来连续迁移先备份，失败回滚，不迁移发布前无版本数据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise DataVersionError('Application database cannot be a link')
    steps = MIGRATIONS if migrations is None else migrations
    with closing(sqlite3.connect(path, timeout=15)) as db:
        current = version(db)
        if current == target:
            return
        if current > target or current < 0:
            raise DataVersionError('This data needs a newer application; no data was changed')
        if current == 0:
            tables = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if tables:
                raise DataVersionError(
                    'Pre-baseline application data is unsupported; use an explicitly reset data directory'
                )
            with db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('PRAGMA user_version=1')
            current = 1
        if current == target:
            return
        if any(number not in steps for number in range(current, target)):
            raise DataVersionError('No complete migration path; no existing data was changed')
        from agent.application.lease import ServiceLease

        lease = ServiceLease(path.parent)
        lease.acquire()
        try:
            _migrate(db, path, current, target, steps)
        finally:
            lease.close()


def _migrate(db, path, current, target, steps):
    # 备份不包含明文凭据；SQLite backup 保留 WAL 中已提交的数据。
    folder = path.parent / 'data-backups'
    folder.mkdir(exist_ok=True)
    backup = folder / f'{path.stem}-v{current}-{uuid.uuid4().hex}.sqlite3'
    with closing(sqlite3.connect(backup)) as saved:
        db.backup(saved)
    try:
        db.execute('BEGIN EXCLUSIVE')
        if version(db) != current:
            raise DataVersionError('Data version changed concurrently; retry after inspection')
        for number in range(current, target):
            steps[number](db)
            if not db.in_transaction:
                raise DataVersionError('A migration must not commit its own transaction')
            db.execute(f'PRAGMA user_version={number + 1}')
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise DataVersionError('Migrated application data failed integrity validation')
        db.commit()
    except Exception:
        db.rollback()
        # 即使开发中的迁移错误地提交，也恢复先前完整库；保留备份供审计。
        with closing(sqlite3.connect(backup.as_uri() + '?mode=ro', uri=True)) as saved:
            saved.backup(db)
        raise
