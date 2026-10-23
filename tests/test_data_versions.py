"""升级只改变有版本的应用库；未知格式和失败迁移不吞掉原件。"""

from contextlib import closing
import sqlite3

import pytest

from agent.storage.schema import initialize, DataVersionError, DATA_VERSION
from agent.application.store import ApplicationStore
from agent.storage.product import ProductDatabase


def test_current_database_has_one_explicit_version_and_no_legacy_import(tmp_path):
    store = ApplicationStore(tmp_path)
    with store.connection() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == DATA_VERSION
    with closing(sqlite3.connect(tmp_path / 'conversations.sqlite3')) as old:
        old.execute('CREATE TABLE sessions (thread TEXT, data TEXT)')
    assert store.list() == []
    ProductDatabase(tmp_path)


@pytest.mark.parametrize('version', [0, DATA_VERSION + 1])
def test_unversioned_or_future_existing_database_is_not_changed(tmp_path, version):
    path = tmp_path / 'application.sqlite3'
    with closing(sqlite3.connect(path)) as db:
        db.execute('CREATE TABLE retained (value TEXT)')
        db.execute("INSERT INTO retained VALUES ('original')")
        db.execute(f'PRAGMA user_version={version}')
        db.commit()
    before = path.read_bytes()
    with pytest.raises(DataVersionError):
        ApplicationStore(tmp_path)
    assert path.read_bytes() == before


def test_migration_backup_and_failure_preserve_original_rows(tmp_path):
    path = tmp_path / 'application.sqlite3'
    initialize(path, target=1)
    with closing(sqlite3.connect(path)) as db:
        db.execute('CREATE TABLE retained (value TEXT)')
        db.execute("INSERT INTO retained VALUES ('original')")
        db.commit()

    def fail(db):
        db.execute("UPDATE retained SET value='changed'")
        raise RuntimeError('interrupted migration')

    with pytest.raises(RuntimeError):
        initialize(path, target=2, migrations={1: fail})
    with closing(sqlite3.connect(path)) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert db.execute('SELECT value FROM retained').fetchone()[0] == 'original'
    initialize(path, target=2, migrations={1: lambda db: db.execute(
        'ALTER TABLE retained ADD COLUMN extra TEXT'
    )})
    with closing(sqlite3.connect(path)) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 2
        assert db.execute('SELECT value FROM retained').fetchone()[0] == 'original'
    assert len(list((tmp_path / 'data-backups').glob('*.sqlite3'))) == 2


def test_migration_cannot_modify_a_live_application_database(tmp_path):
    from agent.application.lease import ServiceLease

    store = ApplicationStore(tmp_path)
    lease = ServiceLease(tmp_path)
    lease.acquire()
    try:
        with pytest.raises(RuntimeError, match='MonoPlanner'):
            initialize(store.path, target=DATA_VERSION + 1, migrations={DATA_VERSION: lambda db: None})
        with store.connection() as db:
            assert db.execute('PRAGMA user_version').fetchone()[0] == DATA_VERSION
        assert not (tmp_path / 'data-backups').exists()
    finally:
        lease.close()
