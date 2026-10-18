"""可选预热不能吞掉落盘失败，也不能重叠复制同一份源码。"""

from concurrent.futures import ThreadPoolExecutor, Future
import errno
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.source.probe_preparation import ProbePreparation
from agent.source.preparation import Preparation
from agent.source import lifecycle


@pytest.mark.parametrize('action', ['finish', 'take', 'start', 'same_version_start'])
def test_failed_background_copy_is_not_discarded_or_rebuilt(tmp_path, action):
    future = Future()
    future.set_exception(OSError(errno.ENOSPC, 'fixture disk full'))
    pool = Mock()
    worker = ProbePreparation(pool, tmp_path)
    cancel = Event()
    if action == 'start':
        cancel.set()
    worker.active = {'version': 'v1', 'future': future, 'cancel': cancel}
    with pytest.raises(OSError):
        if action == 'take':
            worker.take('v1')
        elif action in {'start', 'same_version_start'}:
            worker.start(SimpleNamespace(version='v1'))
        else:
            worker.finish()
    pool.submit.assert_not_called()


def test_running_copy_is_consumed_once_instead_of_falling_back(tmp_path, monkeypatch):
    started, release = Event(), Event()
    snapshot = SimpleNamespace(root=tmp_path / 'copy')
    calls = []

    def copy(view, directory, cancel):
        calls.append(directory)
        started.set()
        assert release.wait(2)
        assert not cancel.is_set()
        return {'snapshot': snapshot, 'version': 'v1'}

    monkeypatch.setattr('agent.source.probe_preparation.prepare_copy', copy)
    monkeypatch.setattr('agent.source.probe_preparation.validate_copy', lambda _: None)
    with ThreadPoolExecutor(max_workers=1) as pool, ThreadPoolExecutor(max_workers=1) as caller:
        worker = ProbePreparation(pool, tmp_path)
        worker.start(SimpleNamespace(version='v1'))
        assert started.wait(1)
        taken = caller.submit(worker.take, 'v1')
        try:
            assert not taken.done()
        finally:
            release.set()
        assert taken.result(timeout=2) is snapshot
    assert len(calls) == 1


def test_index_storage_failure_propagates_instead_of_becoming_optional_miss(tmp_path):
    future = Future()
    future.set_exception(OSError(errno.ENOSPC, 'fixture disk full'))
    prep = Preparation(tmp_path)
    prep.active = {'version': 'v1', 'future': future}
    try:
        with pytest.raises(OSError):
            prep.collect('v1')
    finally:
        prep.pool.shutdown()


@pytest.mark.parametrize('operation', ['collect', 'close'])
def test_outer_source_lifecycle_does_not_swallow_storage_failure(operation):
    prep = Mock(closed=False)
    getattr(prep, operation).side_effect = OSError(errno.ENOSPC, 'fixture disk full')
    loop = SimpleNamespace(preparation=prep, evidence=SimpleNamespace(current='v1'),
                           history_store=None, executor=SimpleNamespace(prepared=None),
                           preparation_errors=[])
    with pytest.raises(OSError):
        getattr(lifecycle, operation)(loop)


def test_probe_session_evidence_write_failure_is_infrastructure_error(tmp_path, monkeypatch):
    from agent.execution.probe_batch import ProbeBatch
    from agent.source.errors import SourceStorageError

    batch = ProbeBatch(None, tmp_path)
    batch.folder = tmp_path
    monkeypatch.setattr(Path, 'write_text', Mock(side_effect=OSError(errno.ENOSPC, 'fixture')))
    with pytest.raises(SourceStorageError):
        batch._save_stats()


def test_source_lookup_evidence_write_failure_is_infrastructure_error(tmp_path, monkeypatch):
    from agent.source.errors import SourceStorageError

    observe = Mock(return_value={'source_version': 'v1', 'files': {}})
    loop = SimpleNamespace(executor=SimpleNamespace(exploration=SimpleNamespace(
        observe=observe, coverage=lambda: {})),
                           state={'source_map': {}}, root=tmp_path, history_store=None)
    monkeypatch.setattr('agent.storage.records.save', Mock(
        side_effect=OSError(errno.ENOSPC, 'fixture')))
    with pytest.raises(SourceStorageError):
        lifecycle.record_lookup(loop, SimpleNamespace(op='read'),
                                {'status': 'succeeded', 'data': {}}, tmp_path)
