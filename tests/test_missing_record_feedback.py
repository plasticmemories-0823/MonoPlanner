"""不存在的猜测路径是查询失败；已登记原件丢失仍属于完整性故障。"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.execution.executor import OperationExecutor


def reader(root, registered):
    value = OperationExecutor(SimpleNamespace(repo_root=root), None, root)
    value.archive = SimpleNamespace(find=lambda *a, **kw: (
        [{'path': 'acceptance/trial/review.json'}] if registered else []))
    return value


def test_unknown_record_path_is_a_correctable_read_error(tmp_path):
    with pytest.raises(ValueError, match='not found.*not registered'):
        reader(tmp_path, False).read('$record/acceptance/trial/review.json', 0, 20)


def test_missing_registered_original_still_stops(tmp_path):
    with pytest.raises(FileNotFoundError):
        reader(tmp_path, True).read('$record/acceptance/trial/review.json', 0, 20)


def test_unknown_record_does_not_suppress_permission_failure(tmp_path, monkeypatch):
    def denied(*args):
        raise PermissionError('cannot access evidence storage')
    monkeypatch.setattr('agent.execution.executor._safe_file', denied)
    with pytest.raises(PermissionError):
        reader(tmp_path, False).read('$record/acceptance/trial/review.json', 0, 20)
