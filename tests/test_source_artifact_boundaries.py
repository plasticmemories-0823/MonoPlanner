"""程序归档不再递归成为源码；未知同名文件必须保留核对机会。"""

import errno
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import zipfile

import pytest

from agent.execution.installation_materials import collect
from agent.source.snapshots import SourceSnapshot, SnapshotError, _save


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def source_archive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('web/package.json', json.dumps({'name': 'monoplanner-web'}))
        archive.writestr('agent/task/runner.py', 'class WorkRunner: pass')
        archive.writestr('agent/storage/history.py', 'task-history-reference')
        archive.writestr('resources/prompts/task.txt', 'submit_plan freeze_acceptance')
        archive.writestr('agent/version.py', "VERSION = '1.0.0'")


def plan(repo):
    return SimpleNamespace(root=repo, files={}, source_projects=[], local_dependencies=[],
                           projects=[{'path': '.', 'has_package': True}])


def test_snapshot_and_installation_exclude_verified_records_and_orphan_product_archives(tmp_path):
    repo = tmp_path / 'project'
    repo.mkdir()
    (repo / 'module.py').write_text('VALUE = 1')
    record = repo / 'renamed-task'
    write_json(record / 'state.json', {'format': 'task-history-reference',
                                      'database': 'history.sqlite3'})
    (record / 'history.sqlite3').write_bytes(b'SQLite format 3\0')
    source_archive(record / 'before-source.zip')
    source_archive(repo / 'orphan/evaluated-source.zip')
    partial = repo / 'interrupted'
    write_json(partial / 'manifest.json', {
        'source_root': 'C:/project', 'root': 'C:/saved/source', 'status': 'failed',
        'files': {}, 'excluded_files': {}, 'excluded': [], 'notes': []})
    (partial / 'source').mkdir()
    (partial / 'source/copied.bin').write_bytes(b'old content')
    portable = repo / 'orphan/MonoPlanner-1.0.0-Windows11-x64.zip'
    with zipfile.ZipFile(portable, 'w') as archive:
        archive.writestr('MonoPlanner/portable.json', json.dumps({
            'product': 'MonoPlanner', 'version': '1.0.0', 'source_fingerprint': 'a' * 64}))
        archive.writestr('MonoPlanner/MonoPlanner.exe', b'MZ')
    snapshot = SourceSnapshot.capture(repo, tmp_path / 'capture')
    assert set(snapshot.files) == {'module.py'}
    assert set(collect(plan(repo))) == {'module.py'}
    assert not snapshot.excluded_files


def test_ordinary_same_named_record_files_and_unrelated_zip_remain_source(tmp_path):
    repo = tmp_path / 'project'
    repo.mkdir()
    write_json(repo / 'fixture/state.json', {})
    (repo / 'fixture/history.sqlite3').write_bytes(b'user fixture')
    with zipfile.ZipFile(repo / 'dataset.zip', 'w') as archive:
        archive.writestr('samples.csv', 'a,b\n1,2')
    expected = {'fixture/state.json', 'fixture/history.sqlite3', 'dataset.zip'}
    assert set(SourceSnapshot.capture(repo, tmp_path / 'capture').files) == expected
    assert set(collect(plan(repo))) == expected


@pytest.mark.parametrize('filename', ['before-source.zip', 'evaluated-source.zip',
                                      'MonoPlanner-1.0.0-Windows11-x64.zip'])
def test_unconfirmed_product_archive_stops_before_copy_without_discarding_it(tmp_path, filename):
    repo = tmp_path / 'project'
    repo.mkdir()
    with zipfile.ZipFile(repo / filename, 'w') as archive:
        archive.writestr('user-source.py', 'print(1)')
    original = (repo / filename).read_bytes()
    with pytest.raises((SnapshotError, ValueError), match='归档|archive'):
        SourceSnapshot.capture(repo, tmp_path / 'capture')
    assert not list((tmp_path / 'capture/source').iterdir())
    with pytest.raises(ValueError, match='归档|archive'):
        collect(plan(repo))
    assert (repo / filename).read_bytes() == original


def test_tracked_archive_remains_authorized_source_in_both_paths(tmp_path):
    repo = tmp_path / 'project'
    repo.mkdir()
    source_archive(repo / 'before-source.zip')
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
    assert set(SourceSnapshot.capture(repo, tmp_path / 'capture').files) == {'before-source.zip'}
    assert set(collect(plan(repo))) == {'before-source.zip'}


def test_explicit_local_dependency_cannot_bypass_product_record_boundary(tmp_path):
    repo = tmp_path / 'project'
    repo.mkdir()
    source_archive(repo / 'before-source.zip')
    selected = plan(repo)
    selected.projects = []
    selected.local_dependencies = ['before-source.zip']
    with pytest.raises(ValueError, match='归档'):
        collect(selected)


def test_manifest_failure_does_not_truncate_previous_capturing_record(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    initial = {'status': 'capturing', 'files': {}}
    _save(path, initial)
    original = path.read_bytes()
    real_open = Path.open

    class FullDisk:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, content):
            self.stream.write(content[:9])
            self.stream.flush()
            raise OSError(errno.ENOSPC, 'fixture disk full')

    def open_full(target, mode='r', *args, **kwargs):
        stream = real_open(target, mode, *args, **kwargs)
        return FullDisk(stream) if 'w' in mode or 'x' in mode else stream

    monkeypatch.setattr(Path, 'open', open_full)
    with pytest.raises(OSError):
        _save(path, {'status': 'failed', 'error': 'fixture'})
    assert path.read_bytes() == original
    assert json.loads(path.read_text()) == initial


def test_excluded_product_zip_is_never_read_as_source_content(tmp_path, monkeypatch):
    import agent.source.snapshots as snapshots

    repo = tmp_path / 'project'
    repo.mkdir()
    source_archive(repo / 'before-source.zip')
    inspect = snapshots._inspect_file
    reads = []

    def checked(path, *args):
        reads.append(path)
        assert path.suffix != '.zip'
        return inspect(path, *args)

    monkeypatch.setattr(snapshots, '_inspect_file', checked)
    assert not SourceSnapshot.capture(repo, tmp_path / 'capture').files
    assert not reads


def test_hash_checks_cancellation_between_chunks(tmp_path, monkeypatch):
    from threading import Event
    from agent.source.snapshots import _inspect_file

    path = tmp_path / 'source.bin'
    path.write_bytes(b'x' * (3 * 1024 * 1024))
    cancel = Event()
    real_open = Path.open
    count = []

    class CancellableRead:
        def __enter__(self):
            self.stream = real_open(path, 'rb')
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            data = self.stream.read(size)
            count.append(len(data))
            cancel.set()
            return data

    monkeypatch.setattr(Path, 'open', lambda *_args, **_kwargs: CancellableRead())
    with pytest.raises(InterruptedError):
        _inspect_file(path, cancel)
    assert count == [1024 * 1024]


def test_legacy_export_manifest_and_large_snapshot_manifest_are_not_copied(tmp_path):
    from agent.source.artifacts import generated_records, ArtifactScopeError

    old = tmp_path / 'old'
    (old / 'source').mkdir(parents=True)
    write_json(old / 'manifest.json', {
        'files': {}, 'excluded_files': {}, 'root': str(old / 'source'), 'status': 'captured',
        'source_version': 'a' * 64,
        'origin': 'SourceVersions archived bytes; no live workspace read'})
    assert generated_records(old)
    large = tmp_path / 'large'
    (large / 'source').mkdir(parents=True)
    write_json(large / 'manifest.json', {'source_root': 'C:/repo', 'root': 'C:/copy/source',
                                       'status': 'failed', 'files': {'long': 'x' * (3 * 1024**2)}})
    with pytest.raises(ArtifactScopeError):
        generated_records(large)
