"""复核同一次测试里的调用/读取证据，不通过重跑补齐缺失回执。"""
import hashlib
import json
from pathlib import Path

from agent.source.snapshots import _safe_file


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_row(output, value, check, row, repository, map_path):
    phase = output.name.split('.')[0]
    test_path = output.with_name(phase + '.tests.json')
    if file_hash(test_path) != value.get('test_report_sha256'):
        raise ValueError('Consumption receipt differs from its original test report')
    source = output.parent / 'test_acceptance.py'
    if file_hash(source) != value.get('test_source_sha256'):
        raise ValueError('Consumption receipt differs from its frozen tests')
    raw = json.loads(test_path.read_text('utf-8'))
    originals = raw.get('consumption', {}).get('checks', [])
    matching = [r for r in originals if r.get('id') == check['id']]
    if matching != [row] or row.get('test_ids') != check['test_ids']:
        raise ValueError('Consumption row is not the original bound-test observation')
    calls = row.get('calls')
    if (not isinstance(calls, dict) or set(calls) != set(check['test_ids'])
            or any(type(n) is not int or n < 0 for n in calls.values())):
        raise ValueError('Consumption call counters are incomplete')
    rows = raw.get('test_results', [])
    tests = {'.'.join(r['id'].split('.')[-2:]): r for r in rows}
    if len(tests) != len(rows):
        raise ValueError('Unittest consumption identities are ambiguous')
    if check['artifact_type'] == 'python_api':
        if row.get('api_entry') != check['api_entry']:
            raise ValueError('A different library interface supplied the receipt')
    elif row.get('artifact_path') != check['entry_file']:
        raise ValueError('A different delivered file supplied the receipt')
    if row['status'] == 'error':
        return
    if any(name not in tests or tests[name]['status'] not in {'passed', 'failed', 'error'}
           or calls[name] < 1 for name in check['test_ids']):
        raise ValueError('A required test did not actually consume the delivery')
    if repository is None:
        raise ValueError('Consumption evidence requires the actual saved repository')
    repository = Path(repository)
    if check['artifact_type'] == 'python_api':
        module, _ = check['api_entry'].split(':')
        path = Path(*module.split('.'))
        relative = (path / '__init__.py' if (repository / path / '__init__.py').is_file()
                    else path.with_suffix('.py'))
        actual = _safe_file(repository, relative)
        if (row['status'] != 'passed' or row.get('syntax_checked') is not True
                or row.get('compiled_definition_checked') is not True
                or row.get('source') != str(map_path(actual))
                or file_hash(actual) != row.get('source_sha256')):
            raise ValueError('Consumed API source does not match the original repository')
    else:
        artifact_root = repository
        snapshot_hash = value.get('artifact_snapshot_sha256')
        if snapshot_hash is not None:
            manifest = output.with_name(phase + '.consumed-files.json')
            if file_hash(manifest) != snapshot_hash:
                raise ValueError('Consumed artifact evidence inventory changed')
            snapshot = json.loads(manifest.read_text('utf-8'))
            if (snapshot.get('format') != 1 or check['entry_file'] not in snapshot['files']
                    or snapshot['files'][check['entry_file']] != row.get('artifact_sha256')):
                raise ValueError('Consumed artifact evidence differs from the original read')
            artifact_root = output.parent / (phase + '.consumed-files')
        actual = artifact_root / check['entry_file']
        if row['status'] == 'failed':
            if (row.get('missing_at_start') is not True or actual.exists()
                    or row.get('artifact_sha256') is not None
                    or any(tests[name]['status'] == 'passed' for name in check['test_ids'])):
                raise ValueError('The missing-artifact failure lacks its original evidence')
            return
        actual = _safe_file(artifact_root, Path(check['entry_file']))
        if (row.get('missing_at_start') is not False
                or file_hash(actual) != row.get('artifact_sha256')
                or type(row.get('bytes_read')) is not int
                or row['bytes_read'] != actual.stat().st_size * sum(calls.values())):
            raise ValueError('The original delivered bytes were not read completely and unchanged')
