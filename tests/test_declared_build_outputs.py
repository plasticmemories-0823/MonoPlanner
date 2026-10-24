"""已确认的输出在编译副本可变，源码和显式输入始终受保护。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.acceptance.contracts import VerificationSpec
from agent.acceptance.final import verify_final
from agent.execution.local import LocalEnvironment
from agent.source.snapshots import SourceSnapshot
from agent.task.requirements import confirmed, guard


def snapshot(tmp_path, runtime=()):
    repo = tmp_path / 'repo'
    (repo / 'output').mkdir(parents=True)
    (repo / 'main.py').write_text('print(42)\n', encoding='utf-8')
    (repo / 'output/result.json').write_text('{"duplicate":false}', encoding='utf-8')
    return SourceSnapshot.capture(repo, tmp_path / 'evidence', runtime_files=runtime)


def test_only_declared_untracked_outputs_can_change(tmp_path):
    saved = snapshot(tmp_path)
    build = saved.build_copy(tmp_path / 'build')
    (build.root / 'output/result.json').write_text('{"duplicate":true}', encoding='utf-8')
    assert not build.verify_build_integrity()['valid']
    report = build.verify_build_integrity(output_paths=('output/result.json',))
    assert report['valid'] and report['build_outputs']['modified'] == ['output/result.json']
    assert saved.verify_integrity()['valid']
    (build.root / 'main.py').write_text('print(0)\n', encoding='utf-8')
    assert not build.verify_build_integrity(output_paths=('output/result.json',))['valid']


def test_declared_output_does_not_override_explicit_input_protection(tmp_path):
    saved = snapshot(tmp_path, runtime=('output/result.json',))
    build = saved.build_copy(tmp_path / 'build')
    (build.root / 'output/result.json').write_text('{}', encoding='utf-8')
    assert not build.verify_build_integrity(output_paths=('output/result.json',))['valid']


def test_retained_test_cache_is_mutable_only_in_build_copy(tmp_path):
    saved = snapshot(tmp_path)
    # 模拟旧内核已经把缓存保存到清单的情形。
    repo = tmp_path / 'legacy'
    repo.mkdir()
    (repo / 'main.py').write_text('print(42)', encoding='utf-8')
    (repo / '.vitest-cache').mkdir()
    (repo / '.vitest-cache/results.json').write_text('1', encoding='utf-8')
    from unittest.mock import patch
    import agent.source.snapshots as snapshots
    with patch.object(snapshots, '_DERIVED_DIRS', snapshots._DERIVED_DIRS - {'.vitest-cache'}):
        old = SourceSnapshot.capture(repo, tmp_path / 'old-evidence')
    build = old.build_copy(tmp_path / 'build')
    (build.root / '.vitest-cache/results.json').write_text('2', encoding='utf-8')
    assert build.verify_build_integrity()['valid']
    assert old.verify_integrity()['valid']


@pytest.mark.parametrize('path', ['../input.json', '/tmp/output', 'C:/output', 'output\\x', ''])
def test_invalid_output_paths_are_rejected(path):
    with pytest.raises(ValueError):
        VerificationSpec(('Produce the original output',), command='run', output_paths=(path,))


def test_worker_cannot_expand_mutability_without_original_delivery_target():
    packet = confirmed('Run the original entry', [{
        'id': 'entry', 'artifact_type': 'repository_entry', 'target_runtime': 'python',
        'entry': 'main.py', 'behavior': 'Run the original entry', 'input_scope': 'Original inputs',
        'quotes': [{'source_id': 'task_request', 'text': 'Run the original entry'}],
    }], confirmation={'id': 'human', 'text': 'Run'})
    loop = SimpleNamespace(state={'requirements': packet, 'verify_command': 'run'})
    with pytest.raises(ValueError):
        guard(loop, {'requirement_digest': packet['digest'], 'execution_checks': [],
                     'output_paths': ['main.py']})


def test_final_command_can_update_confirmed_output_without_changing_evidence(tmp_path):
    saved = snapshot(tmp_path)
    env = LocalEnvironment(tmp_path / 'repo', artifact_root=tmp_path / 'runtime')
    command = 'python -c "from pathlib import Path; Path(\'output/result.json\').write_text(\'42\')"'
    spec = VerificationSpec(('Produce the original output',), command=command,
                            output_paths=('output/result.json',))
    result = verify_final(env, saved, spec, '', tmp_path / 'verify', 30)
    assert result['status'] == 'passed', result
    assert saved.verify_integrity()['valid']
    assert json.loads((saved.root / 'output/result.json').read_text()) == {'duplicate': False}
