"""Wine 兼容边界显式绑定；不伪造编译，不替换入口，不放宽原有验收。"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from agent.acceptance.execution import checked_execution, validate_execution_result
from agent.acceptance.capabilities import describe, require_targets
from resources.drivers.execution import WINE_SCOPE, wine_command


def declaration(**changes):
    return checked_execution([{
        'id': 'entry', 'artifact_type': 'repository_entry', 'consumer': 'command',
        'entry_file': 'clean.cmd', 'interpreter': 'wine-cmd', 'argv': ['--preview'],
        **changes,
    }], ('Run the requested batch artifact in Wine compatibility mode',))[0]


def evidence(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'clean.cmd').write_bytes(b'@echo off\r\necho preview\r\nexit /b 0\r\n')
    folder = tmp_path / 'result-entry'
    folder.mkdir()

    def original(name, content):
        (folder / name).write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    source = hashlib.sha256((repo / 'clean.cmd').read_bytes()).hexdigest()
    row = {
        'id': 'entry', 'artifact_type': 'repository_entry', 'consumer': 'command',
        'target_runtime': 'wine-cmd', 'status': 'passed', 'producer_called': True,
        'producer_source': '/testbed/clean.cmd', 'producer_source_sha256': source,
        'executed_source_sha256': source, 'validation_scope': WINE_SCOPE,
        'runtime_argv': ['/usr/bin/wine', '--version'], 'runtime_exit_code': 0,
        'runtime_version': 'wine-8.0',
        'runtime_stdout_sha256': original('runtime.stdout', b'wine-8.0\n'),
        'runtime_stderr_sha256': original('runtime.stderr', b''),
        'argv': ['/usr/bin/wine', 'cmd.exe', '/d', '/v:off', '/u', '/s', '/c',
                 '%MONOPLANER_CMD_INVOCATION%'],
        'cmd_invocation': wine_command('/testbed/clean.cmd', ['--preview']),
        'cmd_builtin_output_encoding': 'utf-16-le',
        'exit_code': 0, 'producer_result_sha256': original('command.stdout', b'preview\r\n'),
        'stderr_sha256': original('command.stderr', b''),
    }
    output = tmp_path / 'result.json'
    def verify(value=row):
        output.write_text(json.dumps({'checks': [value]}), encoding='utf-8')
        return validate_execution_result(
            output, [declaration()], repository=repo,
            map_path=lambda p: '/testbed' + (
                '/' + Path(p).relative_to(repo).as_posix() if Path(p) != repo else ''))
    return row, folder, verify


def test_receipt_accepts_executed_path_scope_without_fictitious_compilation(tmp_path):
    row, folder, verify = evidence(tmp_path)
    assert verify()['checks'][0]['status'] == 'passed'
    assert not list(folder.glob('compile.*'))


@pytest.mark.parametrize('changes', [
    {'compile_exit_code': 0, 'compile_argv': ['echo', 'pretend']},
    {'validation_scope': 'native-windows-fully-verified'},
    {'executed_source_sha256': '0' * 64},
    {'runtime_exit_code': 1},
    {'argv': ['/usr/bin/wine', 'cmd.exe', '/c', 'different.cmd']},
    {'exit_code': 7},
])
def test_forged_or_wrong_execution_receipts_are_rejected(tmp_path, changes):
    row, _, verify = evidence(tmp_path)
    with pytest.raises(ValueError):
        verify({**row, **changes})


def test_runtime_log_and_original_entry_tampering_are_rejected(tmp_path):
    row, folder, verify = evidence(tmp_path)
    (folder / 'runtime.stdout').write_bytes(b'wine-9.0\n')
    with pytest.raises(ValueError, match='original artifact bytes'):
        verify()
    (folder / 'runtime.stdout').write_bytes(b'wine-8.0\n')
    (tmp_path / 'repo/clean.cmd').write_bytes(b'changed')
    with pytest.raises(ValueError, match='original artifact bytes'):
        verify()


@pytest.mark.parametrize('argument', ['%TEMP%', '!name!', 'a"b', 'x&exit', 'a|b', 'x>y', '^x', '\nexit'])
def test_expansion_or_control_arguments_cannot_masquerade_as_literals(argument):
    with pytest.raises(ValueError, match='literal paths/argv'):
        declaration(argv=[argument])


def test_wrong_entry_type_and_unconfigured_environment_cannot_be_confirmed(tmp_path):
    with pytest.raises(ValueError, match='BAT/CMD'):
        declaration(entry_file='producer.py')
    target = {'artifact_type': 'repository_entry', 'target_runtime': 'wine-cmd',
              'entry': 'clean.cmd', 'argv': []}
    with pytest.raises(ValueError, match='unavailable for this project'):
        require_targets([target], repo=tmp_path, root=tmp_path / 'state')
    assert describe(repo=tmp_path, root=tmp_path / 'state')['wine_cmd']['available'] is False
    with pytest.raises(ValueError, match='no supported execution adapter'):
        require_targets([{**target, 'target_runtime': 'windows-cmd'}])


def test_selected_image_is_checked_without_executing_or_creating_environment(tmp_path, monkeypatch):
    from agent.execution import projects
    image = 'sha256:' + 'a' * 64
    monkeypatch.setattr(projects, 'workspace_binding', lambda *a, **kw: {'image': image})
    calls = []
    def inspect(args):
        calls.append(args)
        return json.dumps([{'Id': image, 'Os': 'linux', 'Architecture': 'amd64',
                            'Config': {'Labels': {'org.monoplanner.wine-cmd': '1'}}}])
    monkeypatch.setattr(projects, 'docker_query', inspect)
    target = {'artifact_type': 'repository_entry', 'target_runtime': 'wine-cmd',
              'entry': 'clean.cmd', 'argv': []}
    require_targets([target], repo=tmp_path)
    assert calls == [['image', 'inspect', image]]
    assert not list(tmp_path.iterdir())
