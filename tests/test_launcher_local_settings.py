"""根目录使用完整运行时后，显式状态位置不会被便携默认值替换。"""

import json
import os
from pathlib import Path
import subprocess

import pytest


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows service launcher')


def configure(tmp_path, settings=None, *, environment=None, portable=True, action='Start'):
    project = tmp_path / 'project'
    project.mkdir()
    if portable:
        (project / 'portable.json').write_text('{}', 'utf-8')
    if settings is not None:
        (project / 'launcher.local.json').write_text(json.dumps(settings), 'utf-8')
    source = (Path(__file__).resolve().parents[1] / 'scripts/start-app.ps1').read_text('utf-8-sig')
    prefix = source[:source.index('$taskMutex = $null')]
    script = tmp_path / 'configuration.ps1'
    script.write_text(prefix + '''
[Console]::OutputEncoding = [Text.Encoding]::UTF8
@{ state_directory = $env:MONOPLANER_STATE_DIR; portable = $taskPortable } | ConvertTo-Json
''', 'utf-8-sig')
    env = dict(os.environ)
    env.pop('MONOPLANER_STATE_DIR', None)
    if environment:
        env['MONOPLANER_STATE_DIR'] = environment
    return subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
         '-File', str(script), '-ProjectRoot', str(project), '-Action', action],
        env=env, capture_output=True, timeout=15,
    )


def test_local_settings_keep_the_existing_state_directory(tmp_path):
    state = tmp_path / 'existing-state'
    result = configure(tmp_path, {'version': 1, 'state_directory': str(state)})
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)['state_directory']) == state


def test_explicit_environment_overrides_even_invalid_local_settings(tmp_path):
    state = tmp_path / 'explicit-state'
    result = configure(tmp_path, {'version': 99}, environment=str(state))
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)['state_directory']) == state


@pytest.mark.parametrize('settings', [
    {'version': 99, 'state_directory': 'C:\\state'},
    {'version': 1, 'state_directory': 'relative-state'},
    {'version': 1, 'state_directory': 'C:relative-state'},
    {'version': 1, 'state_directory': ''},
    {'version': 1, 'state_directory': 7},
    {'version': 1},
    {'version': 1, 'state_directory': 'C:\\state', 'unexpected': True},
])
def test_invalid_local_settings_do_not_silently_switch_history(tmp_path, settings):
    result = configure(tmp_path, settings)
    assert result.returncode != 0


def test_missing_local_settings_keep_portable_default(tmp_path):
    result = configure(tmp_path)
    assert result.returncode == 0, result.stderr
    expected = Path(os.environ['LOCALAPPDATA']) / 'MonoPlanner/Data/state'
    assert Path(json.loads(result.stdout)['state_directory']) == expected


def test_source_checkout_without_local_settings_keeps_existing_default(tmp_path):
    result = configure(tmp_path, portable=False)
    assert result.returncode == 0, result.stderr
    assert not json.loads(result.stdout)['state_directory']


def test_invalid_local_settings_cannot_prevent_an_explicit_stop(tmp_path):
    result = configure(tmp_path, {'version': 99}, action='Stop')
    assert result.returncode == 0, result.stderr
