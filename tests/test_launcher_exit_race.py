"""Windows 退出竞争必须核验原进程，不能吞掉仍存活进程的拒绝错误。"""

import base64
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows service launcher')
@pytest.mark.parametrize('exited', [True, False])
def test_kill_error_only_succeeds_when_the_same_process_is_confirmed_exited(exited):
    source = (Path(__file__).resolve().parents[1] / 'scripts/start-app.ps1').read_text('utf-8-sig')
    function = source[source.index('function Stop-ProjectService('):
                      source.index('# 隐藏启动服务进程')]
    script = r'''
$ErrorActionPreference = 'Stop'
$env:MONOPLANER_DESKTOP_JOB = ''
$taskProject = 'E:\fixture'
$script:taskWhen = [DateTime]::Parse('2026-01-01T00:00:00Z').ToUniversalTime()
class TaskProcessProbe {
    [IntPtr]$Handle = [IntPtr]1
    [DateTime]$StartTime
    [bool]$Exited
    [int]$Calls = 0
    TaskProcessProbe([bool]$exited, [DateTime]$when) {
        $this.Exited = $exited
        $this.StartTime = $when
    }
    [void] Kill() {
        $this.Calls++
        throw [ComponentModel.Win32Exception]::new(5)
    }
    [bool] WaitForExit([int]$milliseconds) { return $this.Exited }
    [void] Dispose() {}
}
function Get-CimInstance {
    param($ClassName)
    return [pscustomobject]@{
        ProcessId = 1234; ParentProcessId = 0
        CreationDate = $script:taskWhen; CommandLine = 'fixture'
    }
}
function Get-Process {
    param($Id, $ErrorAction)
    return $script:taskProbe
}
'''
    script += '\n' + function + '\n'
    script += '$script:taskProbe = [TaskProcessProbe]::new($' + str(exited).lower() + ', $script:taskWhen)\n'
    script += '''
try {
    Stop-ProjectService -Owners @(1234)
    if ($script:taskProbe.Calls -ne 1) { exit 8 }
    Write-Output 'confirmed-exited'
    exit 0
} catch {
    if ($script:taskProbe.Calls -ne 1) { exit 8 }
    Write-Output 'stop-not-confirmed'
    exit 7
}
'''
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand',
         base64.b64encode(script.encode('utf-16le')).decode('ascii')],
        capture_output=True, timeout=15,
    )
    assert result.returncode == (0 if exited else 7), (result.stdout, result.stderr)
    assert (b'confirmed-exited' if exited else b'stop-not-confirmed') in result.stdout
