param(
    [ValidateSet('Start','Stop','Restart')][string]$Action = 'Start',
    [switch]$NoBrowser,
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [ValidateRange(1024,65535)][int]$BackendPort = 2024,
    [ValidateRange(1024,65535)][int]$FrontendPort = 3000,
    [ValidateRange(1,180)][int]$StartupTimeout = 90
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskProject = [IO.Path]::GetFullPath($ProjectRoot)
$taskLogs = Join-Path $taskProject 'validation/launcher-logs'
$taskMutex = $null
$taskLockHeld = $false
$taskClient = $null

function Get-Listener([int]$Port) {
    return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty OwningProcess -Unique)
}

function Test-ProjectProcess([int]$ProcessId) {
    for ($depth = 0; $ProcessId -gt 0 -and $depth -lt 5; $depth++) {
        $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId"
        if (-not $taskProcess) { return $false }
        if ($taskProcess.CommandLine -and
            $taskProcess.CommandLine.IndexOf($taskProject + [IO.Path]::DirectorySeparatorChar,
                                            [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
        $ProcessId = $taskProcess.ParentProcessId
    }
    return $false
}

function Get-ServiceText([int]$Port, [string]$Suffix) {
    $taskResponse = $taskClient.GetAsync("http://127.0.0.1:$Port$Suffix").GetAwaiter().GetResult()
    try {
        if (-not $taskResponse.IsSuccessStatusCode) { throw "HTTP $([int]$taskResponse.StatusCode): $Suffix" }
        return $taskResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    } finally { $taskResponse.Dispose() }
}

function Test-Ready([string]$Kind, [int]$Port) {
    try {
        if ($Kind -eq 'backend') {
            $taskValue = (Get-ServiceText $Port '/service') | ConvertFrom-Json
            return $taskValue.product -eq 'MonoPlanner' -and $taskValue.project -eq $taskExpected.project -and
                   $taskValue.source_hash -eq $taskExpected.source_hash
        }
        $taskBody = Get-ServiceText $Port '/'
        if ($taskBody -notmatch '<title>MonoPlan[n]?er(?: V0.9)?</title>') { return $false }
        $taskScriptPattern = '<script[^>]+src="(/_next/static/[^"?]+(?:\?[^" ]*)?)"'
        $taskScripts = @([regex]::Matches($taskBody, $taskScriptPattern) | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
        if ($taskScripts.Count -eq 0) { return $false }
        $taskHtml = Join-Path $taskProject 'web/.next/server/app/index.html'
        if (Test-Path -LiteralPath $taskHtml) {
            $taskBuiltScripts = @([regex]::Matches([IO.File]::ReadAllText($taskHtml), $taskScriptPattern) | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
            if (@(Compare-Object $taskScripts $taskBuiltScripts).Count -gt 0) { return $false }
        }
        foreach ($taskScript in $taskScripts) { $null = Get-ServiceText $Port $taskScript }
        return $true
    } catch { return $false }
}

function Assert-BackendIdle {
    if (@(Get-Listener $BackendPort).Count -eq 0) { return }
    try {
        $taskQueue = ((Get-ServiceText $BackendPort '/metrics?format=json') | ConvertFrom-Json).queue
        if ($null -eq $taskQueue -or $null -eq $taskQueue.n_running -or $null -eq $taskQueue.n_pending) { throw '任务状态不完整' }
    } catch { throw '无法确认任务是否已停止；未关闭服务。请先在工作界面中暂停任务，再尝试重启或退出。' }
    if ($taskQueue.n_running -gt 0 -or $taskQueue.n_pending -gt 0) {
        throw '任务正在运行或等待执行；未关闭服务。请先在工作界面中暂停任务，并等待当前请求或操作结束。'
    }
}

function Stop-ProjectService([int[]]$Owners) {
    foreach ($taskOwner in $Owners) {
        $taskRoot = Get-CimInstance Win32_Process -Filter "ProcessId=$taskOwner"
        if (-not $taskRoot) { continue }
        $taskParent = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskRoot.ParentProcessId)"
        # Stop the langgraph wrapper too, so it cannot relaunch its worker.
        if ($taskParent -and $taskParent.CommandLine -and
            $taskParent.CommandLine.Contains((Join-Path $taskProject '.venv-web\Scripts\langgraph.exe'))) { $taskRoot = $taskParent }
        $taskProcesses = @(Get-CimInstance Win32_Process)
        $taskOwned = @($taskRoot)
        for ($taskDepth = 0; $taskDepth -lt 8; $taskDepth++) {
            $taskChildren = @($taskProcesses | Where-Object {
                $_.ParentProcessId -in @($taskOwned.ProcessId) -and $_.ProcessId -notin @($taskOwned.ProcessId)
            })
            if ($taskChildren.Count -eq 0) { break }
            $taskOwned += $taskChildren
        }
        foreach ($taskProcess in @($taskOwned | Sort-Object CreationDate -Descending)) {
            $taskCurrent = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskProcess.ProcessId)"
            if ($taskCurrent -and $taskCurrent.CreationDate -eq $taskProcess.CreationDate) {
                Stop-Process -Id $taskCurrent.ProcessId -ErrorAction SilentlyContinue
            }
        }
    }
}

function Start-ServiceProcess([string]$Executable, [string[]]$Arguments, [string]$Directory, [string]$Name) {
    $taskQuoted = @($Arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' })
    return Start-Process -FilePath $Executable -ArgumentList $taskQuoted -WorkingDirectory $Directory `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRunLogs "$Name.stdout.log") `
        -RedirectStandardError (Join-Path $taskRunLogs "$Name.stderr.log")
}

function Wait-Service([string]$Kind, [int]$Port, $Process) {
    $taskDeadline = [DateTime]::UtcNow.AddSeconds($StartupTimeout)
    do {
        if (Test-Ready $Kind $Port) { return }
        if ($Process -and $Process.HasExited) {
            throw "$Kind 启动进程已退出；请查看 $taskRunLogs 中的日志。"
        }
        Start-Sleep -Milliseconds 300
    } while ([DateTime]::UtcNow -lt $taskDeadline)
    throw "$Kind 未在 $StartupTimeout 秒内就绪；请查看 $taskRunLogs 中的日志。"
}

try {
    New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
    $taskHash = [Security.Cryptography.SHA256]::Create()
    try { $taskKey = [BitConverter]::ToString($taskHash.ComputeHash([Text.Encoding]::UTF8.GetBytes($taskProject.ToLowerInvariant()))).Replace('-', '') }
    finally { $taskHash.Dispose() }
    $taskMutex = New-Object Threading.Mutex($false, "Local\MonoPlaner-$taskKey")
    try { $taskLockHeld = $taskMutex.WaitOne($StartupTimeout * 1000) }
    catch [Threading.AbandonedMutexException] { $taskLockHeld = $true }
    if (-not $taskLockHeld) { throw '另一个启动器仍在启动服务，请稍后再试。' }
    $taskRunLogs = Join-Path $taskLogs ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Path $taskRunLogs | Out-Null
    $taskBackend = Join-Path $taskProject '.venv-web/Scripts/langgraph.exe'
    $taskNext = Join-Path $taskProject 'web/node_modules/next/dist/bin/next'
    $taskNode = Get-Command node.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    Add-Type -AssemblyName System.Net.Http
    $taskHandler = New-Object Net.Http.HttpClientHandler
    $taskHandler.UseProxy = $false
    $taskClient = New-Object Net.Http.HttpClient($taskHandler)
    $taskClient.Timeout = [TimeSpan]::FromSeconds(3)
    $taskBackendPids = @(Get-Listener $BackendPort)
    $taskFrontendPids = @(Get-Listener $FrontendPort)
    foreach ($taskOwner in @($taskBackendPids) + @($taskFrontendPids)) {
        if (-not (Test-ProjectProcess $taskOwner)) { throw '启动端口被其他程序占用；不会关闭或替换该程序。' }
    }
    if ($Action -ne 'Start') {
        Assert-BackendIdle
        Stop-ProjectService (@($taskFrontendPids) + @($taskBackendPids))
        if ($Action -eq 'Stop') {
            @{ stopped = $true; project = $taskProject } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskLogs 'latest.json') -Encoding UTF8
            exit 0
        }
        $taskFrontendPids = @()
        $taskBackendPids = @()
    }
    if (-not (Test-Path -LiteralPath $taskBackend)) { throw '未找到 .venv-web/Scripts/langgraph.exe；请按 README 安装后端依赖。' }
    if (-not $taskNode -or -not (Test-Path -LiteralPath $taskNext)) { throw '未找到 Node.js 或前端依赖；请按 README 完成安装。' }
    $taskPython = Join-Path $taskProject '.venv-web/Scripts/python.exe'
    $taskExpected = (& $taskPython -B -X utf8 (Join-Path $taskProject 'agent/server/service.py')) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not $taskExpected.source_hash) { throw '无法核对后端版本；未复用旧服务。' }
    $taskStaleBackend = $taskBackendPids.Count -gt 0 -and -not (Test-Ready 'backend' $BackendPort)
    $taskStaleFrontend = $taskFrontendPids.Count -gt 0 -and -not (Test-Ready 'frontend' $FrontendPort)
    if ($taskStaleBackend -or $taskStaleFrontend) {
        Assert-BackendIdle
        if ($taskStaleFrontend) { Stop-ProjectService $taskFrontendPids; $taskFrontendPids = @() }
        if ($taskStaleBackend) { Stop-ProjectService $taskBackendPids; $taskBackendPids = @() }
    }
    $env:PYTHONUTF8 = '1'
    $env:LOG_COLOR = 'false'
    $env:LANGSMITH_TRACING = 'false'
    $taskBackendProcess = $null
    $taskFrontendProcess = $null
    if ($taskBackendPids.Count -eq 0) {
        $taskBackendProcess = Start-ServiceProcess $taskBackend @('dev','--host','127.0.0.1','--port',"$BackendPort",'--no-browser','--no-reload') $taskProject 'backend'
    }
    Wait-Service 'backend' $BackendPort $taskBackendProcess
    $taskMode = if (Test-Path -LiteralPath (Join-Path $taskProject 'web/.next/BUILD_ID')) { 'start' } else { 'dev' }
    if ($taskFrontendPids.Count -eq 0) {
        $taskFrontendProcess = Start-ServiceProcess $taskNode.Source @($taskNext,$taskMode,'--hostname','127.0.0.1','--port',"$FrontendPort") (Join-Path $taskProject 'web') 'frontend'
    }
    Wait-Service 'frontend' $FrontendPort $taskFrontendProcess
    $taskResult = @{ project = $taskProject; url = "http://127.0.0.1:$FrontendPort/"; frontend_mode = $taskMode;
        backend_reused = $taskBackendPids.Count -gt 0; frontend_reused = $taskFrontendPids.Count -gt 0;
        backend_pids = @(Get-Listener $BackendPort); frontend_pids = @(Get-Listener $FrontendPort);
        logs = $taskRunLogs; ready = $true; time = (Get-Date).ToString('o') }
    $taskResult.backend_updated = $taskStaleBackend
    $taskResult.frontend_updated = $taskStaleFrontend
    $taskJson = $taskResult | ConvertTo-Json -Depth 4
    $taskJson | Set-Content -LiteralPath (Join-Path $taskRunLogs 'result.json') -Encoding UTF8
    $taskJson | Set-Content -LiteralPath (Join-Path $taskLogs 'latest.json') -Encoding UTF8
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$FrontendPort/" }
    exit 0
} catch {
    $taskError = $_.Exception.Message
    if (Test-Path -LiteralPath $taskLogs) { $taskError | Set-Content -LiteralPath (Join-Path $taskLogs 'last-error.txt') -Encoding UTF8 }
    Write-Error $taskError -ErrorAction Continue
    exit 1
} finally {
    if ($taskClient) { $taskClient.Dispose() }
    if ($taskLockHeld) { $taskMutex.ReleaseMutex() }
    if ($taskMutex) { $taskMutex.Dispose() }
}
