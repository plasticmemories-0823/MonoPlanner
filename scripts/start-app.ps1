param(
    [ValidateSet('Start', 'Stop', 'Restart')]
    [string]$Action = 'Start',
    [switch]$NoBrowser,
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [ValidateRange(1024, 65535)]
    [int]$BackendPort = $(if ($env:MONOPLANER_BACKEND_PORT) { [int]$env:MONOPLANER_BACKEND_PORT } else { 2024 }),
    [ValidateRange(1024, 65535)]
    [int]$FrontendPort = $(if ($env:MONOPLANER_UI_PORT) { [int]$env:MONOPLANER_UI_PORT } else { 3000 }),
    [ValidateRange(1, 180)]
    [int]$StartupTimeout = 90
)

# 启动参数、项目目录与日志位置。
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskProject = [IO.Path]::GetFullPath($ProjectRoot)
$taskPortable = Test-Path -LiteralPath (Join-Path $taskProject 'portable.json')
$taskData = Join-Path $env:LOCALAPPDATA 'MonoPlanner/Data'
$taskLogs = if ($taskPortable) {
    Join-Path $taskData 'launcher-logs'
} else {
    Join-Path $taskProject 'validation/launcher-logs'
}
$taskMutex = $null
$taskLockHeld = $false
$taskClient = $null
$taskCloseSignal = $null
$taskProcessSnapshot = $null

function Assert-StartupRequested {
    if ($Action -ne 'Stop' -and $taskCloseSignal -and $taskCloseSignal.WaitOne(0)) {
        throw [OperationCanceledException]::new('窗口正在退出，已取消启动。')
    }
}

# 监听端口、进程归属与服务身份探测。
function Get-Listener([int]$Port) {
    return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
}

function Test-ProjectProcess([int]$ProcessId) {
    if ($null -eq $script:taskProcessSnapshot) {
        $script:taskProcessSnapshot = @(Get-CimInstance Win32_Process)
    }
    for ($depth = 0; $ProcessId -gt 0 -and $depth -lt 5; $depth++) {
        $taskProcess = $script:taskProcessSnapshot | Where-Object ProcessId -eq $ProcessId | Select-Object -First 1
        if (-not $taskProcess) {
            return $false
        }
        if ($taskProcess.CommandLine -and
            $taskProcess.CommandLine.IndexOf($taskProject + [IO.Path]::DirectorySeparatorChar,
                [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $true
        }
        $ProcessId = $taskProcess.ParentProcessId
    }
    return $false
}

# 退出先封锁新操作并取消正在运行的请求；服务不可达也不阻止回收自有进程。
function Request-BackendShutdown([int[]]$Owners) {
    if ($Owners.Count -eq 0) {
        return @{ shutdown = $true; termination_confirmed = $true; complete = $true }
    }
    $taskDeadline = [DateTime]::UtcNow.AddSeconds(6)
    $taskReceipt = @{ shutdown = $false; termination_confirmed = $false; complete = $false }
    do {
        $taskRequest = New-Object Net.Http.HttpRequestMessage([Net.Http.HttpMethod]::Post, "http://127.0.0.1:$BackendPort/service/shutdown")
        if ($env:MONOPLANER_SERVICE_TOKEN) {
            $null = $taskRequest.Headers.TryAddWithoutValidation('X-MonoPlanner-Service', $env:MONOPLANER_SERVICE_TOKEN)
        }
        $taskRequest.Content = New-Object Net.Http.StringContent('{}', [Text.Encoding]::UTF8, 'application/json')
        try {
            $taskResponse = $taskClient.SendAsync($taskRequest).GetAwaiter().GetResult()
            try {
                if (-not $taskResponse.IsSuccessStatusCode) {
                    throw "停止确认接口返回 HTTP $([int]$taskResponse.StatusCode)"
                }
                $taskReceipt = $taskResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
                if ($taskReceipt.termination_confirmed) {
                    break
                }
            } finally {
                $taskResponse.Dispose()
            }
        } catch {
            return @{ shutdown = $false; termination_confirmed = $false; complete = $false;
                error = $_.Exception.Message; requires_review = $true }
        } finally {
            $taskRequest.Dispose()
        }
    } while ([DateTime]::UtcNow.AddSeconds(3) -lt $taskDeadline)
    return $taskReceipt
}

# 接管同一项目已存在的服务，保证桌面退出时不会留下脱离窗口的后台进程。
function Initialize-DesktopJobType {
    if ('MonoPlannerDesktopJob' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MonoPlannerDesktopJob {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern IntPtr OpenJobObject(uint access, bool inherit, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool member);
    [DllImport("kernel32.dll")]
    public static extern bool CloseHandle(IntPtr handle);
}
'@
}

function Join-DesktopLifetime([int[]]$Owners) {
    if (-not $env:MONOPLANER_DESKTOP_JOB -or $Owners.Count -eq 0) {
        return
    }
    Initialize-DesktopJobType
    $taskJob = [MonoPlannerDesktopJob]::OpenJobObject(0x5, $false, $env:MONOPLANER_DESKTOP_JOB)
    if ($taskJob -eq [IntPtr]::Zero) {
        throw '桌面实例已退出，不能接管服务。'
    }
    try {
        # 旧的无窗口服务只接管已核验的内部运行时；打开的用户软件保持独立。
        $taskDocker = Get-Command docker.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        $taskInternal = @($Owners)
        for ($taskDepth = 0; $taskDepth -lt 8; $taskDepth++) {
            $taskDockerParents = @($script:taskProcessSnapshot | Where-Object {
                $_.Name -eq 'docker.exe' -and $_.ProcessId -in $taskInternal
            } | ForEach-Object ProcessId)
            $taskChildren = @($script:taskProcessSnapshot | Where-Object {
                $_.ParentProcessId -in $taskInternal -and $_.ProcessId -notin $taskInternal -and (
                    ($_.ExecutablePath -and $_.ExecutablePath.StartsWith(
                        (Join-Path $taskProject 'launcher/runtime') + '\', [StringComparison]::OrdinalIgnoreCase) -and
                        $_.Name -in @('python.exe', 'node.exe', 'git.exe', 'git-remote-https.exe')) -or
                    ($taskDocker -and $_.ExecutablePath -eq $taskDocker.Source) -or
                    ($_.Name -eq 'docker-buildx.exe' -and $_.ParentProcessId -in $taskDockerParents)
                )
            })
            if ($taskChildren.Count -eq 0) { break }
            $taskInternal += @($taskChildren.ProcessId)
        }
        foreach ($taskOwner in $taskInternal) {
            $taskOriginal = $script:taskProcessSnapshot | Where-Object ProcessId -eq $taskOwner | Select-Object -First 1
            if (-not $taskOriginal) { throw '服务身份记录已消失，不能接管。' }
            $taskProcess = [Diagnostics.Process]::GetProcessById($taskOwner)
            try {
                $null = $taskProcess.Handle
                if ([Math]::Abs(($taskProcess.StartTime.ToUniversalTime() - $taskOriginal.CreationDate.ToUniversalTime()).TotalMilliseconds) -ge 1) {
                    throw '服务进程身份已变化，不能接管。'
                }
                $taskMember = $false
                if (-not [MonoPlannerDesktopJob]::IsProcessInJob($taskProcess.Handle, $taskJob, [ref]$taskMember)) {
                    throw '无法核验服务进程的桌面归属。'
                }
                if (-not $taskMember -and -not [MonoPlannerDesktopJob]::AssignProcessToJobObject($taskJob, $taskProcess.Handle)) {
                    throw '无法接管已有服务；请先退出原实例。'
                }
            } finally {
                $taskProcess.Dispose()
            }
        }
    } finally {
        $null = [MonoPlannerDesktopJob]::CloseHandle($taskJob)
    }
}

function Get-ServiceText([int]$Port, [string]$Suffix) {
    $taskRequest = New-Object Net.Http.HttpRequestMessage([Net.Http.HttpMethod]::Get, "http://127.0.0.1:$Port$Suffix")
    if ($Port -eq $BackendPort) {
        if ($env:MONOPLANER_SERVICE_TOKEN) {
            $null = $taskRequest.Headers.TryAddWithoutValidation('X-MonoPlanner-Service', $env:MONOPLANER_SERVICE_TOKEN)
        }
    }
    try {
        $taskResponse = $taskClient.SendAsync($taskRequest).GetAwaiter().GetResult()
    } finally {
        $taskRequest.Dispose()
    }
    try {
        if (-not $taskResponse.IsSuccessStatusCode) {
            throw "HTTP $([int]$taskResponse.StatusCode): $Suffix"
        }
        return $taskResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    } finally {
        $taskResponse.Dispose()
    }
}

function Test-Ready([string]$Kind, [int]$Port) {
    try {
        if ($Kind -eq 'backend') {
            $taskValue = (Get-ServiceText $Port '/service') | ConvertFrom-Json
            return $taskValue.product -eq 'MonoPlanner' -and
                $taskValue.project_id -eq $taskExpected.project_id -and
                $taskValue.version -eq $taskExpected.version -and
                $taskValue.source_hash -eq $taskExpected.source_hash
        }
        $taskBody = Get-ServiceText $Port '/'
        if ($taskBody -notmatch '<title>MonoPlanner 1\.0</title>') {
            return $false
        }
        $taskScriptPattern = '<script[^>]+src="(/_next/static/[^"?]+(?:\?[^" ]*)?)"'
        $taskScripts = @([regex]::Matches($taskBody, $taskScriptPattern) | ForEach-Object {
                $_.Groups[1].Value
            } | Sort-Object -Unique)
        if ($taskScripts.Count -eq 0) {
            return $false
        }
        $taskHtml = Join-Path $taskProject 'web/.next/server/app/index.html'
        if (Test-Path -LiteralPath $taskHtml) {
            $taskBuiltScripts = @([regex]::Matches([IO.File]::ReadAllText($taskHtml), $taskScriptPattern) | ForEach-Object {
                    $_.Groups[1].Value
                } | Sort-Object -Unique)
            if (@(Compare-Object $taskScripts $taskBuiltScripts).Count -gt 0) {
                return $false
            }
        }
        foreach ($taskScript in $taskScripts) {
            $null = Get-ServiceText $Port $taskScript
        }
        return $true
    } catch {
        return $false
    }
}

# 关闭或重启前确认任务已停止，并管理所属服务进程。
function Assert-BackendIdle {
    if (@(Get-Listener $BackendPort).Count -eq 0) {
        return
    }
    try {
        $taskStatus = (Get-ServiceText $BackendPort '/service/status') | ConvertFrom-Json
        $taskQueue = $taskStatus.queue
        if ($null -eq $taskQueue -or $null -eq $taskQueue.n_running -or $null -eq $taskQueue.n_pending) {
            throw '任务状态不完整'
        }
    } catch {
        throw '无法确认任务是否已停止；未关闭服务。请先在工作界面中暂停任务，再尝试重启或退出。'
    }
    if ($taskStatus.PSObject.Properties['environment_preparations'] -and $taskStatus.environment_preparations -gt 0) {
        throw '项目环境仍在准备；未关闭服务。请在工作目录窗口取消准备，等待安装容器停止后再退出。'
    }
    if ($taskQueue.n_running -gt 0 -or $taskQueue.n_pending -gt 0) {
        throw '任务正在运行或等待执行；未关闭服务。请先在工作界面中暂停任务，并等待当前请求或操作结束。'
    }
}

function Stop-ProjectService([int[]]$Owners) {
    $taskProcesses = @(Get-CimInstance Win32_Process)
    $taskAllOwned = @()
    foreach ($taskOwner in $Owners) {
        $taskRoot = $taskProcesses | Where-Object ProcessId -eq $taskOwner | Select-Object -First 1
        if (-not $taskRoot) {
            continue
        }
        $taskParent = $taskProcesses | Where-Object ProcessId -eq $taskRoot.ParentProcessId | Select-Object -First 1
        # 同时停止 langgraph 包装进程，防止它重新拉起工作进程。
        if ($taskParent -and $taskParent.CommandLine -and
            $taskParent.CommandLine.Contains((Join-Path $taskProject '.venv-web\Scripts\langgraph.exe'))) {
            $taskRoot = $taskParent
        }
        $taskOwned = @($taskRoot)
        for ($taskDepth = 0; $taskDepth -lt 8; $taskDepth++) {
            $taskChildren = @($taskProcesses | Where-Object {
                    $_.ParentProcessId -in @($taskOwned.ProcessId) -and $_.ProcessId -notin @($taskOwned.ProcessId)
                })
            if ($taskChildren.Count -eq 0) {
                break
            }
            $taskOwned += $taskChildren
        }
        $taskAllOwned += $taskOwned
    }
    $taskJob = [IntPtr]::Zero
    if ($env:MONOPLANER_DESKTOP_JOB) {
        Initialize-DesktopJobType
        $taskJob = [MonoPlannerDesktopJob]::OpenJobObject(0x4, $false, $env:MONOPLANER_DESKTOP_JOB)
        if ($taskJob -eq [IntPtr]::Zero) { throw '无法核验桌面服务组。' }
    }
    try {
        foreach ($taskProcess in @($taskAllOwned | Sort-Object ProcessId -Unique | Sort-Object CreationDate -Descending)) {
            $taskCurrent = Get-Process -Id $taskProcess.ProcessId -ErrorAction SilentlyContinue
            if (-not $taskCurrent) { continue }
            try {
                # 持有进程句柄并核验创建时间，避免 PID 重用后误停其他进程。
                $null = $taskCurrent.Handle
                if ($taskJob -ne [IntPtr]::Zero -and $taskProcess.ProcessId -notin $Owners) {
                    $taskMember = $false
                    if (-not [MonoPlannerDesktopJob]::IsProcessInJob($taskCurrent.Handle, $taskJob, [ref]$taskMember)) {
                        throw '无法核验子进程归属。'
                    }
                    if (-not $taskMember) { continue }
                }
                if ([Math]::Abs(($taskCurrent.StartTime.ToUniversalTime() - $taskProcess.CreationDate.ToUniversalTime()).TotalMilliseconds) -lt 1) {
                    $taskCurrent.Kill()
                    $null = $taskCurrent.WaitForExit(500)
                }
            } finally {
                $taskCurrent.Dispose()
            }
        }
    } finally {
        if ($taskJob -ne [IntPtr]::Zero) { $null = [MonoPlannerDesktopJob]::CloseHandle($taskJob) }
    }
}

# 隐藏启动服务进程，并等待其通过就绪检查。
function Start-ServiceProcess(
    [string]$Executable,
    [string[]]$Arguments,
    [string]$Directory,
    [string]$Name
) {
    Assert-StartupRequested
    $taskQuoted = @($Arguments | ForEach-Object {
            '"' + $_.Replace('"', '\"') + '"'
        })
    return Start-Process -FilePath $Executable -ArgumentList $taskQuoted -WorkingDirectory $Directory `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRunLogs "$Name.stdout.log") `
        -RedirectStandardError (Join-Path $taskRunLogs "$Name.stderr.log")
}

function Wait-Service([string]$Kind, [int]$Port, $Process) {
    $taskDeadline = [DateTime]::UtcNow.AddSeconds($StartupTimeout)
    do {
        Assert-StartupRequested
        if (Test-Ready $Kind $Port) {
            return
        }
        if ($Process -and $Process.HasExited) {
            throw "$Kind 启动进程已退出；请查看 $taskRunLogs 中的日志。"
        }
        Start-Sleep -Milliseconds 300
    } while ([DateTime]::UtcNow -lt $taskDeadline)
    throw "$Kind 未在 $StartupTimeout 秒内就绪；请查看 $taskRunLogs 中的日志。"
}

# 获取项目启动锁并建立本次运行日志。
try {
    if ($env:MONOPLANER_START_GATE) {
        $taskGate = [Threading.EventWaitHandle]::OpenExisting($env:MONOPLANER_START_GATE)
        try {
            if (-not $taskGate.WaitOne(5000)) { throw '桌面未接管服务控制器，已停止启动。' }
        } finally {
            $taskGate.Dispose()
        }
    }
    if ($env:MONOPLANER_CLOSE_SIGNAL) {
        $taskCloseSignal = [Threading.EventWaitHandle]::OpenExisting($env:MONOPLANER_CLOSE_SIGNAL)
    }
    Assert-StartupRequested
    try { New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null }
    catch { if ($Action -ne 'Stop') { throw } }
    $taskHash = [Security.Cryptography.SHA256]::Create()
    try {
        $taskKey = [BitConverter]::ToString($taskHash.ComputeHash([Text.Encoding]::UTF8.GetBytes($taskProject.ToLowerInvariant()))).Replace('-', '')
    } finally {
        $taskHash.Dispose()
    }
    $taskMutex = New-Object Threading.Mutex($false, "Local\MonoPlaner-$taskKey")
    try {
        $taskLockDeadline = [DateTime]::UtcNow.AddSeconds($(if ($Action -eq 'Stop') { 3 } else { $StartupTimeout }))
        do {
            Assert-StartupRequested
            $taskLockHeld = $taskMutex.WaitOne(100)
        } while (-not $taskLockHeld -and [DateTime]::UtcNow -lt $taskLockDeadline)
    } catch [Threading.AbandonedMutexException] {
        $taskLockHeld = $true
    }
    if (-not $taskLockHeld) {
        throw '另一个启动器仍在启动服务，请稍后再试。'
    }
    $taskRunLogs = Join-Path $taskLogs ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
    if ($Action -ne 'Stop') {
        New-Item -ItemType Directory -Path $taskRunLogs | Out-Null
    }

    # 选择源码或便携发行版的运行时路径。
    $taskBackend = Join-Path $taskProject '.venv-web/Scripts/python.exe'
    $taskNext = Join-Path $taskProject 'web/node_modules/next/dist/bin/next'
    $taskNode = Get-Command node.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($taskPortable) {
        $taskDockerCommand = Get-Command docker.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        $taskDockerPath = if ($taskDockerCommand) {
            ';' + (Split-Path -Parent $taskDockerCommand.Source)
        } else {
            ''
        }
        $taskBackend = Join-Path $taskProject 'launcher/runtime/python/python.exe'
        $taskNext = Join-Path $taskProject 'web/server.js'
        $taskNode = @{
            Source = (Join-Path $taskProject 'launcher/runtime/node/node.exe')
        }
        if (-not $env:MONOPLANER_STATE_DIR) {
            $env:MONOPLANER_STATE_DIR = Join-Path $taskData 'state'
        }
        $env:MONOPLANER_BACKEND_PORT = "$BackendPort"
        $env:MONOPLANER_UI_PORT = "$FrontendPort"
        $env:PATH = (Join-Path $taskProject 'launcher/runtime/git/cmd') + ';' + (Join-Path $taskProject 'launcher/runtime/node') + $taskDockerPath + ';' + (Join-Path $env:WINDIR 'System32')
        $env:PORT = "$FrontendPort"
        $env:HOSTNAME = '127.0.0.1'
    }

    # 使用当前用户加密的实例凭证；不要通过命令行或 URL 传递。
    Add-Type -AssemblyName System.Security
    $taskAccessDirectory = Join-Path $taskData 'service-access'
    if ($Action -ne 'Stop') {
        New-Item -ItemType Directory -Path $taskAccessDirectory -Force | Out-Null
    }
    $taskAccessFile = Join-Path $taskAccessDirectory "$taskKey-$BackendPort-$FrontendPort.dpapi"
    if (Test-Path -LiteralPath $taskAccessFile) {
        $taskPlain = [Security.Cryptography.ProtectedData]::Unprotect([IO.File]::ReadAllBytes($taskAccessFile), $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        $env:MONOPLANER_SERVICE_TOKEN = [Text.Encoding]::UTF8.GetString($taskPlain)
        [Array]::Clear($taskPlain, 0, $taskPlain.Length)
    } elseif ($Action -ne 'Stop') {
        $taskRandom = New-Object byte[] 32
        $taskGenerator = [Security.Cryptography.RandomNumberGenerator]::Create()
        try {
            $taskGenerator.GetBytes($taskRandom)
        } finally {
            $taskGenerator.Dispose()
        }
        $env:MONOPLANER_SERVICE_TOKEN = [Convert]::ToBase64String($taskRandom)
        $taskProtected = [Security.Cryptography.ProtectedData]::Protect([Text.Encoding]::UTF8.GetBytes($env:MONOPLANER_SERVICE_TOKEN), $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        [IO.File]::WriteAllBytes($taskAccessFile, $taskProtected)
        [Array]::Clear($taskRandom, 0, $taskRandom.Length)
    }
    $env:MONOPLANER_BACKEND_PORT = "$BackendPort"
    $env:MONOPLANER_UI_PORT = "$FrontendPort"

    # 关闭运行时遥测，并清除外部追踪凭证。
    $env:LANGGRAPH_CLI_NO_ANALYTICS = '1'
    $env:LANGGRAPH_NO_VERSION_CHECK = '1'
    $env:NEXT_TELEMETRY_DISABLED = '1'
    $env:OTEL_SDK_DISABLED = 'true'
    $env:LANGCHAIN_TRACING_V2 = 'false'
    $env:LANGSMITH_TRACING = 'false'
    foreach ($taskCloudVariable in @('LANGSMITH_API_KEY', 'LANGCHAIN_API_KEY', 'LANGSMITH_ENDPOINT', 'LANGCHAIN_ENDPOINT')) {
        [Environment]::SetEnvironmentVariable($taskCloudVariable, $null, 'Process')
    }

    # 建立本机服务连接，核验端口归属并处理停止或重启。
    Add-Type -AssemblyName System.Net.Http
    $taskHandler = New-Object Net.Http.HttpClientHandler
    $taskHandler.UseProxy = $false
    $taskClient = New-Object Net.Http.HttpClient($taskHandler)
    $taskClient.Timeout = [TimeSpan]::FromSeconds(3)
    $taskBackendPids = @(Get-Listener $BackendPort)
    $taskFrontendPids = @(Get-Listener $FrontendPort)
    $taskForeignPids = @()
    foreach ($taskOwner in @($taskBackendPids) + @($taskFrontendPids)) {
        if (-not (Test-ProjectProcess $taskOwner)) {
            if ($Action -ne 'Stop') {
                throw '启动端口被其他程序占用；不会关闭或替换该程序。'
            }
            $taskForeignPids += $taskOwner
        }
    }
    $taskBackendPids = @($taskBackendPids | Where-Object { $_ -notin $taskForeignPids })
    $taskFrontendPids = @($taskFrontendPids | Where-Object { $_ -notin $taskForeignPids })
    if ($Action -ne 'Start') {
        $taskShutdown = Request-BackendShutdown $taskBackendPids
        Stop-ProjectService (@($taskFrontendPids) + @($taskBackendPids))
        if ($Action -eq 'Stop') {
            $taskRemaining = @(@(Get-Listener $FrontendPort) + @(Get-Listener $BackendPort) |
                Where-Object { $null -ne $_ -and $_ -notin $taskForeignPids })
            $taskStopResult = @{ stopped = ($taskRemaining.Count -eq 0); project = $taskProject;
                shutdown = $taskShutdown; foreign_pids_preserved = $taskForeignPids }
            try {
                $taskStopResult | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskLogs 'latest.json') -Encoding UTF8
            } catch { Write-Warning '停止回执无法保存；原任务记录仍需核对。' }
            if ($taskRemaining.Count -gt 0) { throw '本实例仍有服务未停止。' }
            if ($taskForeignPids.Count -gt 0) { throw '已停止本实例服务；端口上的其他程序已保留。' }
            exit 0
        }
        $taskFrontendPids = @()
        $taskBackendPids = @()
    }
    Assert-StartupRequested

    # 校验运行时与源码版本，仅在任务空闲时替换过期服务。
    if (-not (Test-Path -LiteralPath $taskBackend)) {
        throw '未找到后端 Python 运行时；请按 README 安装后端依赖。'
    }
    if (-not $taskNode -or -not (Test-Path -LiteralPath $taskNext)) {
        throw '未找到 Node.js 或前端依赖；请按 README 完成安装。'
    }
    $taskPython = if ($taskPortable) {
        $taskBackend
    } else {
        Join-Path $taskProject '.venv-web/Scripts/python.exe'
    }
    $taskExpected = (& $taskPython -B -X utf8 (Join-Path $taskProject 'agent/server/service.py')) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not $taskExpected.source_hash) {
        throw '无法核对后端版本；未复用旧服务。'
    }
    $taskStaleBackend = $taskBackendPids.Count -gt 0 -and -not (Test-Ready 'backend' $BackendPort)
    $taskStaleFrontend = $taskFrontendPids.Count -gt 0 -and -not (Test-Ready 'frontend' $FrontendPort)
    if ($taskStaleBackend -or $taskStaleFrontend) {
        # 更新只负责显示的前端不需要停止任务；替换后端仍必须确认空闲。
        if ($taskStaleBackend) {
            Assert-BackendIdle
        }
        if ($taskStaleFrontend) {
            Stop-ProjectService $taskFrontendPids;
            $taskFrontendPids = @()
        }
        if ($taskStaleBackend) {
            Stop-ProjectService $taskBackendPids;
            $taskBackendPids = @()
        }
    }
    Join-DesktopLifetime (@($taskFrontendPids) + @($taskBackendPids))

    # 先启动后端，再准备与启动前端。
    $env:PYTHONUTF8 = '1'
    $env:LOG_COLOR = 'false'
    $env:LANGSMITH_TRACING = 'false'
    $taskBackendProcess = $null
    $taskFrontendProcess = $null
    if ($taskBackendPids.Count -eq 0) {
        # 两种发行方式都将框架存储放在各自选定的状态目录。
        $taskArguments = @('-B', '-X', 'utf8', '-m', 'agent.server.portable')
        $taskBackendProcess = Start-ServiceProcess $taskBackend $taskArguments $taskProject 'backend'
    }
    Wait-Service 'backend' $BackendPort $taskBackendProcess

    $taskMode = if ($taskPortable) {
        'portable'
    } elseif (Test-Path -LiteralPath (Join-Path $taskProject 'web/.next/BUILD_ID')) {
        'start'
    } else {
        'dev'
    }
    if ($taskFrontendPids.Count -eq 0) {
        $taskArguments = if ($taskPortable) {
            @($taskNext)
        } else {
            @($taskNext, $taskMode, '--hostname', '127.0.0.1', '--port', "$FrontendPort")
        }
        # 前端进程不继承模型凭证、环境代理和 Node 选项。
        # 经过认证的后端请求必须使用固定的回环连接。
        $taskFrontendVariables = @('DEEPSEEK_API_KEY', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
            'NODE_USE_ENV_PROXY', 'NODE_OPTIONS')
        $taskFrontendEnvironment = @{}
        try {
            foreach ($taskVariable in $taskFrontendVariables) {
                $taskFrontendEnvironment[$taskVariable] = [Environment]::GetEnvironmentVariable($taskVariable, 'Process')
                [Environment]::SetEnvironmentVariable($taskVariable, $null, 'Process')
            }
            $taskFrontendProcess = Start-ServiceProcess $taskNode.Source $taskArguments (Join-Path $taskProject 'web') 'frontend'
        } finally {
            foreach ($taskVariable in $taskFrontendVariables) {
                [Environment]::SetEnvironmentVariable($taskVariable, $taskFrontendEnvironment[$taskVariable], 'Process')
            }
        }
    }
    Wait-Service 'frontend' $FrontendPort $taskFrontendProcess

    # 保存服务就绪结果，并按参数打开工作区。
    $taskResult = @{
        project = $taskProject;
        url = "http://127.0.0.1:$FrontendPort/";
        frontend_mode = $taskMode;
        backend_reused = $taskBackendPids.Count -gt 0;
        frontend_reused = $taskFrontendPids.Count -gt 0;
        backend_pids = @(Get-Listener $BackendPort);
        frontend_pids = @(Get-Listener $FrontendPort);
        logs = $taskRunLogs;
        ready = $true;
        time = (Get-Date).ToString('o')
    }
    $taskResult.backend_updated = $taskStaleBackend
    $taskResult.frontend_updated = $taskStaleFrontend
    $taskJson = $taskResult | ConvertTo-Json -Depth 4
    $taskJson | Set-Content -LiteralPath (Join-Path $taskRunLogs 'result.json') -Encoding UTF8
    $taskJson | Set-Content -LiteralPath (Join-Path $taskLogs 'latest.json') -Encoding UTF8
    if (-not $NoBrowser) {
        Start-Process "http://127.0.0.1:$FrontendPort/"
    }
    exit 0
} catch {
    # 保留失败原因，供桌面启动器展示。
    $taskError = $_.Exception.Message
    if (Test-Path -LiteralPath $taskLogs) {
        $taskError | Set-Content -LiteralPath (Join-Path $taskLogs 'last-error.txt') -Encoding UTF8
    }
    Write-Error $taskError -ErrorAction Continue
    exit 1
} finally {
    # 释放连接和项目启动锁。
    if ($taskClient) {
        $taskClient.Dispose()
    }
    if ($taskLockHeld) {
        $taskMutex.ReleaseMutex()
    }
    if ($taskMutex) {
        $taskMutex.Dispose()
    }
    if ($taskCloseSignal) {
        $taskCloseSignal.Dispose()
    }
}
