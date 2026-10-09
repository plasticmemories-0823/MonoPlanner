$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskCompiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
if (-not (Test-Path -LiteralPath $taskCompiler)) {
    $taskCompiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework/v4.0.30319/csc.exe'
}
if (-not (Test-Path -LiteralPath $taskCompiler)) { throw '需要 Windows .NET Framework C# 编译器。' }
$taskOutput = Join-Path $taskProject 'MonoPlanner.exe'
$taskSource = Join-Path $taskProject 'launcher/Program.cs'
$taskRuntime = Join-Path $taskProject 'launcher/runtime'
if (@('WebView2Loader.dll','Microsoft.Web.WebView2.Core.dll','Microsoft.Web.WebView2.WinForms.dll','LICENSE.txt','version.txt' |
        Where-Object { -not (Test-Path -LiteralPath (Join-Path $taskRuntime $_)) }).Count -gt 0) {
    & (Join-Path $PSScriptRoot 'prepare-desktop.ps1')
}
& $taskCompiler /nologo /target:winexe /platform:x64 /optimize+ /codepage:65001 /reference:System.Windows.Forms.dll /reference:System.Drawing.dll `
    "/reference:$taskRuntime/Microsoft.Web.WebView2.Core.dll" "/reference:$taskRuntime/Microsoft.Web.WebView2.WinForms.dll" `
    "/win32manifest:$taskProject/launcher/app.manifest" "/out:$taskOutput" $taskSource (Join-Path $taskProject 'launcher/DesktopWindow.cs')
if ($LASTEXITCODE -ne 0) { throw '启动器编译失败。' }
Write-Output "已生成桌面 App $taskProject/MonoPlanner.exe"
