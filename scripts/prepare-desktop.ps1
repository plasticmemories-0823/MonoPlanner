param([string]$PackagePath)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskVersion = '1.0.4258.31'
$taskHash = '56F7F4B8BF9AEE4B8EFEFBBDD4F67D5F74EBD1B100ED0806DA71BF76AF481AA9'
$taskRuntime = Join-Path $taskRoot 'launcher/runtime'
$taskOwnedPackage = -not $PackagePath
if ($taskOwnedPackage) {
    $PackagePath = Join-Path ([IO.Path]::GetTempPath()) ("monoplanner-webview2-$([guid]::NewGuid()).nupkg")
    Invoke-WebRequest "https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/$taskVersion/microsoft.web.webview2.$taskVersion.nupkg" -OutFile $PackagePath -TimeoutSec 90
}
try {
    if ((Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash -ne $taskHash) { throw 'WebView2 SDK 校验失败。' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $taskZip = [IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($PackagePath))
    try {
        New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
        foreach ($taskEntry in @('lib/net462/Microsoft.Web.WebView2.Core.dll',
                                 'lib/net462/Microsoft.Web.WebView2.WinForms.dll',
                                 'runtimes/win-x64/native/WebView2Loader.dll', 'LICENSE.txt')) {
            $taskFile = $taskZip.GetEntry($taskEntry)
            if (-not $taskFile) { throw "SDK 缺少文件：$taskEntry" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($taskFile, (Join-Path $taskRuntime ([IO.Path]::GetFileName($taskEntry))), $true)
        }
        Set-Content -LiteralPath (Join-Path $taskRuntime 'version.txt') -Value $taskVersion -Encoding ASCII
    } finally { $taskZip.Dispose() }
} finally {
    if ($taskOwnedPackage -and (Test-Path -LiteralPath $PackagePath)) { Remove-Item -LiteralPath $PackagePath }
}
Write-Output "已准备桌面 SDK $taskVersion（仅项目内依赖）。"
