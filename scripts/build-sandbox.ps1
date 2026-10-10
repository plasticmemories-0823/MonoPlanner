$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskCompiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
$taskOutput = Join-Path $taskProject 'launcher\runtime\MonoPlanner.Sandbox.exe'
$taskSource = Join-Path $taskProject 'launcher\Sandbox.cs'
& $taskCompiler /nologo /target:exe /platform:x64 /optimize+ /codepage:65001 /reference:System.Web.Extensions.dll "/out:$taskOutput" $taskSource
if ($LASTEXITCODE -ne 0) { throw '原生沙盒编译失败。' }
Copy-Item -LiteralPath (Join-Path $taskProject 'launcher/Sandbox.exe.config') -Destination ($taskOutput + '.config')
& $taskCompiler /nologo /target:exe /platform:x64 /optimize+ /codepage:65001 /reference:System.Windows.Forms.dll "/out:$(Join-Path $taskProject 'launcher\runtime\MonoPlanner.FilePicker.exe')" (Join-Path $taskProject 'launcher\FilePicker.cs')
if ($LASTEXITCODE -ne 0) { throw '文件选择组件编译失败。' }
& $taskCompiler /nologo /target:exe /platform:x64 /optimize+ /codepage:65001 /reference:System.Web.Extensions.dll "/out:$(Join-Path $taskProject 'launcher\runtime\MonoPlanner.Applications.exe')" (Join-Path $taskProject 'launcher\Applications.cs')
if ($LASTEXITCODE -ne 0) { throw '应用查询组件编译失败。' }
