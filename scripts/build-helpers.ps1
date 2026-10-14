$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
# 使用系统编译器生成桌面辅助程序。
$taskCompiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'

# 文件选择组件。
& $taskCompiler /nologo /target:exe /platform:x64 /optimize+ /codepage:65001 /reference:System.Windows.Forms.dll "/out:$(Join-Path $taskProject 'launcher\runtime\MonoPlanner.FilePicker.exe')" (Join-Path $taskProject 'launcher\FilePicker.cs')
if ($LASTEXITCODE -ne 0) {
    throw '文件选择组件编译失败。'
}

# 应用查询与激活组件。
& $taskCompiler /nologo /target:exe /platform:x64 /optimize+ /codepage:65001 /reference:System.Web.Extensions.dll "/out:$(Join-Path $taskProject 'launcher\runtime\MonoPlanner.Applications.exe')" (Join-Path $taskProject 'launcher\Applications.cs')
if ($LASTEXITCODE -ne 0) {
    throw '应用查询组件编译失败。'
}
