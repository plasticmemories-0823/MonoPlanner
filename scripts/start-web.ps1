$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath "$taskProject/web"
& pnpm dev --hostname 127.0.0.1 --port 3000
exit $LASTEXITCODE
