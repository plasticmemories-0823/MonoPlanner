$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskProject
$env:PYTHONUTF8 = '1'
$env:LOG_COLOR = 'false'
$env:LANGSMITH_TRACING = 'false'
& "$taskProject/.venv-web/Scripts/langgraph.exe" dev --host 127.0.0.1 --port 2024 --no-browser --no-reload
exit $LASTEXITCODE
