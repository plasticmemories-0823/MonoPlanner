# Start the authenticated local frontend/backend pair together.
& (Join-Path $PSScriptRoot 'start-app.ps1') -NoBrowser @args
exit $LASTEXITCODE
