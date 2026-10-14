# 一起启动经过认证的本机前端与后端。
& (Join-Path $PSScriptRoot 'start-app.ps1') -NoBrowser @args
exit $LASTEXITCODE
