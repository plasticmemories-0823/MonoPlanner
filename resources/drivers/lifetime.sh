#!/bin/sh
# 由容器自身检查只读续租文件；宿主与内核同时消失也不依赖 Docker API 收尾。
set -eu
token=$1
shift
main=$$
"$@" &
child=$!
(
    last=''
    stale=0
    while kill -0 "$child" 2>/dev/null; do
        value=''
        if ! IFS= read -r value < /monoplanner-lifetime/heartbeat; then
            kill -KILL "$main"
            exit 78
        fi
        case "$value" in
            "$token "[0-9]*) ;;
            *) kill -KILL "$main"; exit 78 ;;
        esac
        if [ "$value" = "$last" ]; then stale=$((stale + 1)); else stale=0; fi
        last=$value
        if [ "$stale" -ge 15 ]; then
            kill -KILL "$main"
            exit 78
        fi
        sleep 1
    done
) &
monitor=$!
status=0
wait "$child" || status=$?
kill "$monitor" 2>/dev/null || true
exit "$status"
