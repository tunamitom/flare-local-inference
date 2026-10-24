#!/bin/bash
# FLARE container healthcheck (the flare image has no curl/wget — bash /dev/tcp
# is the only HTTP vehicle). Semantics match the house contract:
#   curl --fail --silent --max-time 2 http://localhost:PORT/health
# i.e. succeed only on HTTP 200 from /health. Mounted read-only at
# /healthcheck.sh by serve.sh. Port is substituted at launch time.
set -euo pipefail
PORT="@PORT@"
exec 3<>"/dev/tcp/127.0.0.1/${PORT}" || exit 1
printf 'GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n' >&3
read -r -t 2 line <&3 || exit 1
case "$line" in
  *" 200 "*) exit 0 ;;
  *) exit 1 ;;
esac