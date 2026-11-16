#!/bin/bash
# h4 container health check (rendered with @PORT@ by start-h4.sh; runs inside
# the container via docker --health-cmd).
/opt/flare-local-inference/.venv/bin/python - <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:@PORT@/health", timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
