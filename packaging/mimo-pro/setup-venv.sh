#!/usr/bin/env bash
# Create the export environment: the locked dependencies, then the b12x patches the MiMo recipe needs.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Copy, never hardlink, from uv's cache: the patches edit the installed files.
UV_LINK_MODE=copy uv sync --frozen --extra prepare
B12X=.venv/lib/python3.12/site-packages/flashinfer/experimental
for p in patches/b12x-*.patch; do
  patch --forward -p1 -d "$B12X" < "$p"
done
