#!/usr/bin/env bash
# Build the serving runtime image from this checkout.
# usage: build-image.sh [TAG]
set -euo pipefail
cd "$(dirname "$0")/../.."
commit=$(git rev-parse --short=8 HEAD)
exec docker build --label flare.commit="$commit" -t "${1:-local/flare:rt-mimo-pro-$commit}" .
