#!/usr/bin/env bash
# deploy/hy4/sync.sh — copy the LIVE Hy4 serve files into this branch so `hy4`
# mirrors exactly what h4 serves (owner requirement): the serve runner, the
# production launcher, the image recipe and the healthcheck, plus SERVED.md
# (image tag/ID, launcher name, the b12x SHA baked into the image).
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
cp /opt/h4/h4_serve.py "$here/h4_serve.py"
cp /opt/h4/start-h4.sh "$here/start-h4.sh"
cp /home/tom/.h4-image-build/Dockerfile "$here/Dockerfile"
cp /opt/h4/healthcheck-src.sh "$here/healthcheck-src.sh"
img=$(sed -nE 's/^IMAGE="\$\{IMAGE:-([^}]*)\}"/\1/p' /opt/h4/start-h4.sh)
sha=$(sed -nE 's/^IMAGE_SHA="\$\{IMAGE_SHA:-([^}]*)\}"/\1/p' /opt/h4/start-h4.sh)
running=$(docker inspect h4 --format '{{.Config.Image}}' 2>/dev/null || echo none)
fi_sha=$(git -c safe.directory='*' -C /home/tom/build/hy4-fi rev-parse --short=9 HEAD)
snap=$(ls -t /opt/h4/start-h4.sh.c18-prod-* 2>/dev/null | head -1)
cat > "$here/SERVED.md" <<MD
# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: \`$img\` (\`$sha\`); running container image: \`$running\`
- Production launcher snapshot: \`$(basename "$snap")\`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: \`$fi_sha\`
- Files here are byte copies of /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
MD
echo "synced: $img ($running), b12x $fi_sha"
