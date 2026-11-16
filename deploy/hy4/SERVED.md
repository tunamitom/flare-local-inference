# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg5` (`sha256:79af23f224f36cfce9a74ab9a9fdad1f5d431dea7fe0024ebf88575b97b54309`); running container image: `local/flare:rt-h4-cp8-dg5`
- Production launcher snapshot: `start-h4.sh.c18-prod-p2min-52ed8238`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b47ad5580`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
