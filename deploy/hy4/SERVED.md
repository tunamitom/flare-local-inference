# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88l` (`sha256:d8c9ef861a489071e7b1f37336983832169b9d784b659172cc6aa2ad8a20b226`); running container image: `local/flare:rt-h4-cp8-dg88l`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88l-0bc164ed`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
