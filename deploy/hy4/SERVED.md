# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg69` (`sha256:514ffcdc84fb89243bf30808c84c8391215e2633e06f981dd07caba4245e9c23`); running container image: `local/flare:rt-h4-cp8-dg69`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg69-6d9459ad`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
