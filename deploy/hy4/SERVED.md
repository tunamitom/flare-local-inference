# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg78` (`sha256:702ad9c1eb0332cd178405e388585aae8c59b2a730e9e57d0c9458a6bf20d65f`); running container image: `local/flare:rt-h4-cp8-dg78`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg78-6d9459ad`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
