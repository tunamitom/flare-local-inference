# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg86t` (`sha256:1959bd3a4102cedbed14042f4807b3891e123b33f5ad1f356def89fdd02b5236`); running container image: `local/flare:rt-h4-cp8-dg86t`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg86t-0bc164ed`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
