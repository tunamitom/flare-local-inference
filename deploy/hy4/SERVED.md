# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg82a` (`sha256:1a494841e506bc157cf8cecd7aeefe6731566d227e5a23967391c76fc383b826`); running container image: `local/flare:rt-h4-cp8-dg82a`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg82a-1d7371e4`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
