# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88s` (`sha256:6909b82fe623868b2d708cb0597e5747accad44245e58ec96b1762b2a827c3ad`); running container image: `local/flare:rt-h4-cp8-dg88s`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88r-ced439b0`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
