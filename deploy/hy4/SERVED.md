# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88p` (`sha256:d446bafbaaab9afdb7f35cea6eef0c289eff6660fb1af86cd5d89944498920fb`); running container image: `local/flare:rt-h4-cp8-dg88p`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88o-d9d43f55`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
