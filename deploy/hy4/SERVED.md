# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg72` (`sha256:2c7c93ddb877898a1e4963b7a1fb1c552c2a3c9db654cea9faa94c85bc764e0b`); running container image: `local/flare:rt-h4-cp8-dg72`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg72-6d9459ad`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
