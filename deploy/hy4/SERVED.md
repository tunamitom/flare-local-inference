# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-52a93569` (`sha256:85a43cc1be6c8a6b81279e8bd40b891d581e2996ccf644e1f70d2ca12eb7adcd`); running container image: `local/flare:rt-h4-cp8-52a93569`
- Production launcher snapshot: `start-h4.sh.c18-prod-p1trim-52a93569`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b47ad5580`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
