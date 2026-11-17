# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg67` (`sha256:6942931089f8b36d262dfc5685c162802f1be205b8cf2ed2f2d6c5035534060d`); running container image: `local/flare:rt-h4-cp8-dg67`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg67-6d9459ad`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
