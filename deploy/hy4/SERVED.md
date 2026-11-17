# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg86u` (`sha256:96257f2d8ce6282af3dd10d0b0375e05f4ff59bf2ac17fe7cf1ad7b31f191fe7`); running container image: `local/flare:rt-h4-cp8-dg86u`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg86u-0bc164ed`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
