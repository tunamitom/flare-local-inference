# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg62` (`sha256:24a5fbe58a72284fd02e709af86924961d4555808ca5ece8049f109f9eecd9d3`); running container image: `local/flare:rt-h4-cp8-dg62`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg62-cd703955`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
