# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg79` (`sha256:8981769bf9fec5a1691188f7664daee4ea04047b7eb394298a30ed613ad48edb`); running container image: `local/flare:rt-h4-cp8-dg79`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg79-aca6b411`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
