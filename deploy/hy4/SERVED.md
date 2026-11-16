# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg48` (`sha256:ff5c88204572d2333fbdc1f46837221c10d107bdf8b366a3c240100ab782f8b6`); running container image: `local/flare:rt-h4-cp8-dg48`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg48-cd703955`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
