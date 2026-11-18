# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88v` (`sha256:7cc894c5d8ff5c9a092c5231110cd13a3fd0df75801162e0bf7f8013c2b558b7`); running container image: `local/flare:rt-h4-cp8-dg88v`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88u-08a460f5`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
