# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg16` (`sha256:d87237f7fb6a4b25df31ba1c68f60a98fdfe907a2272859df9466ba56ab7dd11`); running container image: `local/flare:rt-h4-cp8-dg16`
- Production launcher snapshot: `start-h4.sh.c18-prod-p2min-52ed8238`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b47ad5580`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
