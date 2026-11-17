# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg85b` (`sha256:e3617406bc8251ecd6acdf834df28eabfe7c2448be5a75f20e1ff9052319bdf7`); running container image: `local/flare:rt-h4-cp8-dg85b`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg85b-0bc164ed`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
