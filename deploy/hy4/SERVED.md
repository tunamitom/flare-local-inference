# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg81` (`sha256:7d1a66a65b2e0669d5e4b3f7f2f8a789ba17694bbaf5c310305b3d1dbbc4a24b`); running container image: `local/flare:rt-h4-cp8-dg81`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg81-376e1366`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
