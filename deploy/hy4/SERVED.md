# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88t` (`sha256:a576698d392552711d628b54fbc5fafa2231795933f8681b4697849a61871fe0`); running container image: `local/flare:rt-h4-cp8-dg88t`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88s-37a27f4b`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
