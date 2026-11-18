# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88u` (`sha256:eade984a9901d927b3ed1c4a0693ed850202e53a0e5ed7b06f4a1ecc2b869290`); running container image: `local/flare:rt-h4-cp8-dg88u`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88t-3a7683b6`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
