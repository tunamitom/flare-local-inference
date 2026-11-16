# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-37ede0e3` (`sha256:f4a77f8550790af91337d5dfff51ed7e73dca39cd5fa49a3a75e498377b3bf39`); running container image: `local/flare:rt-h4-cp8-37ede0e3`
- Production launcher snapshot: `start-h4.sh.c18-prod-p1fused-37ede0e3`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `3a8fccf1f`
- Files here are byte copies of /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
