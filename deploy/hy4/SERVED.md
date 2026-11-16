# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-4602f878` (`sha256:a6ff30c156902516d591f7c1a1e698d0f11be00e4629a88f3c70a077acb95891`); running container image: `local/flare:rt-h4-cp8-4602f878`
- Production launcher snapshot: `start-h4.sh.c18-prod-e5-4602f878`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `3a8fccf1f`
- Files here are byte copies of /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
