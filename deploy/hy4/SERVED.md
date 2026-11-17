# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg83` (`sha256:6182fb6b00249ea0153baa9175288d906453ceeb3eb6f84b153e81e6d2253d17`); running container image: `local/flare:rt-h4-cp8-dg83`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg83-011525d1`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
