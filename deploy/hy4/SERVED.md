# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88q` (`sha256:d02dc4ac50f7128b7bed04c7b5db7de6574c9662d49d3f5c9208d720ff81d12e`); running container image: `local/flare:rt-h4-cp8-dg88q`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88p-200cd1fc`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
