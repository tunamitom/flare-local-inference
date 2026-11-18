# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88x` (`sha256:e8014d68daa6b189b712162f226f39ac10114c953927841dd948c7848908a66c`); running container image: `local/flare:rt-h4-cp8-dg88x`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88v-821416c1`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
