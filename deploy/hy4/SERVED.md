# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg86r` (`sha256:c387dd8b66416f34d2d29036ceed8698bbc2d08d31a76de4909cd75c95941d46`); running container image: `local/flare:rt-h4-cp8-dg86r`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg86r-0bc164ed`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
