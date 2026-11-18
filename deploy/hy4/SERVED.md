# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88a` (`sha256:39763e05b58929985aaa6e122cd98f194eadefd51daec3641cb49aecb942c73b`); running container image: `local/flare:rt-h4-cp8-dg88a`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88a-0bc164ed`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
