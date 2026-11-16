# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg52` (`sha256:72e77fb3b068251cbe82fd7004563506480ef81cf5141f74e3c9c122487c1e42`); running container image: `local/flare:rt-h4-cp8-dg52`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg52-cd703955`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
