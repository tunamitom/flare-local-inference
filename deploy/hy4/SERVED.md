# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-96a51310` (`sha256:46a53139e1b9e8a6376c0fcffc71879a3526eaf64ad2527b718923760032ab06`); running container image: `local/flare:rt-h4-cp8-96a51310`
- Production launcher snapshot: `start-h4.sh.c18-prod-p2sfdec-96a51310`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b47ad5580`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
