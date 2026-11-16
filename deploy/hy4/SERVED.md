# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg12` (`sha256:2597ba5fce9d9fd3cc27196050e07f564e8f93367effef93833cf6a020c29399`); running container image: `local/flare:rt-h4-cp8-dg12`
- Production launcher snapshot: `start-h4.sh.c18-prod-p2min-52ed8238`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b47ad5580`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
