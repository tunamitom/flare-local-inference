# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg88z` (`sha256:10e70121bc5e539bb8b9253cebe94b7cfba20fb5dc8263f4700ed881fdbbd604`); running container image: `local/flare:rt-h4-cp8-dg88z`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg88z-idle-fix`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
