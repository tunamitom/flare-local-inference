# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg80` (`sha256:15582ab3544dca134c0fba20ecfcdb0322ccb5ca36a1048fbf865f0080c97315`); running container image: `local/flare:rt-h4-cp8-dg80`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg80-0809c31a`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
