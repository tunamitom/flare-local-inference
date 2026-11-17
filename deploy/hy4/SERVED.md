# Hy4 served state (written by deploy/hy4/sync.sh)

- Launcher pin: `local/flare:rt-h4-cp8-dg79` (`sha256:69f88fcdbcc98ba558cfc1041c13c782eb080ee31b2e3a2eb5b40e6a7d3b3220`); running container image: `local/flare:rt-h4-cp8-dg79`
- Production launcher snapshot: `start-h4.sh.c18-prod-dg79-aca6b411`
- flashinfer/b12x (branch hy4 of tunamitom/flashinfer) baked into the image: `b8a4600a1`
- Files here are byte copies of the running container's /opt/h4/h4_serve.py, /opt/h4/start-h4.sh,
  ~/.h4-image-build/Dockerfile and /opt/h4/healthcheck-src.sh.
