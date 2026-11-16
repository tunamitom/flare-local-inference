FROM nvidia/cuda:13.1.1-devel-ubuntu24.04
# h4-flare: the Hy4 serve image — the flare venv + BOTH Hy4 worktrees baked at
# their host absolute paths (the campaign runtime PYTHONPATH), plus the serve
# runner. Local only; never pushed (owner rule).
#   /home/tom/build/hy4-flare-wt   ignition recipes (hy4 branch, CP body)
#   /home/tom/build/hy4-fi         b12x flashinfer experimental (record writer)
#   /opt/flare-local-inference/.venv + the uv CPython 3.12.13 it symlinks to
COPY uvpython-root /home/tom/.local/share/uv/python
COPY venv /opt/flare-local-inference/.venv
COPY hy4-flare-wt /home/tom/build/hy4-flare-wt
COPY hy4-fi /home/tom/build/hy4-fi
COPY h4_serve.py /opt/h4/h4_serve.py
COPY healthcheck-src.sh /opt/h4/healthcheck-src.sh
ENV PYTHONPATH=/home/tom/build/hy4-flare-wt:/home/tom/build/hy4-fi/flashinfer/experimental \
    GIT_NO_LAZY_FETCH=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HOME=/state \
    XDG_CACHE_HOME=/state/.cache \
    TORCHINDUCTOR_CACHE_DIR=/state/.cache/torchinductor
WORKDIR /opt/h4
# Clear the NVIDIA base entrypoint: the launcher passes the full command.
ENTRYPOINT []
# No ENTRYPOINT: the launcher passes the full command (python -m torch.distributed.run ...).
CMD ["/opt/flare-local-inference/.venv/bin/python", "-m", "torch.distributed.run", "--nproc_per_node=8", "--master_port=29855", "/opt/h4/h4_serve.py"]

# Runtime build deps for the b12x native loader (io_uring bounce executor):
# "io_uring bounce support is unavailable: install liburing development
# headers and pkg-config" (first boot 2026-11-15). Appended as a top layer so
# the large COPY layers stay cached.
RUN apt-get update && apt-get install -y --no-install-recommends \
        liburing-dev liburing2 pkg-config \
    && rm -rf /var/lib/apt/lists/*
