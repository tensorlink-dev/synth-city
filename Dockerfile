# synth-city + open-synth-miner (GPU-capable)
#
# Build:
#   docker build -t synth-city .
#
# Run the bridge:
#   docker run --gpus all --env-file .env -p 8377:8377 synth-city bridge
#
# Run the pipeline:
#   docker run --gpus all --env-file .env synth-city pipeline
#
# GPU support requires nvidia-container-toolkit on the host.
# The CUDA 12.2 runtime base image provides the CUDA libraries;
# PyTorch is installed with CUDA 12.1 wheels (forward-compatible).

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv git curl && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip so all subsequent installs work reliably
RUN pip install --no-cache-dir --upgrade pip

# ---- install uv for fast dependency resolution --------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# ---- install open-synth-miner first (changes less often) -------------------
ARG OSM_REPO=https://github.com/tensorlink-dev/open-synth-miner.git
ARG OSM_BRANCH=main

RUN git clone --depth 1 --branch "$OSM_BRANCH" "$OSM_REPO" /app/open-synth-miner

# ---- install PyTorch with CUDA 12.1 BEFORE other deps -----------------------
# Must come before requirements.txt / open-synth-miner so pip doesn't pull in
# CPU-only torch.  CUDA 12.1 wheels are forward-compatible with the 12.2 runtime.
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install open-synth-miner as a package (exposes the 'osa' namespace)
RUN uv pip install -e /app/open-synth-miner

# ---- install synth-city deps (cached layer) ---------------------------------
COPY pyproject.toml requirements.txt ./
RUN uv pip install -r requirements.txt

# ---- copy synth-city source -------------------------------------------------
COPY . .
RUN uv pip install -e .

# ---- verify the research environment works at build time ---------------------
RUN python -c "from osa.research.agent_api import ResearchSession; print('OK: ResearchSession importable')"
RUN python -c "import torch; print(f'torch {torch.__version__}, CUDA build: {torch.version.cuda}')"

# ---- runtime ----------------------------------------------------------------
# Bridge listens on 0.0.0.0 inside the container so Docker port mapping works
ENV BRIDGE_HOST=0.0.0.0
ENV BRIDGE_PORT=8377

EXPOSE 8377

ENTRYPOINT ["python", "main.py"]
CMD ["bridge"]
