# synth-city + open-synth-miner
#
# Build:
#   docker build -t synth-city .
#
# Run the bridge:
#   docker run --env-file .env -p 8377:8377 synth-city bridge
#
# Run the pipeline:
#   docker run --env-file .env synth-city pipeline
#
# With GPU (requires nvidia-container-toolkit):
#   docker run --gpus all --env-file .env -p 8377:8377 synth-city bridge

FROM python:3.11-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- install open-synth-miner first (changes less often) -------------------
ARG OSM_REPO=https://github.com/tensorlink-dev/open-synth-miner.git
ARG OSM_BRANCH=main

RUN git clone --depth 1 --branch "$OSM_BRANCH" "$OSM_REPO" /app/open-synth-miner

RUN pip install --no-cache-dir -e /app/open-synth-miner 2>/dev/null || \
    echo "open-synth-miner has no setup file — will use PYTHONPATH"

ENV PYTHONPATH="/app/open-synth-miner:${PYTHONPATH:-}"

# ---- install synth-city deps (cached layer) ---------------------------------
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- copy synth-city source -------------------------------------------------
COPY . .
RUN pip install --no-cache-dir -e .

# ---- runtime ----------------------------------------------------------------
# Bridge listens on 0.0.0.0 inside the container so Docker port mapping works
ENV BRIDGE_HOST=0.0.0.0
ENV BRIDGE_PORT=8377

EXPOSE 8377

ENTRYPOINT ["python", "main.py"]
CMD ["bridge"]
