# syntax=docker/dockerfile:1
# ChatBotMM — fully local Persian/English RAG chatbot.
#
# Build:  docker build -t chatbotmm .
# Run:    docker run -it -v chatbot-data:/data chatbotmm
#
# Put your documents in the volume's docs/ folder, e.g.:
#   docker run --rm -v chatbot-data:/data -v ./mydocs:/src alpine cp -r /src/. /data/docs/
# Models (~2 GB) download on first run and persist in the same volume.

# ---- builder: install everything into a self-contained venv -----------------
FROM python:3.12-slim AS builder

# Build tools are only needed here, in case a dependency has no matching wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src

# CPU-only torch (the default Linux wheel drags in ~4 GB of CUDA libraries),
# then the package itself, then the quantized llama.cpp backend.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir llama-cpp-python \
         --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# ---- runtime: slim image with just the venv ---------------------------------
FROM python:3.12-slim

# OpenMP runtime, required by faiss and llama.cpp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# All persistent state (documents, index, model cache) lives under /data so a
# single volume mount keeps everything across container restarts.
ENV PATH="/opt/venv/bin:$PATH" \
    CHATBOT_DATA_DIR=/data \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
WORKDIR /data

ENTRYPOINT ["chatbot"]
# Dependencies are baked into the image; skip the self-install step.
CMD ["cli", "--skip-install"]
