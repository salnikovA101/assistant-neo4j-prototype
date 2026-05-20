FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

COPY server/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy && \
    pip install -r requirements.txt

COPY server/ ./server/

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "server.core.app:app", "--host", "0.0.0.0", "--port", "8000"]
