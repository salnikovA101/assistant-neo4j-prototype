# CPU image — compose default for service `reranker`.
#
# Later GPU without breaking this default:
#   1. Add reranker/Dockerfile.gpu (CUDA base, install torch from a CUDA index,
#      then the same requirements.txt).
#   2. Add a compose profile (e.g. rerank-gpu) that sets
#      dockerfile: reranker/Dockerfile.gpu and --device cuda.
#   3. Leave this file and profile `rerank` as the CPU path.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY reranker/requirements.txt ./requirements.txt
# Install CPU torch first, then pin it so later deps cannot replace it with a CUDA wheel.
# GPU image: same pattern, different --index-url (and --device cuda in CMD).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip freeze | grep -i '^torch==' > /tmp/torch.lock \
    && pip install -c /tmp/torch.lock -r requirements.txt

COPY reranker/serve.py ./serve.py

EXPOSE 7997

CMD ["python3", "serve.py", "--host", "0.0.0.0", "--port", "7997", "--device", "cpu"]
