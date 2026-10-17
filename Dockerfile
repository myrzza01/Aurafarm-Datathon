FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CASHWATCH_DEVICE=cpu \
    HOME=/tmp

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
# Install CPU wheels explicitly; the standard Linux index can pull CUDA libraries.
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check

RUN groupadd --gid 10001 forte \
    && useradd --uid 10001 --gid forte --no-create-home forte \
    && mkdir -p /app/data \
    && chown forte:forte /app/data
COPY backend/ backend/
COPY frontend/ frontend/
COPY ml/cashwatch/ ml/cashwatch/
COPY ml/weights/ ml/weights/
COPY ml/SOURCE.json ml/THIRD_PARTY_NOTICES.md ml/

USER 10001:10001
EXPOSE 8000
# Only the private Compose network can reach this port; Caddy supplies proxy headers.
CMD ["uvicorn", "backend.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*", "--no-access-log"]
