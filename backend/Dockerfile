FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# Tesseract OCR with English, Filipino and orientation detection, for photos and scans
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-fil tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

# PyTorch for the reranker. CUDA build by default so it can use an NVIDIA GPU (see
# docker-compose.gpu.yml); CI builds with TORCH_INDEX=.../whl/cpu to stay small.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install torch --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
