# Digest pinneado: el tag :3.12-slim se actualiza upstream e invalida todo el cache
FROM --platform=linux/amd64 python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch (en Linux PyPI el default es CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Deps: opencv-python-headless primero para evitar opencv-python con X11
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt && \
    pip install --no-cache-dir --no-deps rapidocr-onnxruntime

# Pre-bake bge-m3 (~570MB, layer cacheado)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3', device='cpu'); \
print('bge-m3 OK')"

# Pre-warm RapidOCR (inicializa sesión ONNX en build)
RUN python -c "\
from rapidocr_onnxruntime import RapidOCR; \
RapidOCR(); \
print('RapidOCR OK')"

# App source
COPY src/ src/
COPY prompts/ prompts/
COPY pyproject.toml .

# Qdrant storage embebido con todos los vectores del coach
COPY qdrant_storage/ qdrant_storage/

ENV PYTHONPATH=/app/src
ENV QDRANT_PATH=/app/qdrant_storage

# Non-root user
RUN useradd -m -u 1000 appuser \
    && mkdir -p /data /home/appuser/.streamlit \
    && printf '[general]\nemail = ""\n' > /home/appuser/.streamlit/credentials.toml \
    && chown -R appuser:appuser /app /data /home/appuser

USER appuser

VOLUME /data

ENV DATA_DIR=/data
ENV PROMPTS_DIR=/app/prompts
ENV EMBED_DEVICE=cpu
ENV COACH_RESPONSE_BACKEND=anthropic
ENV OLLAMA_ENABLED=false

EXPOSE 8501

# fileWatcherType=none: en prod no hay hot-reload, y el watcher recorre todos los
# módulos en cada rerun (spamea ModuleNotFoundError: torchvision y quema CPU).
CMD ["python", "-m", "streamlit", "run", \
     "src/coach_brain/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
