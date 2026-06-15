FROM --platform=linux/amd64 python:3.12-slim

# Sistema: OpenMP (requerido por onnxruntime/torch CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch — en Linux el wheel de PyPI es CUDA por defecto, forzar CPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Resto de deps
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Pre-bake bge-m3 en la imagen (~570MB, layer cacheado)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3', device='cpu'); \
print('bge-m3 OK')"

# Pre-warm RapidOCR (inicializa sesión ONNX, evita delay en primer request)
RUN python -c "\
from rapidocr_onnxruntime import RapidOCR; \
RapidOCR(); \
print('RapidOCR OK')"

# Código fuente
COPY src/ src/
COPY prompts/ prompts/
COPY pyproject.toml .

ENV PYTHONPATH=/app/src

# Usuario no-root + directorios
RUN useradd -m -u 1000 appuser \
    && mkdir -p /data /home/appuser/.streamlit \
    && printf '[general]\nemail = ""\n' > /home/appuser/.streamlit/credentials.toml \
    && chown -R appuser:appuser /app /data /home/appuser

USER appuser

VOLUME /data

# Cloud defaults (sobreescribibles via fly secrets)
ENV DATA_DIR=/data
ENV PROMPTS_DIR=/app/prompts
ENV EMBED_DEVICE=cpu
ENV COACH_RESPONSE_BACKEND=anthropic
ENV OLLAMA_ENABLED=false
ENV QDRANT_PATH=""

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", \
     "src/coach_brain/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
