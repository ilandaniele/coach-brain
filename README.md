# coach-brain

Asistente conversacional tipo "coach" basado en material real del coach (videos, audios, PDFs, WhatsApp). Pipeline RAG sobre principios + situaciones + estilo, con vision (lectura de screenshots de chats).

## Stack

- **LLM**: Claude Sonnet 4.6 (respuesta) + Haiku 4.5 (parsing/OCR de screenshots)
- **Transcripción**: faster-whisper large-v3 local (GPU)
- **Embeddings**: bge-m3 local (GPU)
- **Reranking**: bge-reranker-v2-m3 local
- **Vector DB**: Qdrant (modo local embebido o Docker)
- **UI**: Streamlit (MVP)
- **Gestor**: uv (Python 3.11+)

## Setup

```powershell
# 1. Crear venv e instalar deps
uv sync

# 2. Copiar env y configurar API key
copy .env.example .env
# editar .env y poner ANTHROPIC_API_KEY

# 3. (Opcional) levantar Qdrant en Docker
docker compose up -d
```

## Pipeline

```
[Drive público]
   ↓ (1) descarga: scripts/download_drive.py
[data/raw/]
   ↓ (2) ingesta:
   ├─ pdfs/      → coach_brain.ingest.pdfs       → JSON
   ├─ videos/    → coach_brain.ingest.transcribe → JSON (Whisper GPU)
   └─ audios/    → coach_brain.ingest.transcribe → JSON
[data/processed/transcripts/]
   ↓ (3) extracción LLM: coach_brain.kb.extract
[data/processed/extracted/]
   ↓ (4) curación manual (opcional)
[data/curated/]
   ↓ (5) indexado: coach_brain.kb.index
[Qdrant local — 3 colecciones: principles, situations, style]
   ↓ (6) consulta:
[Streamlit UI con vision]
```

## Comandos

```powershell
# Ingestar PDFs
uv run python -m coach_brain.ingest.pdfs

# Transcribir todo video/audio en data/raw/
uv run python -m coach_brain.ingest.transcribe

# Extraer principios/situaciones de transcripts
uv run python -m coach_brain.kb.extract

# Indexar en Qdrant
uv run python -m coach_brain.kb.index

# Lanzar app
uv run streamlit run src/coach_brain/app/streamlit_app.py
```

## Estructura

```
coach-brain/
├── data/
│   ├── raw/          # material crudo bajado de Drive
│   ├── processed/    # transcripts y extractions
│   └── curated/      # versión final revisada a mano
├── src/coach_brain/
│   ├── ingest/       # PDFs, transcripción, descarga Drive
│   ├── kb/           # extracción LLM, embeddings, indexado
│   └── app/          # Streamlit UI
├── prompts/          # system + extractor prompts (Markdown)
├── scripts/          # entrypoints CLI
└── tests/
```

## GPU

Requiere NVIDIA con CUDA 12+ y >=8GB VRAM. Whisper large-v3 usa ~5GB, bge-m3 ~2GB. En el equipo de desarrollo (RTX 5070 Ti Laptop 12GB, CUDA 13.1) corre cómodo en paralelo.
