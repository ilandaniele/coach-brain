# coach-brain

Asistente de coaching conversacional con RAG sobre material propio. Le pasás un
screenshot de un chat (o el texto pegado) y devuelve un diagnóstico de la
dinámica más líneas concretas para responder, fundamentadas en un corpus de
material de coaching indexado localmente.

No es un generador de frases genéricas: cada respuesta se apoya en principios,
situaciones y frases recuperadas por similitud semántica del material propio.

## Cómo funciona

```
screenshot ──► OCR local (RapidOCR) ──┐
                                      ├──► embedding (bge-m3) ──► Qdrant (39K vectores)
texto pegado ─────────────────────────┘                                   │
                                                                          ▼
                              respuesta ◄── LLM ◄── prompt + principios/situaciones/frases
```

- **Retrieval** — 3 colecciones (principios, situaciones, frases de estilo)
  consultadas con un umbral de score para no arrastrar contexto irrelevante.
- **Memoria por contacto** — SQLite; el perfil de cada persona se actualiza
  después de cada análisis y se reinyecta en los siguientes.
- **OCR local** — gratis y sin API: la posición de la burbuja decide quién habla.

## Stack

Streamlit · Qdrant (embebido) · sentence-transformers (bge-m3) · RapidOCR ·
SQLite · Docker · Fly.io

Los modelos de respuesta son intercambiables: cualquier proveedor compatible con
la API de OpenAI (Gemini, Groq, Cerebras, OpenRouter), Anthropic, u Ollama local.
Un solo code path los cubre — ver [`src/coach_brain/llm.py`](src/coach_brain/llm.py).

## Correr local

```bash
uv sync
cp .env.example .env      # cargar al menos una API key
uv run streamlit run src/coach_brain/app/streamlit_app.py
```

Con `OLLAMA_ENABLED=true` y Ollama corriendo, funciona sin ninguna API paga.

## Desplegar

```bash
flyctl secrets set GEMINI_API_KEY=... APP_PASSWORD=...
flyctl deploy
```

El índice vectorial va horneado en la imagen; el volumen persistente solo guarda
la base de contactos y las notas personales.

## Notas

- El material de coaching indexado, los chats y la base de contactos **no** están
  en el repo: son datos personales y están en `.gitignore`.
- El acceso está protegido por PIN con rate limiting por IP y backoff exponencial.
