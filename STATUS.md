# STATUS — coach-brain

> Documento de handoff. Si arrancás una sesión Claude nueva, leé esto primero antes de avanzar.

**Última actualización:** 2026-05-28
**Estado general:** Material recibido (~372GB en `C:\Users\Ilan\Desktop\Bonus`). Texto ingestado (78 transcripts). **Extracción LLM (Haiku) y transcripción de 155 videos corriendo en background.** Falta: index a Qdrant + MVP.

---

## Qué es esto

Asistente coach conversacional. Subís screenshot de un chat de WhatsApp → analiza el contexto → devuelve diagnóstico + 3 líneas de respuesta + nivel de riesgo. RAG sobre material del coach (60+ clases en video, audios, PDFs).

Diseño completo y razonamiento en el primer mensaje del chat anterior. Resumen en este repo: `README.md`.

---

## Estado del setup

| Componente | Estado | Notas |
|---|---|---|
| Python 3.12.13 (uv) | OK | `.python-version` fija 3.12 (uv defaultea a 3.14 que rompe) |
| Toolchain (uv, ffmpeg, git) | OK | winget |
| Docker Desktop | NO instalado | UAC fue cancelado. Qdrant en modo embebido. Si querés contenedor: `winget install Docker.DockerDesktop` y `docker compose up -d` |
| PyTorch + CUDA | OK | **`torch==2.12.0+cu130`** (Blackwell sm_120 de la RTX 5070 Ti requiere cu130, NO cu126) |
| Whisper local | OK (descarga al primer uso) | faster-whisper large-v3 |
| Embeddings bge-m3 | OK (cacheado) | dim 1024, en `~/.cache/huggingface/` |
| Qdrant | OK (modo local embebido) | persiste en `./qdrant_storage/` |
| Anthropic API key | OK | en `.env`, Sonnet 4.6 + Haiku 4.5 |
| Smoke test | Pasa | `.\.venv\Scripts\python.exe scripts\smoke_test.py --quick` |
| **Material del coach** | **PENDIENTE** | Drive con permiso "Restricted" — gdown necesita "Anyone with the link" |

---

## Estado actual del pipeline (2026-05-28)

Material en `C:\Users\Ilan\Desktop\Bonus` (~372GB). `data/raw/{videos,pdfs,docs}` son **junctions** a esa carpeta (no se copió nada). Ver gotcha #7.

| Etapa | Estado |
|---|---|
| Ingest texto (62 PDF + 15 docx + 1 epub = 78 transcripts) | ✅ Hecho. 1 PDF escaneado sin OCR quedó afuera. |
| Extracción LLM (Haiku 4.5) | 🔄 Corriendo en background → `data/processed/extracted/`. Log: `logs/extract.log`. Calidad verificada OK. |
| Transcripción 155 videos (Whisper GPU) | 🔄 Corriendo **de a batches** → `scripts/transcribe_batches.ps1` (proceso fresco por batch de 8, libera VRAM entre batches para no trabar la máquina). Log: `logs/transcribe.log`. Resumable. Muchas horas. |
| Index a Qdrant | 🔄 Hecho parcial (15 docx → 579 principios / 309 situaciones / 410 frases). Re-index full con `--reset` cuando termine extracción. |
| MVP Streamlit | ✅ Levantado en http://localhost:8501 (EMBED_DEVICE=cpu mientras Whisper usa GPU). |
| **WhatsApp coaching (Ema↔Ilan, 3 años)** | ✅ Texto parseado: **38.029 mensajes** → `transcripts/whatsapp_ema_coaching.json`. Ingestor: `ingest/whatsapp.py` (CLI `ingest-whatsapp`). Material más valioso: coaching 1-a-1 real. |
| WhatsApp voz (8.399 .opus) | ⬜ Pendiente — transcribir en batches después de los 155 videos (contienen el diagnóstico hablado de la coach). |
| WhatsApp imágenes (8.260 .jpg) | ⬜ Pendiente/diferido — screenshots de chats reales; vision con Haiku (costo ~$8-25). |

### Próximos pasos (cuando terminen los background jobs)
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:PYTHONIOENCODING = "utf-8"
cd C:\Users\Ilan\Desktop\Proyectos\coach-brain
# 1. Re-correr extract para los videos ya transcriptos (saltea lo hecho)
.\.venv\Scripts\python.exe -m coach_brain.kb.extract
# 2. Indexar todo a Qdrant (bge-m3 en GPU — si Whisper sigue corriendo, ojo VRAM: bajar EMBED_DEVICE=cpu en .env)
.\.venv\Scripts\python.exe -m coach_brain.kb.index
# 3. MVP
.\.venv\Scripts\python.exe -m streamlit run src\coach_brain\app\streamlit_app.py
```

### Cómo chequear progreso de los background jobs
```powershell
# transcripts de video hechos / total 155
(Get-ChildItem data\processed\transcripts\video_*.json).Count
# extracciones hechas / total (78 texto + los videos que vayan saliendo)
(Get-ChildItem data\processed\extracted\*.json).Count
```

---

## Pipeline a correr (en orden, después de tener material)

Todos los comandos desde `C:\Users\Ilan\Desktop\Proyectos\coach-brain`. Refrescar PATH primero si es shell nuevo:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:PYTHONIOENCODING = "utf-8"
```

### 1. Ingestar PDFs (rápido, ~segundos por PDF)
```powershell
.\.venv\Scripts\python.exe -m coach_brain.ingest.pdfs
```
Salida: `data/processed/transcripts/pdf_*.json`

### 2. Transcribir videos + audios (largo, ~minutos por clase con GPU)
**De a batches (recomendado — no traba la máquina):**
```powershell
powershell -NoProfile -File scripts\transcribe_batches.ps1 -BatchSize 8
```
Corre un proceso fresco por batch (libera VRAM entre tandas), resumable. O todo de una:
```powershell
.\.venv\Scripts\python.exe -m coach_brain.ingest.transcribe          # las 155
.\.venv\Scripts\python.exe -m coach_brain.ingest.transcribe --limit 8 # solo 8 nuevas
```
Salida: `data/processed/transcripts/video_*.json`, `audio_*.json`

### 3. Extracción LLM (principios + situaciones + frases)
```powershell
.\.venv\Scripts\python.exe -m coach_brain.kb.extract
```
Usa Claude Sonnet 4.6. Costo: ~$0.50–2 para todo el material. Salida: `data/processed/extracted/*.extracted.json`

### 4. Indexar en Qdrant
```powershell
.\.venv\Scripts\python.exe -m coach_brain.kb.index
```
Crea 3 colecciones: `principles`, `situations`, `style_phrases`.

### 5. Lanzar MVP
```powershell
.\.venv\Scripts\python.exe -m streamlit run src\coach_brain\app\streamlit_app.py
```
Abre en http://localhost:8501. Probar con un screenshot real.

---

## Gotchas conocidos (importante, no obvios)

### 1. `uv sync` revierte torch a CPU
El lockfile no respeta el index custom de PyTorch CUDA. Después de cualquier `uv sync`, correr:
```powershell
uv pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu130
```

### 2. `UV_LINK_MODE=copy` debe estar seteado
Persistido en User env vars de Windows. Sin esto, uv intenta symlinks que requieren Developer Mode o admin. Si una sesión nueva no lo tiene, setear con:
```powershell
[System.Environment]::SetEnvironmentVariable("UV_LINK_MODE","copy","User")
```

### 3. `ANTHROPIC_API_KEY` vacío sobrescribe el `.env`
El proceso Python recibe `ANTHROPIC_API_KEY=""` (string vacío, fuente desconocida). Pydantic-settings le da prioridad sobre `.env`. Ya está parcheado en `src/coach_brain/settings.py` con `_purge_empty_envs()` — no tocar esa función.

### 4. PowerShell tool de Claude no persiste CWD ni env vars
Cada comando empieza con el path por defecto. Siempre absoluto y refrescar PATH al inicio.

### 5. Encoding cp1252 en Windows
Console default de Windows rompe con caracteres Unicode (✓, ✗, emojis en código). Setear `$env:PYTHONIOENCODING = "utf-8"` antes de correr scripts que renderizan texto.

### 6. Drive: gdown no acepta carpetas "Restricted"
Devuelve HTTP 401. Permiso necesario: "Anyone with the link". (Ya no aplica — material bajado manual.)

### 7. El material son junctions, no copias
`data/raw/{videos,pdfs,docs}` apuntan a `C:\Users\Ilan\Desktop\Bonus` (372GB, no se copió). Si se rompen, recrear:
```powershell
cmd /c mklink /J "C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\videos" "C:\Users\Ilan\Desktop\Bonus"
```
(idem para `pdfs` y `docs`). `find` de MSYS NO atraviesa junctions; pathlib de Python SÍ (que es lo que usan los ingestores).

### 8. Whisper: HF_HUB_DISABLE_SYMLINKS=1
Seteado a nivel User. Sin esto, la 1ra descarga del modelo falla con `WinError 1314` (symlinks del cache de HF sin Developer Mode).

### 9. CUDA Blackwell (sm_120) + faster-whisper: cublas en el dir de ctranslate2
torch usa CUDA 13, CTranslate2 necesita CUDA 12. Hay que tener `cublas64_12.dll` + `cublasLt64_12.dll` copiadas dentro de `.venv\Lib\site-packages\ctranslate2\` (CTranslate2 las carga por nombre ignorando `add_dll_directory`). `transcribe._register_cuda_dlls()` lo auto-cura. NUNCA copiar `cudnn64_9.dll` ahí (la pone torch; la versión cu12 rompe el import de torch con `WinError 127`). Probado: CTranslate2 corre en sm_120 con esto.

---

### 10. Orden de import: sentence_transformers ANTES que qdrant_client
En Windows con `torch 2.12.0+cu130`, importar `qdrant_client(.models)` **antes** que `sentence_transformers`/torch produce un segfault silencioso (access violation `0xC0000005`, exit `-1073741819`, sin traceback) por conflicto de libs nativas. `kb/index.py` fuerza `import sentence_transformers` arriba de todo (no reordenar ese import). `retrieve.py` y `streamlit_app.py` ya importan `embed` antes que `index`, así que están a salvo. Si creás un módulo nuevo que use ambos, importá embeddings primero.

### 11. Qdrant embebido toma lock de proceso único
El modo local (`QDRANT_PATH`) bloquea la carpeta para UN solo proceso. No se puede indexar mientras Streamlit está levantado. Para re-indexar: parar Streamlit → `index --reset` → relanzar Streamlit. La extracción (solo escribe JSON) no toca Qdrant, corre en paralelo sin problema.

### 12. Re-indexar SIEMPRE con --reset
`index.py` usa UUIDs random como point IDs, así que correr `index` sin `--reset` DUPLICA todo. Siempre `python -m coach_brain.kb.index --reset`.

## Costos estimados del pipeline real

| Etapa | Costo aprox |
|---|---|
| Transcripción (60 clases, ~8h audio total) | $0 (Whisper local) |
| Extracción LLM (Claude Sonnet 4.6) | $0.50–2 |
| Embeddings (bge-m3 local) | $0 |
| Por consulta del MVP | ~$0.005–0.02 |

---

## Estructura del proyecto

```
coach-brain/
├── STATUS.md                 ← este archivo
├── README.md                 ← overview + setup
├── pyproject.toml            ← deps + tool.uv.sources para pytorch-cu130
├── .env                      ← ANTHROPIC_API_KEY (no commitear)
├── .env.example
├── docker-compose.yml        ← para cuando se instale Docker
├── data/
│   ├── raw/                  ← material crudo del Drive
│   │   ├── pdfs/  videos/  audios/  screenshots/
│   │   └── _drive_dump/      ← landing zone si bajaste manual
│   ├── processed/
│   │   ├── transcripts/      ← JSON unificado (pdf/video/audio)
│   │   └── extracted/        ← .extracted.json con principios/situaciones/frases
│   └── curated/              ← versión final revisada a mano
├── src/coach_brain/
│   ├── settings.py           ← config con _purge_empty_envs() — no tocar
│   ├── cli.py                ← `coach <subcomando>` (typer)
│   ├── ingest/
│   │   ├── pdfs.py
│   │   ├── transcribe.py     ← faster-whisper GPU
│   │   └── drive.py          ← gdown
│   ├── kb/
│   │   ├── extract.py        ← Claude → JSON estructurado
│   │   ├── embed.py          ← bge-m3
│   │   ├── index.py          ← Qdrant
│   │   └── retrieve.py       ← query para el RAG
│   └── app/
│       └── streamlit_app.py  ← UI con vision
├── prompts/
│   ├── system_coach.md       ← prompt principal del coach
│   └── extractor.md          ← prompt para destilar transcripts
├── scripts/
│   └── smoke_test.py
└── qdrant_storage/           ← persistencia local de Qdrant
```

---

## Decisiones pendientes / a discutir con Ilan

1. **Curación manual de principios extraídos**: una vez que `kb.extract` produzca JSON, conviene que Ilan revise y consolide a mano antes de indexar. ¿Construir un mini-editor o usar VSCode directo?
2. **Memoria por contacto**: aún no implementada. SQLite con tabla `{contacto, intento_previo, qué_funcionó}`. Mes 2 según roadmap.
3. **WhatsApp**: pendiente todo (parser de export, bot). Primero el cerebro, después WhatsApp.
4. **Fine-tuning futuro**: descartado hasta tener >500 conversaciones etiquetadas (mes 6+).

---

## Cómo retomar los batches tras apagar la PC (lo más importante)

**Apagar la PC corta los jobs, pero NO se pierde nada.** Todo es resumable: cada
etapa saltea lo que ya tiene output en disco. Mañana, 3 comandos:

```powershell
cd C:\Users\Ilan\Desktop\Proyectos\coach-brain

# 1) RETOMAR — abre 3 ventanas (transcripción de a batches / vision imágenes / extracción LLM).
#    Siguen corriendo aunque cierres la ventana original. Seguro correrlo varias veces.
powershell -NoProfile -File scripts\resume.ps1

# 2) VER PROGRESO — cuando quieras, no toca nada. Muestra barras y si queda algo.
powershell -NoProfile -File scripts\status.ps1

# 3) PASO FINAL — SOLO cuando status diga "TODO procesado".
#    Junta screenshots → extrae → re-indexa Qdrant (--reset) → levanta el MVP.
powershell -NoProfile -File scripts\finalize.ps1
```

Qué hace cada script:
- **`resume.ps1`** — lanza los 3 jobs en ventanas separadas con log a `logs\` (Tee-Object). Resumable.
- **`status.ps1`** — cuenta videos / notas de voz / imágenes / extracciones hechas vs total, dice si hay python corriendo. Solo lectura.
- **`finalize.ps1`** — cierra cualquier python (libera el lock de Qdrant), agrega imágenes-chat, corre extract, re-indexa con `--reset` y abre Streamlit. **No correr con batches activos.**

Orden natural de la transcripción: primero las 155 clases, después fluye solo a las 8.399 notas de voz de WhatsApp (mismo job, ver `transcribe._collect_targets()`).

## Cómo retomar (handoff para nueva sesión Claude)

1. `cd C:\Users\Ilan\Desktop\Proyectos\coach-brain`
2. `claude`
3. Mensaje inicial sugerido: *"Leé STATUS.md, después seguimos desde el próximo paso."*

Las memorias del proyecto ya están en `~\.claude\projects\C--Users-Ilan-Desktop-Proyectos-coach-brain\memory\` y se cargan automáticamente.
