# STATUS — coach-brain

> Documento de handoff. Si arrancás una sesión Claude nueva, leé esto primero antes de avanzar.

**Última actualización:** 2026-11-17 22:30
**Estado general:** Pipeline de ingesta **terminado** y app **desplegada en Fly.io**. El índice que está sirviendo hoy tiene 38.343 vectores y no incluye los libros más grandes. Los huecos de aguas arriba (PDFs que llegaron por WhatsApp, capturas sin OCR, el PDF escaneado) ya están **cerrados en código y en disco**. Lo que queda corriendo es la re-destilación de 301 fuentes con Ollama local: unas **49 horas** a 43,4 segundos por chunk, medido sobre el log de Ollama. Ver [Huecos del corpus](#huecos-del-corpus-en-curso-2026-11-17).

---

## Qué es esto

Asistente coach conversacional. Subís screenshot de un chat de WhatsApp (o pegás el
texto) → OCR local → analiza el contexto → devuelve diagnóstico + líneas de respuesta.
RAG sobre material propio del coach (clases en video, notas de voz, PDFs, 3 años de
coaching 1-a-1 por WhatsApp).

Overview y setup en `README.md`. Este archivo es el estado operativo y los gotchas.

---

## Estado del setup

| Componente | Estado | Notas |
|---|---|---|
| Python 3.12.13 (uv) | OK | `.python-version` fija 3.12 (uv defaultea a 3.14 que rompe) |
| Toolchain (uv, ffmpeg, git) | OK | winget |
| Docker Desktop | OK (server 29.8.2) | Qdrant igual corre embebido y la imagen de Fly se buildea remoto (`flyctl deploy`), asi que el pipeline no lo necesita. Ojo: Claude NO puede ejecutar `docker run` (gotcha #17) |
| PyTorch + CUDA | OK | **`torch==2.12.0+cu130`** (Blackwell sm_120 de la RTX 5070 Ti requiere cu130, NO cu126) |
| Whisper local | OK | faster-whisper large-v3, GPU |
| Embeddings bge-m3 | OK (cacheado) | dim 1024, en `~/.cache/huggingface/` |
| Ollama | OK | `C:\Users\Ilan\AppData\Local\Programs\Ollama\ollama.exe`, modelo `qwen2.5:14b-instruct` |
| Qdrant | OK (local embebido) | persiste en `./qdrant_storage/` |
| Material del coach | OK | **`C:\Users\Ilan\Desktop\Ilan\Seduccion`** (360GB) — las tres junctions de `data/raw` apuntan ahi y resuelven bien; ver gotcha #7 |
| Deploy Fly.io | OK | app `coach-brain-ilan`, región `iad`, 4GB / 2 vCPU shared, siempre prendida |
| Tests | ⬜ No hay | `tests/` está vacío. Única verificación: `scripts\smoke_test.py` |

---

## Pipeline de ingesta — TERMINADO

No hay nada que re-transcribir. Conteos verificados en disco (2026-11-17):

| Etapa | Salida | Cantidad |
|---|---|---|
| Transcripción | `data/processed/transcripts/*.json` | **8823** |
| Extracción LLM | `data/processed/extracted/*.json` | **8823** (1:1, pero 3184 vacíos — ver abajo) |
| Índice Qdrant | `qdrant_storage/` | **38.343** vectores |

Desglose de las 8823 fuentes:

| Tipo | Cantidad | Detalle |
|---|---|---|
| `wa_audio` | 8396 | notas de voz de WhatsApp (.opus/.m4a/.mp3), Whisper local |
| `wa_video` | 193 | videos de WhatsApp (de 195 en disco; 2 fallaron) |
| `video` | 154 | clases del curso (de 155; 1 falló por ffmpeg) |
| `pdf` | 62 | libros y manuales |
| `docx` | 15 | |
| `whatsapp` | 2 | exports de chat parseados (38.029 mensajes Ema↔Ilan) + imágenes-chat OCR agregadas |
| `epub` | 1 | |

Índice: `principles=8055 situations=12336 style=17952`. Corrió 2026-06-08 00:05, última
extracción 2026-06-07 18:02 → el índice cubre todo lo que había. Log: `logs/index.log`.

La extracción terminó con **Ollama/qwen2.5:14b local (gratis)**, no con Anthropic: la
primera corrida con Anthropic se quedó sin crédito a mitad (ver abajo).

Las imágenes de WhatsApp (8933 .jpg) se procesaron con **RapidOCR local**, no con vision
de Haiku: el intento con Anthropic murió con `BadRequestError` en masa (`logs/images.log`
muestra `err:BadRequestError: 3927` de 7952). Salida en `data/processed/whatsapp_images_ocr/`
(8953 archivos), agregada a un transcript y extraída.

---

## Huecos del corpus (en curso, 2026-11-17)

Los huecos de arriba abajo ya están cerrados en código y en disco. Lo que queda
corriendo es la re-destilación, que es larga pero mecánica.

### Lo que se arregló el 17/11/2026

| Hueco | Estado |
|---|---|
| 6 PDFs que llegaron por WhatsApp y la etapa de PDFs nunca miraba | Cerrado. `ingest/pdfs.py` ahora recorre `data/raw/whatsapp` además de `data/raw/pdfs` y deduplica por `os.path.realpath`. Resultado: `ok=6 skip=68 sin_texto=1 errores=0` |
| 25 capturas de conversaciones bajo `data/raw/docs/Bonus/Interacciones/Mensaje/` sin OCR | Cerrado. `ingest/whatsapp_images_ocr.py` ahora recorre además `data/raw/docs` y `data/raw/screenshots`. 13 resultaron chats reales, 12 imágenes con poco texto. `transcripts/whatsapp_images_chats.json` pasó a 6.336 chats sobre 8.978 imágenes |
| `Método TED para hablar en público`, un PDF escaneado sin texto extraíble | Cerrado. Tercera pasada de OCR en `ingest/pdfs.py`: rasteriza cada página con pymupdf y le aplica RapidOCR, detrás del flag `--ocr`. Salieron 141.340 caracteres en 59 páginas, `engine=rapidocr`, 26 minutos de CPU |
| 8 PDFs sueltos y `Preguntas cita.docx` en la raíz de `Seduccion` | Cerrado, tienen transcript |
| Las 25 capturas figuraban como "sin cubrir" en el inventario aun después del OCR | Era un bug de `scripts/manifest.py`: `_out_name` les pone un sufijo `__<hash8>` a las imágenes de fuera de `data/raw/whatsapp`, y `ocr_coverage()` matcheaba por stem exacto. Corregido: ahora registra las dos formas |
| Un `.zip` adjunto en WhatsApp con material del coach adentro, invisible para el inventario | Cerrado. `scripts/manifest.py` descartaba en silencio toda extensión fuera de `MEDIA_EXT`; ahora las censa y las reporta (gotcha #20). El zip traía `cliente ideal 1.2.pptx` y `canvas 1.3.pdf`, los dos ingestados |
| Los `.pptx` no se leían en ninguna etapa | Cerrado. `ingest/docs.py` lee `.pptx` con el stdlib (zip de XML), una línea por párrafo y las notas del orador pegadas a su diapositiva, resueltas por el `.rels` y no por el número de archivo. Además la etapa ahora recorre `data/raw/whatsapp`, que nunca había mirado |
| PDFs con capa de texto casi vacía que pasaban como cubiertos sin llegar al OCR | Cerrado. El gate de `ingest/pdfs.py` era `if not segments and ocr`; ahora rasteriza también cuando el texto es pobre para la cantidad de páginas (gotcha #19). `canvas 1.3.pdf` pasó de 510 a 3.506 caracteres, y el libro escaneado `El Arte Del Buen Sexo` (132 páginas) pasó de 796 a **93.830 caracteres** en 126 segmentos, `engine=texto+rapidocr` |

| 62 transcripts vacíos sin explicación | Cerrado por medición, no por criterio. Se midió el `mean_volume` de cada fuente con ffmpeg: **51 dan exactamente -91,0 dB**, que es silencio digital — videos grabados con el micrófono apagado y notas de voz mudas. No se perdió nada. Las 11 con audio audible se re-transcribieron sin VAD (gotcha #22) |
| Una grabación de campo que el VAD se comió | Cerrado. De esas 11, diez devolvieron alucinaciones de Whisper ("Gracias por ver el video", "¡Suscríbete!" en bucle) y se dejaron vacías a propósito: meterlas al corpus sería peor que el hueco. La undécima, `wa_audio_00027726-AUDIO-2025-07-07-11-01-37` (50 s), tenía habla real y se regeneró: 252 caracteres, 7 segmentos, `vad_filter: false` anotado en el metadata |
| 3,43 GB reportados como material sin transcribir que en realidad estaba hecho | Cerrado. `78 que es el valor.mp4` perdió el sufijo `-008` en disco y por eso no matcheaba ni por path ni por nombre contra `video_78 que es el valor-008.json`, que tiene 108.793 caracteres. `scripts/manifest.py` tiene ahora un tercer pase que matchea por duración del audio con ffprobe, tolerancia de 1 segundo (gotcha #23). Cubierto con tests en `tests/test_manifest_duracion.py` |
| Dos videos de WhatsApp que fallaban con `tuple index out of range` | Cerrado. No tienen pista de audio: no hay nada que transcribir. `ingest/transcribe.py` ahora lo detecta con ffprobe antes de llamar a Whisper, los saltea con un mensaje claro y los cuenta aparte (`sin_audio=N` en el resumen) |
| 11 PDFs, 2,4 millones de caracteres de libros enteros, indexados como una tira de glifos sin espacios | Cerrado, y es el hueco más grande que apareció. `pypdf.extract_text()` no reconstruye los espacios cuando el PDF los implementa como posicionamiento: devuelve la cantidad correcta de caracteres, así que pasa cualquier control de volumen, pero el texto es `Estoesparaquevayasdiaadia`. pymupdf sí los reconstruye, por geometría, y era solo el fallback de "no salió nada". Ahora `ingest/pdfs.py` compara la proporción de espacios y gana el mejor (gotcha #24). Los 11 se rehicieron: `El poder de los hábitos` pasó de 0,0008 a 0,1487 de espacios, `Basta ya de ser un tipo lindo` y `Mejorando las charlas` de 0,0000 a 0,1536 y 0,1503. Mínimo del corpus ahora: 0,1081. Cubierto con tests en `tests/test_ingest_pptx_pdf.py` |
| El inventario decía que los 5 archivos sin cubrir se perdían, y de 4 de los 5 era falso | Cerrado. `scripts/manifest.py` escribe ahora el motivo de cada uno y lo comprueba en la corrida: `sin pista de audio` (ffprobe no encuentra ninguna), `HUECO: contenedor ilegible`, `HUECO: sin verificar` cuando no se pudo correr ffprobe, `excluido a propósito` (el transcript está en `_fuera_del_corpus/`) o `HUECO` de verdad. Informa los huecos reales aparte del total (gotcha #25). Hoy: **1 hueco real**, el video truncado que necesita `untrunc` |
| Un transcript hecho sin VAD no se distinguía de uno normal | Cerrado. `transcribe.py` anota `vad_filter` en el metadata y `scripts/verificar.py` los lista aparte. Importa porque sin el filtro de voz Whisper alucina sobre ruido (gotcha #22), así que ese texto hay que leerlo antes de confiar en él. Cubierto con tests en `tests/test_transcribe_sin_audio.py` |
| `finalize.ps1` podía matar la extracción en curso y volar el índice sin aviso | Cerrado. Aborta si detecta un proceso del pipeline corriendo (lista los PIDs en vez de matarlos), pide escribir `SI` antes del `--reset` salvo que se le pase `-Yes`, y agrega una segunda pasada de `kb.extract` para no perder el transcript que escribe su propio paso 1 (gotcha #15) |

### Lo que sigue corriendo

Re-destilación con Ollama local (`qwen2.5:14b-instruct`, gratis), log en
`logs/extract_fix.log`. Al 17/11 a las 22:25 quedaban **301 fuentes, unos 4.062
chunks**:

| Tipo | Archivos | Chunks |
|---|---|---|
| `pdf` | 44 | ~3.190 |
| `wa` | 244 | ~276 |
| `video` | 12 | ~17 |
| `whatsapp_images_chats` | 1 | ~579 |

**Ritmo medido, no estimado:** el log de Ollama
(`C:\Users\Ilan\AppData\Local\Ollama\server.log`) timestampea cada generación.
Entre 21:58:22 y 22:20:04 hizo 31 llamadas en 1.302 segundos, o **43,4 segundos
por chunk**, con la GPU al 51 % y 10.390 de 12.227 MiB ocupados — el modelo
corre entero en placa, no hay nada mal configurado. A ese ritmo son unas **49
horas**.

Ese log es la única forma de ver progreso: `kb/extract.py` loguea una línea por
*archivo*, no por chunk, y la barra de `rich` es invisible en un log sin tty.
Un archivo grande como `whatsapp_images_chats` (579 chunks) deja el log mudo
siete horas sin estar colgado.

Las 244 fuentes `wa` son un hallazgo nuevo: de las 3.140 extracciones `wa`
vacías, 244 venían de transcripts de 1.200 caracteres o más, y al mirarlas una
por una resultaron ser grabaciones de campo enteras — interacciones completas
que la corrida original dejó en cero cuando Anthropic se quedó sin crédito. Sus
extracciones se movieron a `data/processed/_reextraer/`. Las otras ~2.900 son
legítimas: mediana 258 caracteres, notas de voz sin nada que destilar.

### El verificador

`scripts/verificar.py` existe para que esta clase de hueco no vuelva a
depender de que alguien lo busque a mano. Es de solo lectura y devuelve 1 si
encuentra algo sospechoso, así que sirve de gate. Revisa:

1. Transcripts con `full_text` vacío, separando el silencio digital (mide el
   volumen de la fuente) del hueco real.
2. Transcripts con menos de 60 caracteres por minuto de audio, en fuentes de
   más de 30 segundos. El habla normal ronda los 800-1000.
3. PDFs cuya capa de texto no llega a `OCR_MIN_CHARS_PER_PAGE`.
4. Texto sin espacios: menos de `MIN_SPACE_RATIO = 0.08` de espacios sobre el
   total de caracteres, en transcripts de 300 caracteres o más. Es el único
   control que detecta un texto con el volumen correcto y el contenido
   ilegible (gotcha #24).
5. Transcripts sin extracción, y extracciones con 0 ítems — marcando aparte
   las que vienen de un transcript de 1.500 caracteres o más, que son las que
   no deberían estar vacías.
6. Extracciones cuyo transcript ya no existe.

Además lista aparte, **sin contarlos como sospechosos**, los transcripts hechos
sin VAD (`metadata.vad_filter == false`). No son un hueco: el flag se usa a
propósito para auditar un transcript vacío. Pero sin el filtro de voz Whisper
alucina frases de su corpus de entrenamiento sobre música y ruido (gotcha #22),
así que ese texto hay que leerlo antes de confiar en él, y el que corra el
pipeline mañana no tiene otra forma de saber cuál es cuál. Hoy aparece uno:
`wa_audio_00027726`, la grabación de campo que sí se recuperó.

```powershell
.venv\Scripts\python.exe -I -X utf8 scripts/verificar.py
.venv\Scripts\python.exe -I -X utf8 scripts/verificar.py --sin-audio  # sin ffmpeg, más rápido
```

### Pendiente cuando termine la extracción

1. **Segunda pasada de `kb.extract`** para los transcripts que se crearon
   después de que la corrida armara su lista de archivos, que por eso no entran
   en ella (`transcripts = sorted(glob("*.json"))` corre una sola vez al
   arrancar, gotcha #15). Ninguno tiene todavía un `.extracted.json`, así que
   basta con volver a correr la etapa, sin borrar nada:
   - `Método TED para hablar en público` (~31 chunks)
   - `pptx_cliente ideal 1.2` (1.665 caracteres)
   - `pdf_canvas 1.3` (3.506 caracteres, `engine=texto+rapidocr`)
   - `pdf_El Arte Del Buen Sexo by Academia Para Caballeros (z-lib.org)`, el
     libro de 132 páginas rescatado con OCR
2. **Verificar** que no quedaron extracciones vacías que no deberían estarlo.
3. **Re-indexar** con `python -m coach_brain.kb.index --reset`. Destructivo:
   borra y reconstruye las 3 colecciones. El backup del índice que funciona hoy
   está en `qdrant_storage.bak/`. Hay que parar Streamlit antes, por el lock de
   Qdrant (gotcha #11), y siempre con `--reset` (gotcha #12).
4. **`flyctl deploy`**, que hornea el índice dentro de la imagen.
5. **Regenerar el inventario** con `.venv\Scripts\python.exe scripts/manifest.py`
   y confirmar que la lista de `cubierto = NO` quedó solo con lo decidido dejar
   afuera.

### Lo que no se puede cerrar

`4 Hablarle a una chica del trabajo, sexualiza.mp4` (977.018.880 bytes) no tiene
transcript y no se puede transcribir. Se escanearon sus bytes completos: tiene
`ftyp` en el byte 4 y `mdat` en el 9587508, y **no tiene átomo `moov` ni `moof`
en ninguna parte**. Es una descarga truncada, sin índice, indecodificable. Volver
a correr `ingest.transcribe` no sirve.

La única vía es reconstruirle el índice con `untrunc`, usando otra clase del
mismo curso como referencia. Ojo que `anthwlock/untrunc` es el nombre del repo
de GitHub, **no** existe como imagen en Docker Hub (`pull access denied ...
repository does not exist`). Las que sí existen y bajan bien son `zany/untrunc`
(126 MB, es ese mismo fork), `rodolfoap/untrunc` y `thej6s/untrunc`. El comando,
con los dos archivos ya enlazados por hardlink en el scratchpad de la sesión:

```powershell
docker run --rm -v "<dir>:/work" zany/untrunc /work/ref.mp4 /work/roto.mp4
```

`ref.mp4` es `75 Reglas de un seductor.mp4` (sano, 559.818.443 bytes). No toca
ninguno de los dos archivos: escribe un `roto_fixed.mp4` nuevo al lado. El
clasificador de permisos de auto mode rechaza `docker run` desde Claude, así que
este paso lo tiene que correr Ilan a mano.

Los otros dos faltantes son clips de WhatsApp, y ya no son un misterio:
`00031653-VIDEO-2025-09-08-11-54-41.mp4` y
`00033588-GIF-2025-10-03-00-26-43.mp4`. Se los pasó por ffprobe: **ninguno de
los dos declara una pista de audio**. El `tuple index out of range` que tiraba
`ingest.transcribe` era eso, dicho mal. Nunca van a dar texto, y ahora la etapa
los saltea con un mensaje que lo explica en vez de fallar.

### Decisión sobre borrar los 360 GB

Está documentada aparte, en [BORRAR_MATERIAL.md](BORRAR_MATERIAL.md) y
[INVENTARIO.md](INVENTARIO.md). **No está autorizado borrar nada.**

---

## La app (qué está construido)

`src/coach_brain/app/streamlit_app.py` — 3 tabs: **💬 Chat** (screenshot o texto →
diagnóstico + líneas), **🎯 Líneas** (líneas para una situación puntual), **📝 Frases**
(notas personales).

- **Auth** — PIN vía `APP_PASSWORD`, con rate limiting por IP y backoff exponencial
  (30s, 60s, 120s… tope 15 min). Sesión persistente con cookie firmada HMAC usando el PIN
  como clave. Sin `APP_PASSWORD` seteado la app queda abierta (solo para uso local).
- **OCR** — RapidOCR local y gratis; la posición de la burbuja decide quién habla.
  Stripea el encabezado de WhatsApp (nombre + "en línea") que se pegaba al primer mensaje.
- **Memoria por contacto** — `src/coach_brain/kb/memory.py`, SQLite en `DATA_DIR`. El
  perfil se actualiza después de cada análisis y se reinyecta en los siguientes.
  Autodetección de contacto por nombre.
- **LLM intercambiable** — `src/coach_brain/llm.py`, un solo code path: Anthropic, Ollama
  local, o cualquier proveedor compatible con la API de OpenAI (Gemini, Groq, Cerebras,
  OpenRouter — varios con tier gratis). Se elige desde la UI.
- **PWA** — `src/coach_brain/app/pwa.py`, instalable en el teléfono. Layout mobile.
- **Warmup** — un `encode` dummy al arrancar pagina los pesos de bge-m3 con spinner;
  sin eso la primera consulta real tardaba ~145s. Después quedan en ~0.4s.

### Deploy

```powershell
flyctl secrets set GEMINI_API_KEY=... APP_PASSWORD=...
flyctl deploy
```

App `coach-brain-ilan`. El índice de Qdrant va **horneado en la imagen** (`COPY
qdrant_storage/`); el volumen persistente `/data` solo guarda la base de contactos y las
notas. `auto_stop_machines = 'off'` + `min_machines_running = 1`: cargar bge-m3 (2.2GB)
desde cero tarda ~3 min, y con la máquina apagándose por inactividad se pagaba ese arranque
en cada visita.

---

## Pipeline completo (referencia)

Refrescar PATH primero si es shell nuevo:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:PYTHONIOENCODING = "utf-8"
```

| Paso | Comando | Salida |
|---|---|---|
| PDFs | `python -m coach_brain.ingest.pdfs` | `transcripts/pdf_*.json` |
| Docs | `python -m coach_brain.ingest.docs` | `transcripts/docx_*.json` |
| WhatsApp texto | `python -m coach_brain.ingest.whatsapp` | `transcripts/whatsapp_*.json` |
| WhatsApp imágenes | `python -m coach_brain.ingest.whatsapp_images_ocr` | `whatsapp_images_ocr/` |
| Videos/audios | `powershell -NoProfile -File scripts\transcribe_batches.ps1 -BatchSize 8` | `transcripts/video_*.json`, `wa_*.json` |
| Extracción | `python -m coach_brain.kb.extract` | `extracted/*.extracted.json` |
| Índice | `python -m coach_brain.kb.index --reset` | 3 colecciones Qdrant |
| App | `python -m streamlit run src\coach_brain\app\streamlit_app.py` | http://localhost:8501 |

También `coach <subcomando>` (typer, `src/coach_brain/cli.py`): `ingest-pdfs`, `ingest-docs`,
`ingest-whatsapp`, `transcribe`, `download-drive`, `extract`, `index`, `app-run`.

### Scripts de orquestación

- **`scripts\resume.ps1`** — lanza transcripción (GPU) + OCR de imágenes (CPU) en ventanas
  separadas con log a `logs\`. Resumable, seguro correrlo varias veces.
- **`scripts\status.ps1`** — progreso hecho/total. Solo lee.
- **`scripts\watch.ps1`** — dashboard en vivo, autoactualiza. Solo lee.
- **`scripts\transcribe_batches.ps1`** — un proceso fresco por batch de 8 (libera VRAM
  entre tandas, no traba la máquina).
- **`scripts\finalize.ps1`** — agrega imágenes-chat, extrae con Ollama, re-indexa con
  `--reset`, abre Streamlit. **Ojo: el `--reset` destruye el índice actual.** No correr con
  batches activos.
- **`scripts\upload_to_qdrant_cloud.py`** — migra los vectores del Qdrant embebido a Qdrant
  Cloud (alternativa a hornearlos en la imagen; hoy no se usa).
- **`scripts\smoke_test.py`** — `python scripts\smoke_test.py --quick`.

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
Console default de Windows rompe con caracteres Unicode (✓, ✗, emojis en código). Setear `$env:PYTHONIOENCODING = "utf-8"` antes de correr scripts que renderizan texto. Los logs en `logs\` quedaron en UTF-16 por el Tee-Object de PowerShell: para leerlos desde Bash, `tr -d '\0'`.

### 6. Drive: gdown no acepta carpetas "Restricted"
Devuelve HTTP 401 (`data/_drive_download.log`). Permiso necesario: "Anyone with the link". Ya no aplica — el material está local, y **no hay** copia en Drive accesible: la local es la única.

### 7. El material son tres junctions que apuntan a la MISMA carpeta
`data/raw/videos`, `data/raw/pdfs` y `data/raw/docs` son junctions de NTFS y las tres apuntan
a **`C:\Users\Ilan\Desktop\Ilan\Seduccion`** (360GB), donde `Bonus` es una subcarpeta.
Verificado el 2026-11-17: **resuelven bien**, no están colgadas (una versión anterior de este
documento decía que apuntaban al path viejo `C:\Users\Ilan\Desktop\Bonus`; eso ya no es
cierto). Esa topología de tres nombres para un solo árbol tiene dos consecuencias que
mordieron a este proyecto: un ingestor que recorre los tres roots sin deduplicar por
`os.path.realpath` ve los mismos 360GB tres veces, y al revés, el material alcanzable solo
por un root que el ingestor nunca recorría quedaba silenciosamente afuera. `scripts/manifest.py`
deduplica por realpath, y también lo hacen `ingest/pdfs.py` y `ingest/whatsapp_images_ocr.py`
después del arreglo del 2026-11-17.

Si alguna vez hay que re-apuntar una junction (borrar el enlace viejo primero, después
recrearlo):
```powershell
cmd /c rmdir "C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\videos"
cmd /c mklink /J "C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\videos" "C:\Users\Ilan\Desktop\Ilan\Seduccion"
```
(idem `pdfs` y `docs`). Usar `cmd /c rmdir` y **nunca** `Remove-Item -Recurse`: en PowerShell 5.1 este último puede entrar en la junction y borrar los 360GB del destino. `find` de MSYS NO atraviesa junctions; pathlib de Python SÍ (que es lo que usan los ingestores), así que un `find` vacío no prueba que la junction esté rota.

### 8. Whisper: HF_HUB_DISABLE_SYMLINKS=1
Seteado a nivel User. Sin esto, la 1ra descarga del modelo falla con `WinError 1314` (symlinks del cache de HF sin Developer Mode).

### 9. CUDA Blackwell (sm_120) + faster-whisper: cublas en el dir de ctranslate2
torch usa CUDA 13, CTranslate2 necesita CUDA 12. Hay que tener `cublas64_12.dll` + `cublasLt64_12.dll` copiadas dentro de `.venv\Lib\site-packages\ctranslate2\` (CTranslate2 las carga por nombre ignorando `add_dll_directory`). `transcribe._register_cuda_dlls()` lo auto-cura. NUNCA copiar `cudnn64_9.dll` ahí (la pone torch; la versión cu12 rompe el import de torch con `WinError 127`). Probado: CTranslate2 corre en sm_120 con esto.

### 10. Orden de import: sentence_transformers ANTES que qdrant_client
En Windows con `torch 2.12.0+cu130`, importar `qdrant_client(.models)` **antes** que `sentence_transformers`/torch produce un segfault silencioso (access violation `0xC0000005`, exit `-1073741819`, sin traceback) por conflicto de libs nativas. `kb/index.py` fuerza `import sentence_transformers` arriba de todo (no reordenar ese import). `retrieve.py` y `streamlit_app.py` ya importan `embed` antes que `index`, así que están a salvo. Si creás un módulo nuevo que use ambos, importá embeddings primero.

### 11. Qdrant embebido toma lock de proceso único
El modo local (`QDRANT_PATH`) bloquea la carpeta para UN solo proceso. No se puede indexar mientras Streamlit está levantado. Para re-indexar: parar Streamlit → `index --reset` → relanzar Streamlit. La extracción (solo escribe JSON) no toca Qdrant, corre en paralelo sin problema.

### 12. Re-indexar SIEMPRE con --reset
`index.py` usa UUIDs random como point IDs, así que correr `index` sin `--reset` DUPLICA todo. Siempre `python -m coach_brain.kb.index --reset`.

### 13. Las etapas saltean por existencia de archivo, no por contenido
Si una etapa escribió un archivo vacío por un error (lo que pasó con los 32 PDFs), re-correrla no lo reintenta. Para forzar: borrar el output vacío primero.

### 14. Anthropic se quedó sin crédito a mitad del pipeline
Es lo que dejó el hueco del corpus. Todo el pipeline tiene camino gratis: Whisper local, RapidOCR local, Ollama/qwen2.5:14b para extracción, bge-m3 local para embeddings. Preferir esos para corridas masivas.

### 15. `kb.extract` arma su lista de archivos una sola vez, al arrancar
`transcripts = sorted(transcripts_dir.glob("*.json"))` corre una vez antes del loop. Un transcript que se crea *después* de ese glob no entra en esa corrida, aunque la corrida siga horas. Pasó con `Método TED`: su transcript se escribió a las 22:10:10 y la corrida había armado la lista a las 21:57. Hace falta una segunda pasada.

### 16. La extracción no loguea progreso por chunk
`kb/extract.py` imprime una línea por *archivo* (`OK <nombre> -> p=.. s=.. f=..`), no por chunk, y la barra de `rich` no se ve en un log sin tty. Un archivo de 579 chunks deja `logs/extract_fix.log` mudo siete horas sin estar colgado. Para ver progreso real y medir el ritmo, usar el log de Ollama, que timestampea cada generación: `C:\\Users\Ilan\AppData\Local\Ollama\server.log`, líneas `POST "/api/chat"`.

### 17. `docker run` está bloqueado para Claude en auto mode
El clasificador de permisos lo rechaza con `Reason: [Undermining Oversight]`, con montaje de volumen o sin él. Cualquier paso que necesite Docker (la recuperación con `untrunc`) lo tiene que correr Ilan a mano.

### 18. El OCR de imágenes fuera de `data/raw/whatsapp` lleva sufijo de hash
`_out_name` en `ingest/whatsapp_images_ocr.py` le agrega `__<hash8>` al nombre de salida de las imágenes que vienen de otros roots, para evitar colisiones de nombre. Cualquier cosa que cruce cobertura contra esos archivos tiene que contemplar las dos formas del nombre. `scripts/manifest.py` lo hacía mal y quedaba reportando 25 capturas como "sin cubrir" después de haberles pasado OCR.

### 19. Un PDF con capa de texto casi vacía pasaba como "cubierto"
`ingest/pdfs.py` solo rasterizaba cuando la extracción devolvía **cero** segmentos. Un PDF que trae unas pocas líneas de texto y el resto en imágenes quedaba registrado como cubierto y nunca llegaba al OCR. Dos casos reales: `canvas 1.3.pdf`, una exportación de Canva, con 510 caracteres en 2 páginas (el canvas de negocio del coach estaba todo en la imagen), y `El Arte Del Buen Sexo`, un libro escaneado de 132 páginas registrado con 796 caracteres en total. Ahora el gate es `OCR_MIN_CHARS_PER_PAGE = 400`: si el texto extraído es pobre para la cantidad de páginas, también rasteriza, y se queda página por página con el resultado más largo. El motor queda anotado como `texto+rapidocr` en el metadata del transcript cuando se mezclaron las dos fuentes.

Para encontrar más casos así, el criterio es `len(full_text) / pages`: un PDF de texto real ronda los 1500-3000 caracteres por página.

### 20. Material real escondido dentro de un `.zip`
`scripts/manifest.py` descartaba en silencio cualquier extensión que no estuviera en `MEDIA_EXT` (`if ext not in MEDIA_EXT: continue`), así que un formato que el pipeline no maneja no aparecía en ninguna parte del inventario: parecía no existir. Había un `.zip` adjunto en WhatsApp con un `.pptx` y un `.pdf` adentro, los dos con contenido del coach. Ahora el inventario tiene una sección "Formatos que el inventario no clasifica" que cuenta y ejemplifica todo lo descartado. Lo que quedó afuera a propósito: dos `.vcf` (tarjetas de contacto) y un `.drawio` de 15 KB con el mismo diagrama que ya está en el `.pptx`.

---

### 21. Un transcript vacío no es lo mismo que un hueco del corpus

De los 62 transcripts vacíos, 51 vienen de fuentes que **no tienen sonido**:
videos grabados con el micrófono apagado, notas de voz mudas. El test que los
separa es barato y no necesita decodificar nada:

```powershell
ffmpeg -hide_banner -nostats -i "<archivo>" -vn -af volumedetect -f null -
```

El silencio digital puro da `mean_volume: -91.0 dB`, siempre ese valor exacto.
Habla real, incluso lejana o de fondo, queda bien arriba de -60 dB. Sin esta
medición los 62 parecen 62 huecos y mandan a re-transcribir horas de audio
que no tiene nada que transcribir.

### 22. El VAD de Whisper decide que no hay habla, y la etapa no reintenta

`transcribe_file` corre con `vad_filter=True`. Cuando el detector de voz no
encuentra habla, Whisper devuelve cero segmentos y el transcript se guarda
vacío. Como la etapa saltea por existencia del archivo (gotcha #13), ese vacío
es permanente.

Para auditarlo está `--sin-vad`:

```powershell
.venv\Scripts\python.exe -m coach_brain.ingest.transcribe --force --limit 1 --sin-vad
```

**Hay que leer el resultado antes de confiar en él.** Sin VAD, Whisper alucina
sobre música y ruido con frases de su corpus de entrenamiento: "Gracias por ver
el video", "¡Suscríbete!", "Más información www.cdc.gov.ar", repetidas en
bucle. De las 11 fuentes audibles con transcript vacío, diez devolvieron
exactamente eso y se dejaron vacías a propósito. Solo una tenía habla real.

### 23. Un archivo renombrado en disco aparece como un hueco del corpus

`scripts/manifest.py` cruzaba fuentes con transcripts por path y por nombre de
archivo. `78 que es el valor.mp4` perdió el sufijo `-008` en algún momento, así
que no matcheaba contra `video_78 que es el valor-008.json` por ninguna de las
dos vías, y el inventario reportaba 3,43 GB de material sin transcribir que
estaba transcripto hace meses.

El tercer pase matchea por duración del audio, leída del contenedor con
ffprobe, que es lo único que sobrevive a un renombre. Tolerancia de 1 segundo,
y el transcript se consume al matchear para que no pueda matchear dos veces.
Solo corre si quedó algo sin cubrir, así que no cuesta 8.823 llamadas a
ffprobe en la corrida normal.

### 24. Un PDF puede dar el volumen de texto correcto y ser ilegible igual

`pypdf.extract_text()` no reconstruye los espacios cuando el PDF los implementa
como posicionamiento de glifos en vez de glifos de espacio. Devuelve la
cantidad correcta de caracteres —así que pasa el control de volumen del gotcha
#19 y cualquier otro— pero el texto es una tira pegada:
`Estoesparaquevayasdiaadia`. El LLM no puede destilar nada de ahí y los
vectores quedan sin sentido semántico.

Once PDFs del corpus estaban así, 2,4 millones de caracteres, libros enteros, y
llevaban meses indexados y sirviendo. El motivo estructural es el mismo que el
gotcha #19: pymupdf estaba puesto como fallback de "pypdf no devolvió nada", y
el gate preguntaba "¿salió texto?" en vez de "¿sirve?".

El test que los separa es la proporción de espacios sobre el total de
caracteres. En el corpus la mediana es **0,1921** y el mínimo real es **0,1081**
(el OCR de capturas de chat); los once roto iban de **0,0000 a 0,0716**. El
umbral quedó en `MIN_SPACE_RATIO = 0.08`, y vive en dos lugares a propósito:
`ingest/pdfs.py` lo usa para elegir motor y `scripts/verificar.py` para
detectar lo que ya está escrito en disco.

Esto no es mojibake: los caracteres de reemplazo (`�`) del corpus son 0 en
todos los transcripts. El `�` que aparece en la terminal es el renderizado
cp1252 de la consola (gotcha #5), no el archivo.

### 25. "Sin cubrir" no significa "se pierde": hay cuatro motivos distintos

`INVENTARIO.md` es el registro sobre el que se decide si el material crudo se
borra, y durante un tiempo dijo de los cinco archivos sin cubrir que "su
contenido se pierde y no hay forma de recuperarlo". De cuatro de los cinco era
falso, y el registro servía para lo contrario de lo que se escribió.

`scripts/manifest.py` ahora escribe el motivo de cada uno, y lo **comprueba** en
la corrida en vez de darlo por sabido:

| Motivo | Cómo se detecta | ¿Se pierde algo? |
|---|---|---|
| `sin pista de audio` | `ffprobe -select_streams a` devuelve 0 y no imprime ningún índice | No: es un GIF o un video grabado con el micrófono apagado |
| `HUECO: contenedor ilegible` | ffprobe corrió y falló al abrir el archivo | **Sí.** Es el caso que hay que atender |
| `excluido a propósito` | hay un transcript con ese stem en `data/processed/_fuera_del_corpus/` | No: el material es ruido para el corpus, el texto está guardado |
| `HUECO: sin verificar` | no se pudo correr ffprobe | **Se asume que sí**, por las dudas |
| `HUECO: falta procesarlo` | ninguno de los anteriores | **Sí** |

Los dos últimos son la parte que importa del diseño. "No se pudo correr
ffprobe" **no** es "el archivo está roto": decir lo segundo sería afirmar una
pérdida que nunca se comprobó, que es el error que este gotcha viene a
arreglar. Y la duda se resuelve siempre del lado conservador — sin verificación
el archivo queda marcado como hueco, así que no entra en la lista de lo que se
puede borrar.

La exclusión se detecta por la presencia del archivo, no por una lista
hardcodeada, así que sigue funcionando si mañana se saca otra cosa del corpus.
El resumen y el stdout informan los huecos reales aparte del total: hoy, **5
sin cubrir y 1 hueco real** — el video truncado que necesita `untrunc`.

## Costos reales

| Etapa | Costo |
|---|---|
| Transcripción (8823 fuentes, Whisper local GPU) | $0 |
| OCR de 8953 imágenes (RapidOCR local) | $0 |
| Extracción LLM (Ollama/qwen2.5:14b local) | $0 |
| Embeddings (bge-m3 local) | $0 |
| Fly.io (4GB / 2 vCPU shared, siempre prendida) | ~$10-15/mes |
| Por consulta | $0 con proveedor free tier (Gemini/Groq/OpenRouter `:free`); ~$0.005–0.02 con Anthropic |

---

## Estructura del proyecto

```
coach-brain/
├── STATUS.md                 ← este archivo
├── README.md                 ← overview + setup
├── Dockerfile                ← imagen de Fly (digest pinneado, bge-m3 y RapidOCR pre-bakeados)
├── fly.toml                  ← app coach-brain-ilan
├── requirements-cloud.txt    ← deps de la imagen (torch CPU)
├── pyproject.toml            ← deps + tool.uv.sources para pytorch-cu130
├── .env                      ← keys + config (no commitear)
├── data/
│   ├── raw/
│   │   ├── videos/ pdfs/ docs/   ← junctions COLGADAS (gotcha #7)
│   │   ├── whatsapp/             ← real: 195 mp4, 8310 .opus, 8933 .jpg
│   │   └── _drive_dump/          ← vacío
│   ├── processed/
│   │   ├── transcripts/      ← 8823 JSON unificados
│   │   ├── extracted/        ← 8823 .extracted.json (3184 vacíos)
│   │   ├── whatsapp_images_ocr/  ← 8953 salidas de RapidOCR
│   │   └── whatsapp_images/
│   └── curated/              ← vacío (curación manual nunca se hizo)
├── src/coach_brain/
│   ├── settings.py           ← config con _purge_empty_envs() — no tocar
│   ├── llm.py                ← un code path para todos los proveedores
│   ├── cli.py                ← `coach <subcomando>` (typer)
│   ├── ingest/               ← pdfs, docs, transcribe, whatsapp, whatsapp_images(_ocr), drive
│   ├── kb/                   ← extract, embed, index, retrieve, memory
│   └── app/                  ← streamlit_app.py, pwa.py
├── prompts/
│   ├── system_coach.md       ← prompt principal
│   ├── situational_lines.md  ← prompt del tab Líneas
│   └── extractor.md          ← prompt para destilar transcripts
├── scripts/                  ← resume, status, watch, transcribe_batches, finalize, smoke_test, upload_to_qdrant_cloud
├── tests/                    ← VACÍO
├── logs/                     ← UTF-16 (gotcha #5)
└── qdrant_storage/           ← 38.343 vectores — IRREMPLAZABLE sin re-correr todo
```

---

## Decisiones pendientes / a discutir con Ilan

1. **Cerrar el hueco de los PDFs** — en curso, corriendo con Ollama local. Cuando termine:
   segunda pasada para `Método TED`, `index --reset` y `flyctl deploy`. Ver arriba.
2. **Recuperar `4 Hablarle a una chica del trabajo, sexualiza.mp4`** con `untrunc`, o
   aceptar que esa clase queda afuera. El comando está arriba; lo tiene que correr Ilan
   porque Claude no puede ejecutar `docker run` (gotcha #17).
3. **Borrar o mover los 360 GB de material crudo** — documentado en `BORRAR_MATERIAL.md`
   e `INVENTARIO.md`. **No está autorizado borrar nada.** La recomendación es moverlo a un
   disco externo en lugar de borrarlo, para no perder la opción de re-transcribir.
4. **Curación manual** — `data/curated/` sigue vacío. La idea era revisar a mano los
   principios extraídos antes de indexar; nunca se hizo. ¿Vale la pena a esta altura?
5. **Tests** — `tests/` está vacío. El proyecto está en producción sin una sola prueba
   automatizada.
6. **Bot de WhatsApp** — sigue pendiente. Hoy la entrada es screenshot/texto en la web.
7. **Fine-tuning** — descartado hasta tener >500 conversaciones etiquetadas.

---

## Cómo retomar (handoff para nueva sesión Claude)

1. `cd C:\Users\Ilan\Desktop\Proyectos\coach-brain`
2. `claude`
3. Mensaje inicial sugerido: *"Leé STATUS.md, después seguimos desde el próximo paso."*

Las memorias del proyecto están en `~\.claude\projects\C--Users-Ilan-Desktop-Proyectos-coach-brain\memory\`
y se cargan automáticamente.
