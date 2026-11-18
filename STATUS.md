# STATUS — coach-brain

> Documento de handoff. Si arrancás una sesión Claude nueva, leé esto primero antes de avanzar.

**Última actualización:** 2026-11-17 22:30
**Estado general:** Pipeline de ingesta **terminado** y app **desplegada en Fly.io**. El índice que está sirviendo hoy tiene 38.343 vectores y no incluye los libros más grandes. Los huecos de aguas arriba (PDFs que llegaron por WhatsApp, capturas sin OCR, el PDF escaneado) ya están **cerrados en código y en disco**. Lo que queda corriendo es la re-destilación de 301 fuentes con Ollama local (medidas el 17/11 a las 22:25): unas **49 horas** a 43,4 segundos por chunk, medido sobre el log de Ollama. Ver [Huecos del corpus](#huecos-del-corpus-en-curso-2026-11-17).

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

No hay nada que re-transcribir. Conteos verificados en disco (2026-11-17, 23:50):

| Etapa | Salida | Cantidad |
|---|---|---|
| Transcripción | `data/processed/transcripts/*.json` | **8838** |
| Extracción LLM | `data/processed/extracted/*.json` | **8535**, más 303 en curso con Ollama |
| Índice Qdrant | `qdrant_storage/` | **38.343** vectores, del índice viejo |

Los 303 que faltan no son un hueco nuevo: 250 son extracciones que se mandaron a
re-destilar a propósito (`data/processed/_reextraer/`: 244 notas de WhatsApp y 5 PDF
reconstruidos), 44 estaban vacías y quedaron respaldadas en `_extracted_empty_backup/`,
y el resto son transcripts nuevos. El índice que sirve hoy se armó antes de todo esto,
así que todavía no los tiene: hay que reconstruirlo cuando la extracción termine.

Desglose de las 8838 fuentes:

| Tipo | Cantidad | Detalle |
|---|---|---|
| `wa_audio` | 8396 | notas de voz de WhatsApp (.opus/.m4a/.mp3), Whisper local |
| `wa_video` | 193 | videos de WhatsApp (de 195 en disco; los 2 que faltan no tienen pista de audio) |
| `video` | 154 | clases del curso (de 155; al que falta se le truncó el contenedor) |
| `pdf` | 75 | libros y manuales (eran 62: +13 al cerrarse el hueco de los que llegaron por WhatsApp) |
| `docx` | 16 | |
| `pptx` | 1 | |
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

### Lo que se arregló el 18/11/2026

**Los 154 videos de clase nunca se perdieron: se movieron.** Las tres junctions
de `data/raw` apuntan todas a `Desktop\Ilan\Seduccion` (gotcha 7), y después de
la ingesta el material se reorganizó una carpeta más abajo, dentro de
`Seduccion\Bonus\`. El resultado fue que **224 de los 8.838 transcripts
quedaron con un `source_path` que ya no resuelve**: 154 `video`, 54 `pdf`, 15
`docx` y 1 `epub`.

El inventario nunca lo notó, y por un buen motivo: `scripts/manifest.py` matchea
por nombre y por duración sobre el árbol real, no por la ruta guardada
(gotcha 23). Así que `INVENTARIO.md` siempre estuvo bien y el corpus siempre
estuvo completo. El daño era **diferido**: no faltaba nada, pero cualquier etapa
que vuelve a leer la fuente —re-transcribir sin VAD, recalcular una duración,
re-extraer un PDF— fallaba con "no existe" sobre un archivo que está en disco.
Es la clase de defecto que no se ve hasta que se necesita.

`scripts/reparar_source_path.py` lo cerró: **219 transcripts reapuntados**, 218
por nombre y 1 estirando el sufijo `-NNN` que agrega el pipeline
(`78 que es el valor-008.mp4` → `78 que es el valor.mp4`). Cada uno tiene copia
en `data/processed/_transcripts_backup_source_path/` y la ruta vieja guardada en
`metadata.source_path_previo`. Dos detalles del diseño que importan:

- **Se niega a adivinar.** Si un nombre tiene más de un candidato en disco,
  reporta `ambiguo` y no escribe. Apuntar un transcript al archivo equivocado no
  lo avisa nadie después.
- **Escribe con `os.replace`.** La extracción con Ollama estaba leyendo esos
  mismos transcripts en paralelo, y `write_text` no es atómico: un lector podía
  haber visto un archivo truncado a mitad de camino.

Después del arreglo, `manifest.py` informa **8.833 transcripts con path exacto y
5 a resolver por nombre**, contra 224 antes.

**Los 5 que quedan no son un hueco: son duplicados.** Cada uno es el mismo
contenido que otro transcript cuyo archivo sí está en disco. La misma fuente
entró al pipeline dos veces con nombres distintos, y después se borró una de las
dos copias del archivo.

| Transcript sin fuente | Gemelo con archivo en disco |
|---|---|
| `video_14 juego de texto-056` | `video_15 juego de texto-069` |
| `video_36 Clase vip-029` | `video_36 Clase vip-095` |
| `video_38 Clase Vip-014` | `video_38 Clase Vip-092` |
| `video_39 Clase vip-015` | `video_39 Clase vip-091` |
| `pdf_21-Claves-Sexualizar-Convencer (Luis Tejedor (Egoland))-2` | `pdf__21-claves-sexualisar-convenser` |

Buscando sobre todo el corpus aparecen **9 grupos duplicados**, no 5: los 4
videos de clase, 3 libros con doble nombre (`Mode One`, `Conversador
Irresistible`, `21 Claves`) y 2 notas de voz idénticas al milisegundo. Cero
pérdida de contenido; lo que hay es **texto repetido en el corpus y en el
índice**, que sesga la búsqueda porque el mismo pasaje ocupa dos lugares entre
los resultados más parecidos.

**Sacar un duplicado borra datos, así que no se tocó.** `INVENTARIO.md` nombra
los dos gemelos de cada par y `scripts/verificar.py` los lista en cada corrida.
Es decisión de Ilan (ver gotcha 26 y "Decisiones pendientes").

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
2. Transcripts con menos de `MIN_CHARS_POR_MINUTO = 250` caracteres por minuto
   de audio, en fuentes de más de 30 segundos. El piso está calibrado contra la
   distribución real del corpus, medida sobre los 4.097 `wa_audio` de 30 s o
   más (2026-11-17):

   | mínimo | p1 | p5 | p25 | mediana | p75 |
   |---|---|---|---|---|---|
   | 0,0 | 467,9 | 713,0 | 846,1 | **916,9** | 980,2 |

   El 99% de las notas de voz pasa los 467 c/min. El piso estaba en 60, siete
   veces por debajo del percentil 1: un transcript al que el VAD le había
   comido el 70% del habla pasaba limpio. Con 250 quedan marcados 10 archivos
   de 120 s o más (eran 7), y sigue muy por debajo de la cola legítima: las
   grabaciones de calle con minutos de ruido ambiente tienen densidad baja y
   están completas. Marcar no es decidir: hay que leer el transcript.
3. PDFs cuya capa de texto no llega a `OCR_MIN_CHARS_PER_PAGE`.
4. Texto sin espacios: menos de `MIN_SPACE_RATIO = 0.08` de espacios sobre el
   total de caracteres, en transcripts de 300 caracteres o más. Es el único
   control que detecta un texto con el volumen correcto y el contenido
   ilegible (gotcha #24).
5. Transcripts sin extracción, y extracciones con 0 ítems — marcando aparte
   las que vienen de un transcript de 1.500 caracteres o más, que son las que
   no deberían estar vacías.
6. Extracciones cuyo transcript ya no existe.
7. Resultados de la auditoría sin VAD esperando en `data/processed/_novad/`
   sin que nadie los haya juzgado. **No cuenta como sospechoso**: el verificador
   devuelve 1 cuando hay sospechosos y `finalize.ps1` lo usa de gate, y trabajo
   pendiente de que lo lea una persona no es un hueco del corpus.

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

### La auditoría sin VAD

`scripts/retranscribir_sin_vad.py` es la contraparte del gotcha #22: vuelve a
transcribir **con el filtro de voz apagado** a los archivos que el verificador
marca como sospechosos, para ver si el VAD se comió habla real. Es una
auditoría, no una etapa del pipeline.

**Lo que sale de ahí no entra solo al corpus.** Los resultados van a
`data/processed/_novad/`, que no es un directorio que lea ninguna etapa, y cada
uno lleva `vad_filter: false` en su `metadata`, así que el verificador los
lista aparte para siempre (ver "El verificador", punto final). Promover uno es
copiarlo a mano a `data/processed/transcripts/` y re-indexar, y eso lo decide
una persona que leyó el texto.

**A quién manda a re-transcribir lo decide el verificador, no una lista.** Los
umbrales se importan de `scripts/verificar.py` — `MIN_CHARS_POR_MINUTO`,
`DUR_MINIMA_PARA_JUZGAR`, `SILENCIO_DB` — y hay un test que falla si se
separan (`test_umbrales_vienen_de_verificar`). La primera versión tenía 14
nombres escritos a mano, y tres de ellos no matcheaban nunca porque el nombre
en disco tiene mojibake horneado (U+2560 + U+00FC donde va una "ú"): derivar la
lista del disco hace que eso no importe. Candidatos reales: **25**.

Son **dos poblaciones distintas**, y confundirlas es el error que hay que no
cometer:

| Población | Cómo se juzga | Por qué |
|---|---|---|
| Transcript con texto | densidad, en fuentes de 30 s o más | Poco texto para mucho audio es el síntoma de que el filtro recortó. |
| Transcript vacío | volumen medio de la fuente contra `SILENCIO_DB` | 0 caracteres puede ser un video muteado (-91 dB medido) o una falla del filtro. La duración acá no importa. |

Mezclarlas da 33 candidatos, 18 de ellos con 0 caracteres, que es exactamente
la población que **no** hay que mandar a re-transcribir: sin VAD, sobre
silencio, Whisper devuelve alucinación. La decisión vive en `clasificar()`,
separada de la lectura del disco, con 13 tests que fijan sobre todo los casos
en los que negarse es lo correcto.

**El detector de alucinaciones.** Sin filtro de voz Whisper no devuelve silencio
cuando no hay habla: devuelve frases de los subtítulos de YouTube con los que se
entrenó. De los primeros 17 resultados, **13 son basura y 4 pueden ser habla
real**. Leerlos todos a ojo no escala, así que `parece_alucinacion()` marca con
tres señales, las tres medidas en este corpus:

1. El texto entero es una de esas frases (`"Gracias por ver el video."`).
2. Sacando las frases inventadas queda menos de `RESTO_MINIMO = 0.25` del
   texto, así que no había nada más. El caso bueno conserva el 96 %.
3. Una secuencia corta de palabras cubre `COBERTURA_PARA_BUCLE = 0.6` o más del
   texto: el decoder en bucle.

La tercera señal hizo falta porque en bucle Whisper muchas veces no pone
puntuación. `"La ciudad de Bogotá"` nueve veces seguidas es **una sola frase**
para cualquier criterio que parta por `.!?`, y así se escapaban cuatro
resultados obvios. La cobertura del n-grama cuenta **posiciones distintas**, no
apariciones: multiplicar apariciones por largo sobrecuenta cuando se solapan —
diez `"o sea"` seguidos ocupan 20 palabras, pero el 4-grama aparece en 7
posiciones solapadas y multiplicando daría 28 de 40, un 70 % que no existe.

El veredicto solo dice `DESCARTAR` o `REVISAR`; **no existe "promover"**, y hay
un test que lo fija (`test_veredicto_nunca_promueve_solo`). La calibración
carga del lado de no marcar habla real: el caso bueno de 6.445 caracteres
contiene `"Muchas gracias"` quince veces y tiene que seguir pasando. Entre
`test_retranscribir_sin_vad.py` (12), `test_novad_alucinaciones.py` (20) y
`test_novad_avisos.py` (12), 44 tests.

**Los avisos, que no son veredictos.** Los resultados no se parten en limpios y
basura: hay transcripts que son mitad habla real y mitad bucle del decoder. En
`wa_audio_00029933`, `"¡Vaya que se cae!"` repetido 9 veces cubre el 49 % del
texto, por debajo del 60 % que hace falta para descartar — y el veredicto
`REVISAR` es el correcto, pero no decía dónde mirar. Bajar el umbral para
atrapar ese caso sería calibrar sobre un archivo; `avisos()` en cambio **anota
sin cambiar el veredicto**, con umbrales más bajos (`AVISO_COBERTURA = 0.25`,
`AVISO_INVENTADO = 0.1`):

```
wa_audio_00029933-Parque Villa Biarritz
    REVISAR: +264 chars contra la version con VAD
    ojo: '¡vaya que se cae!' repetido 9 veces cubre el 49% del texto
    ojo: el 20% del texto son frases que Whisper inventa
```

La propiedad que los hace seguros de agregar está en un test
(`test_los_avisos_no_cambian_el_veredicto`): **un aviso de más cuesta una línea
en el reporte, un veredicto de más descarta trabajo de GPU sin que nadie lo
haya leído.** Separan bien sobre los datos reales: de los 6 resultados a
revisar, los 3 que parecen habla de verdad no juntan ninguno.

**El detector corrido sobre el corpus vivo.** Que el detector exista para la
auditoría sin VAD no significa que el corpus con VAD esté limpio: el transcript
**con VAD** de `wa_audio_00021896` son 1.907 caracteres de `"¡Gracias por ver
el video!"` repetido 25 veces. Corrido sobre los 8.838 transcripts vivos (solo
lectura, sin tocar nada):

| | |
|---|---|
| Marcados como alucinación | **4**, 36 caracteres en total |
| De esos, con items en el índice | **0** |
| Con aviso pero no marcados | **36** |

Los 4 marcados son transcripts de 8 a 12 caracteres (`"Gracias."` tres veces,
`"¡Suscríbete!"` una): ruido que el VAD dejó pasar.

**Un transcript alucinado no llega al índice por sí solo**, y eso cambia qué
hay que leer del informe. Las 3 colecciones se arman desde `extracted/`: un
transcript entra al índice únicamente a través de los items que el extractor
sacó de él. Los 4 marcados extrajeron **0 items** — el extractor ya los
filtró, no aportaron un solo vector, no hay nada que sacar. Por eso el reporte
imprime los items al lado de cada marcado, y `0 items` no es lo mismo que `sin
extraer`: lo segundo significa que la etapa todavía no pasó por ahí y que la
respuesta no se sabe. `wa_audio_00021896` está hoy en ese estado, porque su
extracción se movió a `_reextraer/` y la corrida con Ollama todavía no llegó.

De los 36 avisados, **uno solo junta las dos señales**: `wa_audio_00021896`,
33 % de cobertura y 33 % de frases inventadas. Veinte de los 36 tienen items
extraídos, y eso no los vuelve basura: son notas de voz reales en las que
alguien repite una frase. El aviso es una anotación, no un veredicto. Los otros 35 son notas de voz
cortas donde una muletilla repetida tres o cuatro veces cubre más del 25 % del
texto justamente porque el texto es corto — `"qué bonito"` seis veces en 256
caracteres es alguien hablando, no el decoder en bucle. Eso acota para qué
sirve el aviso de cobertura: **en un texto corto cualquier repetición cubre
mucho**, así que ahí solo vale leído junto al otro.

**Sacar o re-transcribir cualquier transcript vivo cambia lo que devuelve el
índice, así que lo decide Ilan**, igual que la promoción de un resultado sin
VAD. El escaneo vive en `scripts/escanear_corpus.py`, importa el criterio de la
auditoría en vez de copiarlo, y **no escribe nada**, así que se puede correr
cuando sea. 10 tests, ninguno sobre la detección: sobre el recorrido (que no
escriba, que un transcript vacío no cuente como alucinación, que un JSON roto
no corte la corrida) y sobre el conteo de items (que `sin extraer` no se cuente
como 0, que una extracción ilegible tampoco).

**Es el paso 6 de `finalize.ps1`**, de solo lectura y sin abortar. Está ahí y no
solo documentado porque un informe que nadie corre no existe: es el mismo
defecto que tenía el staging sin VAD, que era seguro justamente porque nada lo
leía. Va después de `--juzgar` y antes del re-index, que es el momento en que
una persona decide a mano qué entra y qué sale del índice.

```powershell
.venv\Scripts\python.exe -I -X utf8 scripts/retranscribir_sin_vad.py --listar
.venv\Scripts\python.exe -I -X utf8 scripts/retranscribir_sin_vad.py          # transcribe
.venv\Scripts\python.exe -I -X utf8 scripts/retranscribir_sin_vad.py --juzgar # informa qué hay
.venv\Scripts\python.exe -I -X utf8 scripts/escanear_corpus.py                # el corpus vivo
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
6. **Juzgar la auditoría sin VAD** con
   `scripts/retranscribir_sin_vad.py --juzgar` y leer uno por uno los que caen en
   `A REVISAR`. Promover alguno es decisión de Ilan (ver "Decisiones pendientes",
   punto 7); si se promueve, el `--reset` del paso 3 tiene que correr **después**.

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

La auditoría a escala la hace `scripts/retranscribir_sin_vad.py`, que deriva
a quién re-transcribir de los umbrales de `verificar.py`, escribe en un
directorio de staging que ninguna etapa lee, y marca la alucinación en vez de
dejar que alguien la lea a ojo. Ver "La auditoría sin VAD".

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

### 26. Un `source_path` muerto no rompe nada hasta que algo relee la fuente

Después de reorganizar el material dentro de `Seduccion\Bonus\`, 224
transcripts quedaron apuntando a rutas que ya no existen. Durante semanas eso no
produjo **un solo error**: el pipeline estaba terminado y ninguna etapa volvía a
abrir el archivo original. El inventario tampoco se quejó, porque matchea por
nombre y por duración sobre el árbol real.

El defecto apareció recién al querer re-transcribir sin VAD un puñado de audios
sospechosos: "el archivo no existe", sobre archivos que están en disco. Dos
lecciones que valen para el resto del proyecto:

- **Una ruta absoluta guardada en un artefacto es una dependencia con fecha de
  vencimiento.** Sobrevive mientras nadie mueva nada, y falla en silencio
  después. Los transcripts ahora guardan la ruta vieja en
  `metadata.source_path_previo` cuando se los repara, así que el rastro queda.
- **Un corpus puede estar completo y roto a la vez.** "No falta contenido" y
  "las etapas pueden volver a correr" son dos propiedades distintas, y el
  verificador medía solo la primera.

Lo que queda sin fuente después de reparar **no es necesariamente una pérdida**.
Antes de declarar un hueco hay que comparar contenido: los 5 restantes son
duplicados exactos de transcripts cuyo archivo sí está en disco. El criterio de
duplicado vive en un solo lugar, `scripts/verificar.py:es_duplicado`, y lo
importan `manifest.py` y `reparar_source_path.py` — tener la misma tolerancia
escrita en tres archivos termina en tres herramientas que contestan distinto
sobre el mismo par.

El criterio pide **arranque idéntico (200 caracteres) y además largo igual o
duración igual**. Los dos pedazos hacen falta:

- El arranque solo no alcanza: dos notas de voz distintas pueden empezar con la
  misma muletilla. Agrupando por duración redondeada salen 159 grupos y casi
  todos son casualidad.
- El largo solo tampoco: dos pasadas de Whisper sobre el mismo video dan la
  duración del contenedor idéntica al microsegundo (5948,7201875 s) pero largos
  que difieren en miles de caracteres, porque el decodificado no es
  determinista. Dos PDF idénticos, en cambio, dan el largo exacto y no tienen
  duración.

### 27. `os.walk` sobre una ruta que no se puede recorrer devuelve cero archivos

No levanta excepción: no encuentra nada. `scripts/manifest.py` recorre el
material así, y desde que es el paso 8 de `finalize.ps1` corre **solo**,
sobrescribiendo `INVENTARIO.md`. Con las junctions de `data/raw` apuntando a
una carpeta movida o a un disco desconectado, ese paso escribiría un
inventario que dice que en disco no hay material — exactamente lo que alguien
querría leer antes de borrar 360 GB.

El caso parcial es el más fácil de no notar: si sobrevive la junction de
`videos` y se cae la de `whatsapp`, el recorrido trae unos cientos de archivos
en vez de 17.801. No es cero, así que un chequeo de "vino algo" no lo atrapa.

`exigir_piso()` corta antes de escribir nada, y el piso sale de un invariante,
no de una calibración: **cada transcript lo produjo un archivo fuente, así que
en disco no puede haber menos archivos que transcripts.** Hoy son 17.801 contra
8.838, así que el margen es amplio y solo salta cuando algo se rompió de
verdad. `--sin-piso` lo desactiva, para el caso en que el material se haya
borrado a propósito y se quiera un inventario que lo refleje. 7 tests.

### 28. Una métrica puede medir dos cosas y parecer que mide una

El detector de alucinaciones tenía los dos defectos de esta clase, los dos
encontrados leyendo resultados reales y no el código.

**La frase inventada pegada a otra frase.** `_resto_sin_frases_inventadas`
sacaba una frase solo cuando la frase **entera** era una alucinación conocida.
Whisper muchas veces las escupe sin puntuación, así que se fusionan con lo que
tienen al lado y dejan de ser una frase entera: `"paloma dahi paloma de tennis
gracias por ver el video"` contaba sus 53 caracteres como habla real. Ahora se
sacan como subcadena, con límites de palabra — sin `\b`, `"fin"` (que está en
la lista) saldría de "definitivamente", "final" y "afinar", y las tres
aparecen en el habla real de este corpus.

**El denominador medía otra cosa que el numerador.** `total` se tomaba sobre el
texto entero y `queda` sobre las frases ya partidas por `[.!?\n]+`. El split se
come sus separadores y `_normalizar` recorta la puntuación de borde de cada
frase, así que la diferencia entre los dos **no eran frases inventadas: era
puntuación**. El sesgo crece con el largo y con cuánto se puntúe, no con cuánta
alucinación haya: `pdf_RSD Max- the natural`, 95.941 caracteres, informaba "el
11 % del texto son frases que Whisper inventa" teniendo adentro seis frases de
la lista, unos 30 caracteres. Casi todos los avisos sobre textos largos eran
ese artefacto. El veredicto nunca cambió de sentido, porque el sesgo bajaba el
resto y el resto bajo es lo que descarta: descartaba de más, nunca de menos.

La lección es la misma en los dos: **una fracción solo significa algo si arriba
y abajo se miden sobre el mismo texto**, y una comparación de igualdad solo
detecta lo que viene exactamente igual.

## Costos reales

| Etapa | Costo |
|---|---|
| Transcripción (8838 fuentes, Whisper local GPU) | $0 |
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
│   │   ├── transcripts/      ← 8838 JSON unificados
│   │   ├── extracted/        ← 8535 .extracted.json (+303 en curso)
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
├── scripts/
│   ├── finalize.ps1          ← los 9 pasos que faltan, en orden; se niega a correr con el pipeline vivo
│   ├── verificar.py          ← el verificador: huecos, duplicados, pendientes sin VAD
│   ├── manifest.py           ← escribe INVENTARIO.md y el CSV; aborta si el disco no se pudo recorrer
│   ├── retranscribir_sin_vad.py  ← la auditoria sin VAD y el detector de alucinaciones
│   ├── escanear_corpus.py    ← pasa ese detector por el corpus vivo; solo lee
│   ├── reparar_source_path.py    ← reescribe los source_path muertos (gotcha #26)
│   └── resume, status, watch, transcribe_batches, smoke_test, upload_to_qdrant_cloud
├── tests/                    ← 210 tests en 17 archivos; `python -m pytest tests/ -q`
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
5. **Tests** — ya no está vacío: **210 tests pasando** en 17 archivos
   (`.venv\Scripts\python.exe -m pytest tests/ -q`). Cubren los umbrales del
   verificador, los cuatro motivos de "sin cubrir", el criterio de duplicado, a quién
   manda a re-transcribir la auditoría sin VAD, su detector de alucinaciones con sus
   avisos, el piso que protege a `INVENTARIO.md` de un recorrido fallido y el escaneo
   del corpus vivo. Desde el 18-11 también las dos etapas que deciden qué llega al
   índice: que el chunking reparta el texto sin perder un carácter, que
   `_extract_json()` recupere las respuestas sucias del modelo (fences, prosa
   alrededor, coma colgante, cortadas por `max_tokens`), que el salteo resumible sea
   por **existencia** del archivo y no por contenido (gotcha #13) y que una extracción
   con 0 items no escriba un solo punto en Qdrant — la frase de la que depende el
   informe del paso 6. Y el camino de la consulta, que es el que el usuario toca en
   cada pregunta: que el umbral de score llegue a `query_points` (sin él la búsqueda
   devuelve los principios menos lejanos para cualquier pregunta, y el system prompt
   le dice al modelo que lo recuperado le gana a su criterio), que un resultado vacío
   se vea como `(ninguno recuperado)` en vez de un bloque en blanco, que una
   colección caída no se lleve las otras dos, que el modelo se pueda pisar con
   `COACH_MODEL_<PROVEEDOR>` cuando el proveedor lo deprecia, que el `token_floor`
   levante un `max_tokens` chico (los modelos con thinking devuelven contenido vacío,
   no un error) y que `complete_fast()` nunca levante: devuelve `""` y la tarea chica
   no corta la respuesta principal. Corren con un doble en lugar de Ollama, de Qdrant
   y de `httpx`, así que no piden GPU, ni abren el índice, ni hacen una llamada de
   red. Lo que sigue **sin una sola prueba** es la transcripción: necesita GPU y un
   archivo de audio de verdad.
6. **Los 9 grupos de contenido duplicado** — el mismo material entró al pipeline dos
   veces con nombres distintos (4 videos de clase, 3 libros, 2 notas de voz). No falta
   nada: sobra texto repetido en el corpus y en el índice, y eso sesga la búsqueda porque
   el mismo pasaje ocupa dos lugares entre los resultados más parecidos.
   `scripts/verificar.py` los lista en cada corrida e `INVENTARIO.md` nombra los dos
   gemelos de cada par. **Sacar uno de los dos borra datos, así que no se tocó.** Si se
   decide sacarlos hay que re-indexar con `--reset` para que el índice deje de
   arrastrarlos.
7. **Los resultados de la auditoría sin VAD** — están en `data/processed/_novad/`, que
   no es un directorio que lea ninguna etapa. El detector los separa en `REVISAR` y
   `DESCARTAR` (`--juzgar`): de los 22 ya re-transcriptos, **7 a revisar y 15 a
   descartar** (quedan 3 corriendo). Promover uno es copiarlo a
   `data/processed/transcripts/` y re-indexar con `--reset`, y eso cambia lo que
   devuelve la búsqueda: **lo decide Ilan después de leer el texto.** Los tres que
   recuperan texto de verdad son `wa_audio_00032678` (+6.189 caracteres contra la
   versión con VAD), `wa_audio_00021896` (+3.391 — es el del punto 8) y
   `wa_audio_00021741` (+1.352, diálogo de campo real); los otros cuatro suman menos de
   600 caracteres cada uno.
8. **`wa_audio_00021896`, el mismo archivo por los dos lados** — su transcript **vivo**
   son 1.907 caracteres de `"¡Gracias por ver el video!"` repetido 25 veces, el único de
   los 8.838 que junta las dos señales del detector; y su re-transcripción **sin VAD**
   son 5.298 caracteres de diálogo de campo real, sin una sola marca ni aviso. O sea que
   las dos decisiones son una sola: lo que está en el corpus es basura, y el reemplazo ya
   está transcripto, medido y esperando en `_novad/`. **Hoy no aporta nada al índice**:
   su extracción se movió a `_reextraer/` y la corrida con Ollama todavía no llegó, así
   que el paso 6 de `finalize.ps1` lo imprime como `sin extraer`. Si cuando llegue saca
   items, los va a sacar del texto alucinado. Promover la versión sin VAD y re-indexar
   lo decide Ilan.
9. **Bot de WhatsApp** — sigue pendiente. Hoy la entrada es screenshot/texto en la web.
10. **Fine-tuning** — descartado hasta tener >500 conversaciones etiquetadas.

---

## Cómo retomar (handoff para nueva sesión Claude)

1. `cd C:\Users\Ilan\Desktop\Proyectos\coach-brain`
2. `claude`
3. Mensaje inicial sugerido: *"Leé STATUS.md, después seguimos desde el próximo paso."*

Las memorias del proyecto están en `~\.claude\projects\C--Users-Ilan-Desktop-Proyectos-coach-brain\memory\`
y se cargan automáticamente.
