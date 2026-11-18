# Borrar el material crudo: qué se puede y qué no

El material de origen ocupa unos 360 GB en `C:\Users\Ilan\Desktop\Ilan\Seduccion`.
El conocimiento ya destilado ocupa unos 670 MB y vive en el repo:

| Qué | Dónde | Tamaño |
|---|---|---|
| Texto completo de cada fuente | `data/processed/transcripts/` | 197 MB |
| Principios, situaciones y frases | `data/processed/extracted/` | 40 MB |
| OCR de las capturas de WhatsApp | `data/processed/whatsapp_images_ocr/` | 34 MB |
| Índice vectorial que usa la app | `qdrant_storage/` | 399 MB |

La app en producción solo necesita `qdrant_storage/`, que ya va horneado dentro
de la imagen de Docker. Los transcripts y las extracciones son el insumo para
reconstruir ese índice.

Las cifras archivo por archivo están en [INVENTARIO.md](INVENTARIO.md) y, con
una fila por archivo, en `data/processed/manifest_fuentes.csv`. Ese CSV se
regenera cuando quieras con:

```bash
.venv\Scripts\python.exe scripts/manifest.py
```

## Antes de borrar nada

Hay cuatro cosas pendientes que todavía necesitan el material crudo. Hasta que
las cuatro estén hechas y verificadas, no conviene borrar.

1. **La extracción que está corriendo ahora.** Está re-destilando 301
   fuentes cuya extracción salió vacía porque Anthropic se quedó sin crédito
   a mitad del pipeline original: los 30 PDFs grandes, los 12 videos, el
   transcript agregado de las capturas de WhatsApp, y 244 grabaciones de
   campo que llegaron como notas de voz de WhatsApp y que al revisarlas una
   por una resultaron ser material real, no charla personal (el umbral usado
   fue 1200 caracteres de transcript; por debajo de eso son notas de voz
   cortas sin nada que destilar). Son unos 4062 chunks a 43,4 segundos cada
   uno: unas 49 horas con Ollama local. Esta etapa trabaja sobre los
   transcripts, no sobre los archivos originales, así que en rigor no
   necesita el material crudo — pero su resultado es lo que hay que verificar
   antes de dar el corpus por cerrado.
2. **El índice.** Cuando la extracción termine hay que correr
   `python -m coach_brain.kb.index --reset`, que borra y reconstruye las tres
   colecciones de Qdrant. Hay una copia de seguridad del índice que funciona
   hoy en `qdrant_storage.bak/`.
3. **El PDF escaneado.** `Método TED para hablar en público ... (Jeremey
   Donovan) (z-lib.org).pdf` no tiene texto extraíble: es un escaneo. Para
   incorporarlo hace falta pasarle OCR, y para eso hace falta el PDF original.
   Si se borra, ese libro queda afuera para siempre. **Ya está resuelto:** se
   le agregó a `ingest/pdfs.py` una tercera pasada que rasteriza cada página
   con pymupdf y le aplica RapidOCR, detrás del flag `--ocr` para que el costo
   de CPU sea opcional. El libro salió completo: 141.340 caracteres, 59
   páginas, `engine=rapidocr`, 26 minutos de CPU. La separación de palabras en
   la tapa quedó imperfecta ("Lossecretos"), pero el cuerpo del texto está
   bien.

4. **Lo que estaba escondido y se rescató el 17/11.** Tres archivos figuraban
   como cubiertos sin estarlo de verdad, y los tres necesitaron el original
   para arreglarse. Ya está hecho, pero vale saber qué tipo de hueco fue,
   porque es el que no se ve:
   - `data/raw/whatsapp/00025319-cliente ideal, canva,.zip`. El inventario
     descartaba en silencio cualquier extensión que no estuviera en
     `MEDIA_EXT`, así que el zip no aparecía en ninguna parte. Adentro había
     `cliente ideal 1.2.pptx` y `canvas 1.3.pdf`, los dos con el canvas de
     negocio del coach. Se extrajo a
     `data/raw/whatsapp/00025319-cliente ideal canva/` y los dos entraron al
     pipeline. El inventario ahora lista los formatos que no clasifica, así
     que un caso así vuelve a ser visible.
   - `canvas 1.3.pdf`. Una exportación de Canva: traía 510 caracteres de capa
     de texto en 2 páginas y todo el contenido en la imagen. Con OCR pasó a
     3.506 caracteres, y lo que faltaba era el canvas entero (segmentos de
     clientes, dolores, propuesta única de valor).
   - `El Arte Del Buen Sexo by Academia Para Caballeros (z-lib.org).pdf`. Un
     libro escaneado de 132 páginas que estaba registrado con 796 caracteres
     en total. El OCR no lo alcanzaba porque solo se disparaba con cero texto
     extraído.

   Lo que quedó afuera a propósito, por si aparece en el inventario: dos
   `.vcf` (tarjetas de contacto de WhatsApp) y un `Diagrama sin título.drawio`
   de 15 KB que trae el mismo diagrama que ya está en el `.pptx`.

## Qué se pierde para siempre si borrás el material crudo

Esto es irreversible y no hay copia accesible en Drive: la descarga de Drive
falla (ver el gotcha 6 de [STATUS.md](STATUS.md)). Lo que se pierde:

- **La posibilidad de re-transcribir con un modelo mejor.** Los transcripts
  actuales los hizo faster-whisper large-v3. Si en dos años hay un modelo
  que transcribe mejor el acento argentino o separa mejor las voces, sin los
  audios y videos originales no se puede aprovechar. Esta es la pérdida más
  grande y la menos visible hoy.
- **El OCR del PDF escaneado** del punto 3 de arriba.
- **Los tres archivos que fallaron al transcribir** y todavía no tienen texto:
  `4 Hablarle a una chica del trabajo, sexualiza.mp4`,
  `00031653-VIDEO-2025-09-08-11-54-41.mp4` y
  `00033588-GIF-2025-10-03-00-26-43.mp4` (este último es un GIF convertido a
  mp4, no tiene pista de audio y nunca va a dar texto). Sobre el primero hay
  una corrección importante respecto de lo que decía esta nota antes: no es un
  problema de ffmpeg que se arregle re-transcribiendo. Se escanearon sus
  977.018.880 bytes completos y el archivo **no tiene átomo `moov` ni `moof`
  en ninguna parte** — tiene `ftyp` en el byte 4 y `mdat` en el 9587508, y nada
  más. Es una descarga truncada, sin índice, y por eso es indecodificable. La
  única vía de recuperación realista es reconstruirle el índice con `untrunc`
  usando otra clase del mismo curso como referencia. Eso se intentó y quedó
  bloqueado, pero no por falta de permiso tuyo: el clasificador de permisos
  de Claude rechaza cualquier `docker run`, con o sin montaje de volumen. La
  imagen correcta en Docker Hub es `zany/untrunc` (el fork de anthwlock; su
  entrypoint es `/bin/untrunc`); `anthwlock/untrunc` es el nombre del repo de
  GitHub y no existe como imagen. Lo tenés que correr vos, con el comando que
  está más abajo.
- **Todo archivo que figure como `cubierto = NO`** en el CSV. Esos no pasaron
  por transcripción ni por OCR: su contenido no está en ninguna parte del
  repo.
- **La imagen y el video en sí.** Los transcripts guardan lo que se dijo, no
  lo que se vio. Cualquier cosa que esté en lo visual — un gesto, algo escrito
  en una pizarra, una demostración — no quedó registrada.

Lo que **no** se pierde: el texto de todo lo ya transcripto, las extracciones,
el OCR de las capturas y el índice que usa la app. Nada de eso necesita el
material crudo para seguir funcionando ni para reconstruirse.

## Los 6 archivos sin cubrir, uno por uno

Después de cerrar los huecos evitables, de 17.799 archivos y 360,50 GB solo
quedan 6 archivos (4,34 GB) sin transcript ni OCR, y ninguno de los 6 es una
pérdida nueva. Las imágenes quedaron en 8.964 de 8.964 cubiertas y los audios
en 8.396 de 8.396. Los 6 son:

**1. Tres fallas de transcripción ya conocidas.**
`4 Hablarle a una chica del trabajo, sexualiza.mp4` (0,91 GB) no tiene átomo
`moov`: es una descarga truncada y no se puede decodificar (ver arriba).
`00031653-VIDEO-2025-09-08-11-54-41.mp4` y
`00033588-GIF-2025-10-03-00-26-43.mp4` son clips de WhatsApp; el segundo es un
GIF sin pista de audio.

**2. Un video de 3,43 GB que en realidad sí está cubierto.**
`data\raw\docs\78 que es el valor.mp4` no matchea ningún transcript por
nombre, pero existe `video_78 que es el valor-008.json`, hecho sobre
`78 que es el valor-008.mp4`: es la misma clase en otra copia, con otro
sufijo de descarga. El contenido está. Lo mismo pasa al revés con los 6
transcripts "huérfanos" que lista `INVENTARIO.md`: son versiones viejas de
clases que después se volvieron a descargar con otro sufijo
(`36 Clase vip-029` contra `36 Clase vip-095` en disco, por ejemplo). No son
una pérdida.

**3. Los dos CVs.** `00021049-CV_divertido_Ilan_Daniele.pdf` y
`00021052-CV_divertido_Ilan_Daniele_coqueto.pdf` figuran sin cubrir a
propósito: sí se procesaron, pero son ruido para el corpus y sus transcripts
se movieron a `data/processed/_fuera_del_corpus/`. Están ahí si los querés de
vuelta.

El `Método TED`, que antes estaba en esta lista, ya salió: se le pasó OCR y
tiene transcript. Las 25 capturas que esta nota listaba como grupo 5 tampoco
están más: pasaron por OCR y el inventario las reportaba mal por un bug de
matcheo en `scripts/manifest.py`, ya arreglado (gotcha #18 de STATUS.md).

**4. Seis PDFs que llegaron por WhatsApp y nunca se procesaron. YA RESUELTO.**
Estaban en `data\raw\whatsapp\` y la etapa de PDFs no miraba esa carpeta:
`00003796-Mode One (Alan Roger Currie -.pdf`,
`00005134-174881311-Frases-Push-and-Pull.pdf`,
`00005244-36 preguntas para generar confort y enamorar.pdf`,
`00004206-Ejercicion 6 dias para Ilan.pdf` y dos CVs
(`00021049-CV_divertido_Ilan_Daniele.pdf`,
`00021052-CV_divertido_Ilan_Daniele_coqueto.pdf`).

Se arregló la causa en `src/coach_brain/ingest/pdfs.py`, que ahora también
recorre `data/raw/whatsapp` y deduplica por path canónico. Los seis entraron
(`ok=6 skip=68 sin_texto=1 errores=0`). Los cuatro útiles aportaron
227.045 caracteres (`Mode One`, 115 páginas), 70.987 (`Frases-Push-and-Pull`,
40 páginas), 2.763 (`36 preguntas para generar confort y enamorar`) y 2.346
(`Ejercicion 6 dias para Ilan`). Los dos CVs son ruido para el corpus, así que
sus transcripts se movieron a `data/processed/_fuera_del_corpus/`; están ahí si
los querés de vuelta.

Ojo con un detalle: la etapa de PDFs solo saltea un PDF cuando su transcript
existe, así que al correr `ingest.pdfs` de nuevo los dos CVs se vuelven a
generar. Si vuelven a aparecer en `data/processed/transcripts/`, moverlos otra
vez a `_fuera_del_corpus/`.

**5. Veinticinco capturas de conversaciones sin OCR. YA RESUELTO.**
Estaban bajo `data\raw\docs\Bonus\Interacciones\Mensaje\` (subcarpetas
`Capturas\Nueva carpeta*`, `Melina\`, `maria contadora ilegal\`). La etapa de
OCR solo recorría `data/raw/whatsapp`, así que quedaban afuera.

Se arregló en `src/coach_brain/ingest/whatsapp_images_ocr.py`, que ahora
recorre además `data/raw/docs` y `data/raw/screenshots`. Las 25 pasaron por OCR
sin errores: 13 resultaron chats reales y 12 imágenes con poco texto.
`aggregate` rehizo `transcripts/whatsapp_images_chats.json`, que ahora tiene
6.336 chats sobre 8.978 imágenes. Como ese transcript ya tenía extracción
hecha, su archivo se movió a `data/processed/_reextraer/` para que la etapa de
extracción no lo saltee y lo vuelva a destilar con las capturas nuevas.

Los grupos 4 y 5 eran la única pérdida evitable que valía la pena atender antes
de borrar, y ya están cerrados.

## Orden recomendado

1. Esperar a que termine la extracción y verificar que no quedaron
   extracciones vacías que no deberían estarlo.
2. Correr `python -m coach_brain.kb.index --reset` y verificar el conteo de
   vectores en `logs/index.log`.
3. Correr `flyctl deploy` y probar la app en producción.
4. ~~Procesar los PDFs de WhatsApp y las 25 capturas sin OCR.~~ Hecho.
5. ~~Pasarle OCR al PDF escaneado `Método TED`.~~ Hecho. Su transcript entró
   después de que la extracción en curso armara su lista de archivos, así que
   hace falta una segunda pasada de `kb.extract` para destilarlo (son unos 31
   chunks).
6. Recuperar `4 Hablarle a una chica del trabajo, sexualiza.mp4` con `untrunc`,
   o aceptar que esa clase queda afuera. Re-transcribirlo no es una opción: no
   tiene átomo `moov`. Esto lo tenés que correr vos; Claude no puede ejecutar
   `docker run`. Los dos archivos ya están preparados como hardlinks (no
   ocupan espacio extra) en el directorio de trabajo de la sesión:

   ```
   docker run --rm -v "C:/Users/Ilan/AppData/Local/Temp/claude/761b3e6a4181af87/cf70a6bc-7876-4422-a8de-f326e9711069/scratchpad/untrunc:/w" zany/untrunc /w/ref.mp4 /w/roto.mp4
   ```

   `ref.mp4` es `75 Reglas de un seductor.mp4` (sano, mismo encoder) y
   `roto.mp4` es el archivo truncado. Si sale bien deja un
   `roto_fixed.mp4` al lado. Ojo: el scratchpad de la sesión se puede limpiar,
   así que si los hardlinks no están, rehacelos con `mklink /H` antes de
   correr el comando.
7. Si se procesó algo nuevo en los pasos 4 a 6, volver a correr la extracción
   y el índice.
8. Regenerar el inventario (`scripts/manifest.py`) y confirmar que la lista de
   `cubierto = NO` quedó solo con lo que decidiste dejar afuera.
9. Recién entonces borrar.

## Cómo borrar, si llega el momento

`data/raw/` contiene junctions de NTFS que apuntan a
`C:\Users\Ilan\Desktop\Ilan\Seduccion`. Esto importa y es fácil de arruinar:

**`Remove-Item -Recurse` sobre una junction en PowerShell 5.1 puede entrar en
la junction y borrar el contenido del destino.** Es decir, borrar lo que
parece un atajo dentro del repo puede borrar los 360 GB reales.

Para quitar una junction sin tocar su destino, usá `cmd /c rmdir "<junction>"`,
que elimina solamente el enlace.

Para borrar el material de verdad, borrá la carpeta real
`C:\Users\Ilan\Desktop\Ilan\Seduccion` directamente, y recién después quitá
las junctions que quedaron apuntando a la nada.

Antes de eso, considerá mover los videos a un disco externo en lugar de
borrarlos: 360 GB entran en cualquier disco de 1 TB, y eso conserva la opción
de re-transcribir más adelante.
