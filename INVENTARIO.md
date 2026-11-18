# INVENTARIO de fuentes

Generado por `scripts/manifest.py`. El detalle fila por fila esta en
`data/processed/manifest_fuentes.csv`, una fila por archivo.

**Total en disco:** 17801 archivos, 360.50 GB

| Tipo | Archivos | Tamano | Cubiertos | Tamano cubierto |
|---|---|---|---|---|
| audio | 8396 | 1.32 GB | 8396 | 1.32 GB |
| doc | 95 | 0.30 GB | 93 | 0.30 GB |
| imagen | 8964 | 0.98 GB | 8964 | 0.98 GB |
| video | 346 | 357.91 GB | 343 | 356.99 GB |

## Archivos SIN cubrir

Estos no tienen transcript ni OCR. El motivo de cada uno esta al lado,
y es lo que decide si borrar el crudo pierde algo: un video sin pista de
audio no tiene habla que perder, y un transcript excluido a proposito
tampoco es una perdida. Solo los marcados `HUECO` son contenido que se
pierde sin forma de recuperarlo.

**5 archivos, 0.91 GB — de los cuales 1 es hueco real**

- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\docs\Bonus\4 Hablarle a una chica del trabajo, sexualiza.mp4` (0.91 GB, video) — HUECO: contenedor ilegible, ffprobe no lo puede abrir
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00031653-VIDEO-2025-09-08-11-54-41.mp4` (0.00 GB, video) — sin pista de audio: no hay habla que transcribir
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00033588-GIF-2025-10-03-00-26-43.mp4` (0.00 GB, video) — sin pista de audio: no hay habla que transcribir
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00021049-CV_divertido_Ilan_Daniele.pdf` (0.00 GB, doc) — excluido a proposito (transcript en _fuera_del_corpus/)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00021052-CV_divertido_Ilan_Daniele_coqueto.pdf` (0.00 GB, doc) — excluido a proposito (transcript en _fuera_del_corpus/)

## Con transcript pero sin extraccion

301 archivos. El texto ya esta guardado en `data/processed/transcripts/`;
falta destilarlo. Para eso no hace falta el archivo original.

## Con extraccion vacia (0 items)

2896 archivos devolvieron 0 principios, 0 situaciones y 0 frases.
En audios cortos eso suele ser legitimo. Re-extraer no necesita el
archivo original, solo el transcript.

## Formatos que el inventario no clasifica

El recorrido solo cuenta las extensiones de MEDIA_EXT. Lo que sigue
quedo afuera: puede ser basura, o puede ser material real en un formato
que el pipeline todavia no lee. Conviene revisarlo antes de borrar nada.

**5 archivos, 0.00 GB**

| Extension | Archivos | Tamano | Ejemplo |
|---|---|---|---|
| `.zip` | 1 | 0.00 GB | `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00025319-cliente ideal, canva,.zip` |
| `.drawio` | 1 | 0.00 GB | `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00025319-cliente ideal canva\Diagrama sin título.drawio` |
| `.vcf` | 2 | 0.00 GB | `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00000312-Gasto╠ün Menache ≡ƒç║≡ƒç╛.vcf` |
| `(sin extension)` | 1 | 0.00 GB | `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\.gitkeep` |

## Transcripts cuya fuente ya no esta en disco

5 transcripts cuyo `source_path` ya no resuelve.

De esos, **5 son duplicados**: la misma fuente entro al pipeline dos veces con nombres distintos y despues se borro una de las dos copias del archivo. El contenido sigue en disco bajo el otro nombre, asi que no hay nada que recuperar. Lo que hay es texto repetido en el corpus y en el indice, que sesga la busqueda; sacarlo borra datos y lo decide una persona.

- `pdf_21-Claves-Sexualizar-Convencer (Luis Tejedor (Egoland))-2.json` - fuente no esta en disco: data/raw/pdfs/Libros de Ganso a Gavilan/Seduccion/21-Claves-Sexualizar-Convencer (Luis Tejedor (Egoland))-2.pdf
  DUPLICADO de `pdf__21-claves-sexualisar-convenser.json`, cuyo archivo SI esta en disco: el contenido no se pierde, sobra texto repetido
- `video_14 juego de texto-056.json` - fuente no esta en disco: data/raw/videos/14 juego de texto-056.mp4
  DUPLICADO de `video_15 juego de texto-069.json`, cuyo archivo SI esta en disco: el contenido no se pierde, sobra texto repetido
- `video_36 Clase vip-029.json` - fuente no esta en disco: data/raw/videos/Clases VIP/36 Clase vip-029.mp4
  DUPLICADO de `video_36 Clase vip-095.json`, cuyo archivo SI esta en disco: el contenido no se pierde, sobra texto repetido
- `video_38 Clase Vip-014.json` - fuente no esta en disco: data/raw/videos/Clases VIP/38 Clase Vip-014.mp4
  DUPLICADO de `video_38 Clase Vip-092.json`, cuyo archivo SI esta en disco: el contenido no se pierde, sobra texto repetido
- `video_39 Clase vip-015.json` - fuente no esta en disco: data/raw/videos/Clases VIP/39 Clase vip-015.mp4
  DUPLICADO de `video_39 Clase vip-091.json`, cuyo archivo SI esta en disco: el contenido no se pierde, sobra texto repetido

