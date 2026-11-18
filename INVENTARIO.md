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

Estos no tienen transcript ni OCR. Si se borra el material crudo, su
contenido se pierde y no hay forma de recuperarlo.

**5 archivos, 0.91 GB**

(1 archivo(s) mas matchearon por duracion: estan renombrados en disco respecto del `source_path` del transcript, pero su contenido YA esta en el corpus. Columna `via` del CSV.)

- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\docs\Bonus\4 Hablarle a una chica del trabajo, sexualiza.mp4` (0.91 GB, video)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00031653-VIDEO-2025-09-08-11-54-41.mp4` (0.00 GB, video)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00033588-GIF-2025-10-03-00-26-43.mp4` (0.00 GB, video)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00021049-CV_divertido_Ilan_Daniele.pdf` (0.00 GB, doc)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00021052-CV_divertido_Ilan_Daniele_coqueto.pdf` (0.00 GB, doc)

## Con transcript pero sin extraccion

299 archivos. El texto ya esta guardado en `data/processed/transcripts/`;
falta destilarlo. Para eso no hace falta el archivo original.

## Con extraccion vacia (0 items)

2897 archivos devolvieron 0 principios, 0 situaciones y 0 frases.
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

5 transcripts. El texto esta a salvo; el original no.
- `pdf_21-Claves-Sexualizar-Convencer (Luis Tejedor (Egoland))-2.json` - fuente no esta en disco: data/raw/pdfs/Libros de Ganso a Gavilan/Seduccion/21-Claves-Sexualizar-Convencer (Luis Tejedor (Egoland))-2.pdf
- `video_14 juego de texto-056.json` - fuente no esta en disco: data/raw/videos/14 juego de texto-056.mp4
- `video_36 Clase vip-029.json` - fuente no esta en disco: data/raw/videos/Clases VIP/36 Clase vip-029.mp4
- `video_38 Clase Vip-014.json` - fuente no esta en disco: data/raw/videos/Clases VIP/38 Clase Vip-014.mp4
- `video_39 Clase vip-015.json` - fuente no esta en disco: data/raw/videos/Clases VIP/39 Clase vip-015.mp4

