# INVENTARIO de fuentes

Generado por `scripts/manifest.py`. El detalle fila por fila esta en
`data/processed/manifest_fuentes.csv`, una fila por archivo.

**Total en disco:** 17799 archivos, 360.50 GB

| Tipo | Archivos | Tamano | Cubiertos | Tamano cubierto |
|---|---|---|---|---|
| audio | 8396 | 1.32 GB | 8396 | 1.32 GB |
| doc | 93 | 0.30 GB | 91 | 0.30 GB |
| imagen | 8964 | 0.98 GB | 8964 | 0.98 GB |
| video | 346 | 357.91 GB | 342 | 353.56 GB |

## Archivos SIN cubrir

Estos no tienen transcript ni OCR. Si se borra el material crudo, su
contenido se pierde y no hay forma de recuperarlo.

**6 archivos, 4.34 GB**

- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\docs\78 que es el valor.mp4` (3.43 GB, video)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\docs\Bonus\4 Hablarle a una chica del trabajo, sexualiza.mp4` (0.91 GB, video)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00031653-VIDEO-2025-09-08-11-54-41.mp4` (0.00 GB, video)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00033588-GIF-2025-10-03-00-26-43.mp4` (0.00 GB, video)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00021049-CV_divertido_Ilan_Daniele.pdf` (0.00 GB, doc)
- `C:\Users\Ilan\Desktop\Proyectos\coach-brain\data\raw\whatsapp\00021052-CV_divertido_Ilan_Daniele_coqueto.pdf` (0.00 GB, doc)

## Con transcript pero sin extraccion

300 archivos. El texto ya esta guardado en `data/processed/transcripts/`;
falta destilarlo. Para eso no hace falta el archivo original.

## Con extraccion vacia (0 items)

2896 archivos devolvieron 0 principios, 0 situaciones y 0 frases.
En audios cortos eso suele ser legitimo. Re-extraer no necesita el
archivo original, solo el transcript.

## Transcripts cuya fuente ya no esta en disco

6 transcripts. El texto esta a salvo; el original no.
- `pdf_21-Claves-Sexualizar-Convencer (Luis Tejedor (Egoland))-2.json` - fuente no esta en disco: data/raw/pdfs/Libros de Ganso a Gavilan/Seduccion/21-Claves-Sexualizar-Convencer (Luis Tejedor (Egoland))-2.pdf
- `video_14 juego de texto-056.json` - fuente no esta en disco: data/raw/videos/14 juego de texto-056.mp4
- `video_36 Clase vip-029.json` - fuente no esta en disco: data/raw/videos/Clases VIP/36 Clase vip-029.mp4
- `video_38 Clase Vip-014.json` - fuente no esta en disco: data/raw/videos/Clases VIP/38 Clase Vip-014.mp4
- `video_39 Clase vip-015.json` - fuente no esta en disco: data/raw/videos/Clases VIP/39 Clase vip-015.mp4
- `video_78 que es el valor-008.json` - fuente no esta en disco: data/raw/videos/78 que es el valor-008.mp4

