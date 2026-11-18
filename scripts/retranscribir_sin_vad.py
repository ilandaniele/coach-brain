"""Re-transcribe sin VAD los transcripts con densidad de texto sospechosa.

Para qué sirve
--------------
El filtro de voz de Whisper (`vad_filter=True`) decide por su cuenta qué partes
del audio son habla, y cuando se equivoca devuelve menos texto del que hay —
en el peor caso medido, 256 caracteres sobre 964 segundos de conversación real,
una pérdida del 97%. La etapa de transcripción no reintenta: el transcript
queda escrito, corto y plausible, y nada vuelve a mirarlo (gotcha 22).

Este script toma los transcripts que el verificador marca por densidad baja y
los vuelve a transcribir **sin** el filtro, para poder comparar.

Lo que sale de acá NO entra solo al corpus
------------------------------------------
Sin filtro de voz, Whisper alucina frases de su corpus de entrenamiento sobre
música y ruido ambiente. Un resultado real de esta misma corrida:

    con VAD:   60 chars  "¿Todo tranqui? ¿Cómo estás? ¿Qué haces por acá?"
    sin VAD:  179 chars  "La ciudad de Bogotá La ciudad de Bogotá La ciudad..."

Eso no es habla recuperada, es basura, y meterla al corpus sería peor que el
hueco. Así que cada resultado se guarda en `data/processed/_novad/`, que es
staging y no el corpus vivo, y **hay que leerlo** antes de promoverlo. El
transcript queda con `vad_filter=False` en el metadata, así que se identifica
solo si alguno se cuela.

Por qué la lista no está escrita a mano
---------------------------------------
La versión anterior de este script tenía los 14 nombres hardcodeados, y tres de
ellos traen mojibake horneado en el nombre del archivo: donde va una "u"
acentuada tienen literalmente U+2560 más U+00FC, que son los bytes UTF-8 `CC 81`
mal decodificados como cp437 en el momento en que se creó el archivo. Tipear
esos nombres no matchea nunca, ni copiándolos de la terminal.

Ahora la lista se deriva del umbral de `scripts/verificar.py`, que es el mismo
que decide qué es sospechoso en el resto del proyecto. Los nombres salen del
disco, así que el mojibake deja de ser un problema, y el día que el umbral se
recalibre esta lista se mueve con él en vez de quedarse vieja.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO / "data" / "processed" / "transcripts"
SALIDA = REPO / "data" / "processed" / "_novad"

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _umbrales():
    """Carga `verificar.py` para usar sus umbrales y su medición de volumen.

    Los umbrales están calibrados contra la distribución medida del corpus y
    tienen tests que los fijan. Repetirlos acá sería tener dos definiciones de
    "sospechoso" que se separan en la primera recalibración.
    """
    spec = importlib.util.spec_from_file_location(
        "verificar_umbral_densidad", Path(__file__).resolve().parent / "verificar.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_V = _umbrales()
MIN_CHARS_POR_MINUTO = _V.MIN_CHARS_POR_MINUTO
DUR_MINIMA_PARA_JUZGAR = _V.DUR_MINIMA_PARA_JUZGAR
SILENCIO_DB = _V.SILENCIO_DB
volumen_medio = _V.volumen_medio


def clasificar(chars: int, dur: float, db: float | None) -> str | None:
    """Por qué este transcript merece una pasada sin VAD, o None si no la merece.

    La decisión está acá, separada de la lectura del disco, porque es la parte
    que tiene que estar bien: un criterio demasiado ancho manda a re-transcribir
    silencio, y lo que vuelve de ahí es alucinación.

    `db` es el volumen medio de la fuente, o None si no se pudo medir. Para un
    transcript vacío es obligatorio: sin la medición no hay forma de distinguir
    un video muteado de una falla del filtro de voz, y se prefiere no tocarlo.
    """
    if not dur or dur <= 0:
        return None
    if chars == 0:
        # La duracion no importa: 0 caracteres sobre audio audible es falla del
        # filtro, dure 2 segundos o 2 minutos.
        if db is None or db <= SILENCIO_DB:
            return None
        return "vacio con %.1f dB de audio" % db
    if dur < DUR_MINIMA_PARA_JUZGAR:
        return None
    densidad = chars / (dur / 60.0)
    if densidad >= MIN_CHARS_POR_MINUTO:
        return None
    return "%.1f chars/minuto" % densidad


def candidatos(medir_audio: bool = True) -> list[dict]:
    """Los transcripts que vale la pena volver a transcribir sin VAD.

    Son dos poblaciones distintas, y el verificador ya las juzga por separado:

    - **Densidad baja** (`chars > 0`, duración ≥ 30 s, menos de
      MIN_CHARS_POR_MINUTO): hay texto pero es poco para el tiempo que dura.
    - **Vacíos con audio audible**: 0 caracteres sobre una fuente cuyo volumen
      medio está por encima de SILENCIO_DB. Eso es exactamente una falla del
      filtro de voz: hay sonido y el filtro no reconoció nada.

    Un vacío cuya fuente mide -91 dB es silencio digital —video muteado, nota de
    voz muda— y **no entra**. Volver a transcribir silencio sin filtro de voz no
    recupera nada: es el caso donde Whisper alucina más, porque no tiene ninguna
    señal sobre la que apoyarse. Medir el volumen cuesta una pasada de ffmpeg por
    archivo, y ahorra re-transcribir decenas de archivos mudos para tirar el
    resultado después.
    """
    out = []
    for tp in sorted(TRANSCRIPTS.glob("*.json")):
        try:
            d = json.loads(tp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print("ILEGIBLE %s: %s" % (tp.name, e), flush=True)
            continue
        dur = (d.get("metadata") or {}).get("duration_seconds")
        if not isinstance(dur, (int, float)) or not dur or dur <= 0:
            continue
        dur = float(dur)
        txt = (d.get("full_text") or "").strip()
        fuente = Path(d.get("source_path") or "")
        c = {
            "stem": tp.stem,
            "fuente": fuente,
            "tipo": d.get("source_type") or "wa_audio",
            "dur": dur,
            "chars": len(txt),
            "densidad": len(txt) / (dur / 60.0),
            "arranque": txt[:200],
            "db": None,
        }
        if len(txt) == 0:
            # Medir cuesta una pasada de ffmpeg, asi que solo se mide lo vacio,
            # que es lo unico cuya decision depende del volumen.
            if not medir_audio or not fuente.exists():
                continue
            c["db"] = volumen_medio(fuente)
        motivo = clasificar(c["chars"], dur, c["db"])
        if motivo is None:
            continue
        c["motivo"] = motivo
        out.append(c)
    out.sort(key=lambda c: c["densidad"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-transcribe sin VAD lo que el verificador marca por densidad baja.")
    ap.add_argument("--listar", action="store_true",
                    help="Solo muestra los candidatos y sale, sin cargar el modelo")
    ap.add_argument("--limite", type=int, default=0,
                    help="Procesa como maximo N candidatos (0 = todos)")
    args = ap.parse_args()

    print("umbrales desde verificar.py: piso %.1f chars/minuto, silencio %.1f dB"
          % (MIN_CHARS_POR_MINUTO, SILENCIO_DB), flush=True)
    print("midiendo el volumen de los transcripts vacios con ffmpeg...", flush=True)
    cands = candidatos()
    print("candidatos: %d" % len(cands), flush=True)

    pendientes = []
    for c in cands:
        destino = SALIDA / (c["stem"] + ".json")
        estado = "ya hecho" if destino.exists() else (
            "FUENTE AUSENTE" if not c["fuente"].exists() else "pendiente")
        print("  %-14s %7.0fs  %-26s  %s"
              % (estado, c["dur"], c["motivo"], c["stem"]), flush=True)
        if estado == "pendiente":
            pendientes.append(c)

    print("pendientes: %d" % len(pendientes), flush=True)
    if args.listar:
        return 0
    if not pendientes:
        print("nada que hacer.", flush=True)
        return 0
    if args.limite:
        pendientes = pendientes[: args.limite]

    SALIDA.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "src"))
    from coach_brain.ingest.transcribe import extract_audio, transcribe_file
    from faster_whisper import WhisperModel

    # CPU int8 a proposito: la GPU la esta usando la extraccion con Ollama, y
    # las dos juntas no entran en 12 GB.
    print("cargando large-v3 en CPU int8...", flush=True)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    for c in pendientes:
        src = c["fuente"]
        print("\n=== %s  (%.0fs, %.1f c/min)" % (c["stem"], c["dur"], c["densidad"]),
              flush=True)
        print("    con VAD:  %d chars | %s" % (c["chars"], c["arranque"]), flush=True)
        try:
            if src.suffix.lower() in VIDEO_EXT:
                tmp = Path(tempfile.gettempdir()) / ("_novad_" + src.stem + ".wav")
                try:
                    extract_audio(src, tmp)
                    nd = transcribe_file(model, tmp, c["tipo"], src, vad=False)
                finally:
                    tmp.unlink(missing_ok=True)
            else:
                nd = transcribe_file(model, src, c["tipo"], src, vad=False)
        except Exception as e:  # noqa: BLE001
            print("    ERROR: %s" % e, flush=True)
            continue
        destino = SALIDA / (c["stem"] + ".json")
        destino.write_text(json.dumps(nd, ensure_ascii=False, indent=2), encoding="utf-8")
        print("    sin VAD:  %d chars, %d segmentos -> %s"
              % (len(nd["full_text"]), len(nd["segments"]), destino), flush=True)
        print("    %s" % nd["full_text"][:1500], flush=True)

    print("\nlisto. LEER data/processed/_novad/ antes de promover nada al corpus:",
          flush=True)
    print("sin el filtro de voz Whisper alucina sobre ruido, y eso seria peor que el hueco.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
