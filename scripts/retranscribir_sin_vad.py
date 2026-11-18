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
import re
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


# Frases que Whisper inventa cuando no hay habla que transcribir. Vienen de los
# subtitulos de YouTube con los que se entreno, y aparecen enteras como si
# fueran el contenido del audio. Medidas en este corpus: "Gracias.",
# "Gracias por ver el video." y "La ciudad de Bogota" repetida nueve veces.
ALUCINACIONES = (
    "gracias",
    "gracias por ver el video",
    "gracias por ver el vídeo",
    "gracias por su atencion",
    "gracias por su atención",
    "muchas gracias",
    "gracias por mirar",
    "suscribete",
    "suscríbete",
    "suscribete al canal",
    "suscríbete al canal",
    "suscribanse",
    "suscríbanse",
    "subtitulos realizados por la comunidad de amara.org",
    "subtítulos realizados por la comunidad de amara.org",
    "amara.org",
    "musica",
    "música",
    "subtitulado por",
    "hasta la proxima",
    "hasta la próxima",
    "nos vemos en el proximo video",
    "nos vemos en el próximo video",
    "no te olvides de suscribirte",
    "y hasta luego",
    "fin",
    "continuara",
    "continuará",
    "mas informacion",
    "más información",
)

# Whisper tambien inventa dominios, que en una nota de voz no existen.
_URL = re.compile(r"www\.\S+|https?://", re.I)

# Las mismas frases, para sacarlas de adentro de una frase mas larga. Cuando
# Whisper las escupe sin puntuacion se fusionan con lo que tienen al lado y
# dejan de ser una frase entera: "paloma dahi paloma de tennis gracias por ver
# el video" contaba sus 53 caracteres como habla real.
#
# Los limites de palabra no son decorativos: "fin" esta en la lista, y sin
# \b saldria de "definitivamente", "final" y "afinar". Las mas largas van
# primero para que "gracias por ver el video" gane contra "gracias".
_INVENTADAS = re.compile(
    r"\b(?:" + "|".join(
        re.escape(f) for f in sorted(ALUCINACIONES, key=len, reverse=True)
    ) + r")\b", re.I)

# Un texto que colapsa a una sola frase corta repetida es un bucle del decoder,
# no habla. Con menos repeticiones que esto puede ser enfasis real.
REPETICIONES_PARA_BUCLE = 3

# Cuando una secuencia corta de palabras cubre esta fraccion del texto, el
# decoder esta en bucle. Medido: el caso bueno de 6445 caracteres tiene su
# n-grama mas repetido cubriendo 2.7%, y los bucles reales cubren mas del 90%.
COBERTURA_PARA_BUCLE = 0.6
MAX_PALABRAS_DEL_NGRAMA = 8

# Si al sacar las frases inventadas queda menos que esto, no habia nada mas.
# El caso bueno conserva el 96% del texto; los descartados, cero.
RESTO_MINIMO = 0.25


def _normalizar(t: str) -> str:
    """Minusculas, sin puntuacion de borde ni espacios repetidos."""
    t = " ".join(t.lower().split())
    return t.strip(" .,;:!?¡¿-–—\"'()[]")


def _frases(t: str) -> list[str]:
    """El texto partido en frases normalizadas, sin vacias."""
    return [f for f in (_normalizar(x) for x in re.split(r"[.!?\n]+", t)) if f]


def _ngrama_dominante(t: str):
    """La secuencia corta de palabras más repetida y qué fracción del texto cubre.

    Hace falta porque Whisper en bucle muchas veces no pone puntuación: "La
    ciudad de Bogota" nueve veces seguidas es UNA sola frase para cualquier
    criterio que parta por puntos, y así se escapó de la primera versión.

    Devuelve (secuencia, veces, cobertura). La cobertura es qué parte de las
    palabras del texto ocupa esa secuencia repetida: un bucle real cubre casi
    todo, mientras que una muletilla dentro de una conversación cubre poco.

    La cobertura cuenta **posiciones distintas**, no apariciones. Multiplicar
    apariciones por largo sobrecuenta cuando se solapan: diez "o sea" seguidos
    ocupan 20 palabras, pero el 4-grama "o sea o sea" aparece en 7 posiciones
    solapadas y multiplicando daria 28 palabras de 40, un 70% que no existe.
    """
    palabras = _normalizar(t).split()
    if not palabras:
        return ("", 0, 0.0)
    mejor = ("", 0, 0.0)
    for n in range(2, MAX_PALABRAS_DEL_NGRAMA + 1):
        if len(palabras) < n * REPETICIONES_PARA_BUCLE:
            break
        posiciones: dict[str, list[int]] = {}
        for k in range(len(palabras) - n + 1):
            posiciones.setdefault(" ".join(palabras[k:k + n]), []).append(k)
        g, ks = max(posiciones.items(), key=lambda kv: len(kv[1]))
        cubiertas = {i for k in ks for i in range(k, k + n)}
        cobertura = len(cubiertas) / len(palabras)
        if cobertura > mejor[2]:
            mejor = (g, len(ks), cobertura)
    return mejor


def _resto_sin_frases_inventadas(t: str) -> float:
    """Qué fracción del texto queda al sacar las frases que Whisper inventa.

    Separa el caso bueno del malo mejor que contar frases. El transcript de
    6445 caracteres que recuperó diálogo real incluye "Muchas gracias" quince
    veces, y conserva el 96%. Los que son solo "Gracias por ver el video."
    repetido conservan cero, por mucho que las repeticiones no lleguen al piso
    de tres.
    """
    # El denominador se mide sobre las mismas frases que el numerador. Medido
    # sobre el texto entero, cada punto y cada salto de linea que se come el
    # split contaba como texto borrado: el PDF de 95941 caracteres daba 11%
    # "inventado" teniendo adentro seis frases de la lista, unos 30 caracteres.
    total = 0
    queda = 0
    for f in _frases(t):
        total += len(f)
        limpio = _INVENTADAS.sub(" ", f)
        limpio = _URL.sub(" ", limpio)
        queda += len(" ".join(limpio.split()))
    if not total:
        return 0.0
    return queda / total


def parece_alucinacion(texto: str) -> str | None:
    """Por qué este texto parece inventado, o None si parece habla real.

    No decide nada por su cuenta: solo marca, para que revisar los resultados
    sea leer con atención los que pueden ser habla real en vez de todos. Un
    falso positivo se ve al leerlo; lo que no se puede permitir es que una
    alucinación entre al corpus sin que nadie la haya mirado.

    Tres señales, las tres medidas en este corpus:

    - El texto entero es una frase que Whisper arrastra de los subtítulos de
      YouTube con los que se entrenó ("Gracias por ver el video.").
    - Sacando esas frases no queda casi nada, así que no había nada más.
    - Una secuencia corta de palabras cubre casi todo el texto: el decoder en
      bucle ("La ciudad de Bogota" nueve veces, sin un solo punto).
    """
    t = texto.strip()
    if not t:
        return "vacio"
    n = _normalizar(t)
    if n in ALUCINACIONES:
        return "es solo %r, frase que Whisper inventa sobre ruido" % t.strip()

    resto = _resto_sin_frases_inventadas(t)
    if resto < RESTO_MINIMO:
        return ("sacando las frases que Whisper inventa queda el %.0f%% del texto"
                % (resto * 100))

    frases = _frases(t)
    if frases:
        distintas = set(frases)
        if len(frases) >= REPETICIONES_PARA_BUCLE and len(distintas) == 1:
            return ("una sola frase repetida %d veces, decoder en bucle: %r"
                    % (len(frases), frases[0]))
        mas_comun = max(distintas, key=frases.count)
        veces = frases.count(mas_comun)
        if veces >= REPETICIONES_PARA_BUCLE and veces / len(frases) >= 0.8:
            return ("%d de %d frases son %r, decoder en bucle"
                    % (veces, len(frases), mas_comun))

    g, veces, cobertura = _ngrama_dominante(t)
    if veces >= REPETICIONES_PARA_BUCLE and cobertura >= COBERTURA_PARA_BUCLE:
        return ("%r repetido %d veces cubre el %.0f%% del texto, decoder en bucle"
                % (g, veces, cobertura * 100))
    return None


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


# Un bucle que cubre menos que COBERTURA_PARA_BUCLE no alcanza para descartar,
# pero el que lee tiene que saber que esta ahi. Medido: el caso bueno de 6445
# caracteres tiene su n-grama mas repetido cubriendo 2.7%, asi que 25% no lo
# toca, y el caso mitad-habla-mitad-bucle de wa_audio_00029933 cubre 49%.
AVISO_COBERTURA = 0.25

# Lo mismo con las frases inventadas: por debajo de RESTO_MINIMO se descarta,
# pero perder una decima parte del texto ya merece una mirada.
AVISO_INVENTADO = 0.1


def avisos(texto: str) -> list[str]:
    """Senales que no alcanzan para descartar pero que hay que leer sabiendo.

    Existen porque los resultados no se parten en limpios y basura: hay
    transcripts que son mitad habla real y mitad bucle del decoder. El
    veredicto los manda a revisar, que es lo correcto, y estos avisos dicen
    donde mirar. No cambian el veredicto: un aviso de mas cuesta una linea, y
    una alucinacion promovida sin que nadie la viera cuesta el corpus.
    """
    t = (texto or "").strip()
    if not t:
        return []
    out = []
    g, veces, cobertura = _ngrama_dominante(t)
    if veces >= REPETICIONES_PARA_BUCLE and cobertura >= AVISO_COBERTURA:
        out.append("ojo: %r repetido %d veces cubre el %.0f%% del texto"
                   % (g, veces, cobertura * 100))
    inventado = 1.0 - _resto_sin_frases_inventadas(t)
    if inventado >= AVISO_INVENTADO:
        out.append("ojo: el %.0f%% del texto son frases que Whisper inventa"
                   % (inventado * 100))
    return out


def _veredicto(nuevo: str, chars_con_vad: int) -> str:
    """Qué conviene hacer con un resultado sin VAD. Marca, no decide.

    Promover un transcript cambia lo que el índice devuelve, y descartarlo tira
    trabajo de GPU. Las dos cosas las decide una persona que haya leído el
    texto; esto solo ordena los 22 resultados para que haya que leer con
    atención los que pueden ser habla real.
    """
    nuevo = (nuevo or "").strip()
    sospecha = parece_alucinacion(nuevo)
    if sospecha:
        return "DESCARTAR: " + sospecha
    if len(nuevo) <= chars_con_vad:
        return ("DESCARTAR: no recupero nada (%d chars contra %d con VAD)"
                % (len(nuevo), chars_con_vad))
    return "REVISAR: %+d chars contra la version con VAD" % (len(nuevo) - chars_con_vad)


def juzgar() -> int:
    """Informa qué hay en `_novad/` y qué conviene hacer con cada resultado.

    No promueve nada ni borra nada.
    """
    if not SALIDA.exists():
        print("no hay nada en %s" % SALIDA, flush=True)
        return 0
    res = sorted(SALIDA.glob("*.json"))
    print("resultados sin VAD: %d" % len(res), flush=True)
    descartar, revisar = [], []
    for rp in res:
        try:
            nd = json.loads(rp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print("ILEGIBLE %s: %s" % (rp.name, e), flush=True)
            continue
        nuevo = (nd.get("full_text") or "").strip()
        orig_p = TRANSCRIPTS / rp.name
        viejo = ""
        if orig_p.exists():
            try:
                viejo = (json.loads(orig_p.read_text(encoding="utf-8"))
                         .get("full_text") or "").strip()
            except Exception:  # noqa: BLE001
                pass
        v = _veredicto(nuevo, len(viejo))
        fila = (rp.stem, len(viejo), len(nuevo), v, nuevo)
        (descartar if v.startswith("DESCARTAR") else revisar).append(fila)

    for titulo, grupo in (("A REVISAR", revisar), ("A DESCARTAR", descartar)):
        print("\n%s: %d" % (titulo, len(grupo)), flush=True)
        for stem, lv, ln, v, texto in grupo:
            print("\n  %s" % stem, flush=True)
            print("    %s" % v, flush=True)
            print("    con VAD %d chars | sin VAD %d chars" % (lv, ln), flush=True)
            if titulo == "A REVISAR":
                # En el grupo descartado no suman: ya tienen su motivo arriba.
                for a in avisos(texto):
                    print("    %s" % a, flush=True)
            print("    %s" % " ".join(texto[:600].split()), flush=True)
    print("\nNada se promueve ni se borra automaticamente: "
          "lo decide una persona que leyo el texto.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-transcribe sin VAD lo que el verificador marca por densidad baja.")
    ap.add_argument("--listar", action="store_true",
                    help="Solo muestra los candidatos y sale, sin cargar el modelo")
    ap.add_argument("--juzgar", action="store_true",
                    help="Informa que hay en _novad/ y que hacer con cada resultado")
    ap.add_argument("--limite", type=int, default=0,
                    help="Procesa como maximo N candidatos (0 = todos)")
    args = ap.parse_args()
    if args.juzgar:
        return juzgar()

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
        print("\n=== %s  (%.0fs, %s)" % (c["stem"], c["dur"], c["motivo"]), flush=True)
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
        print("    %s" % _veredicto(nd["full_text"], c["chars"]), flush=True)
        print("    %s" % nd["full_text"][:1500], flush=True)

    print("\nlisto. LEER data/processed/_novad/ antes de promover nada al corpus:",
          flush=True)
    print("sin el filtro de voz Whisper alucina sobre ruido, y eso seria peor que el hueco.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
