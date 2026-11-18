"""Tests de `scripts/manifest.py` y del criterio de duplicado de `verificar.py`.

Estos dos scripts no tenían ninguna prueba, y son los que producen
`INVENTARIO.md`: el registro sobre el que se decide si los ~360 GB de material
crudo se pueden borrar. Un error acá no rompe nada visible, escribe un registro
que dice lo que uno quiere leer antes de borrar.

Lo que se fija:

1. **El piso del inventario.** `os.walk` sobre una ruta que no se puede
   recorrer no falla: devuelve cero archivos (gotcha #27). Con las junctions de
   `data/raw` apuntando a un disco desconectado, el inventario saldría diciendo
   que en disco no hay material — y el script SOBREESCRIBE `INVENTARIO.md`. El
   piso sale de un invariante, no de una calibración: cada transcript lo produjo
   un archivo, así que en disco no puede haber menos archivos que transcripts.
2. **Los cuatro motivos de "sin cubrir" son distintos.** Un video grabado con
   el micrófono apagado no tiene habla que perder; un contenedor ilegible sí es
   una pérdida; y "no se pudo correr ffprobe" no es "está roto". Mezclarlos hace
   que el registro no sirva para decidir.
3. **El matcheo por duración.** Es el pase que evitó un falso "no lo tengo"
   sobre una clase de 2 horas y 3.4 GB que estaba transcripta con otro nombre.
4. **El criterio de duplicado vive en un solo lugar.** `manifest.py` lo importa
   de `verificar.py` por ruta justamente para que no haya dos tolerancias que se
   desincronicen. La prueba comprueba que siga siendo la misma función.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _cargar(nombre: str):
    spec = importlib.util.spec_from_file_location(
        nombre, REPO / "scripts" / f"{nombre}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _cargar("manifest")
V = _cargar("verificar")


# --- 1) el criterio de duplicado, en un solo lugar --------------------------


def test_manifest_usa_el_criterio_de_verificar_sin_copiarlo():
    """Si alguien copia la regla en vez de importarla, las dos herramientas
    empiezan a contestar distinto sobre el mismo par de archivos el día que
    alguien ajuste la tolerancia en una sola."""
    assert M.es_duplicado.__code__.co_filename.endswith("verificar.py")
    assert M.es_duplicado.__code__.co_code == V.es_duplicado.__code__.co_code
    assert M.CABECERA_DUPLICADO == V.CABECERA_DUPLICADO


def test_dos_transcripts_con_el_mismo_largo_y_arranque_son_duplicados():
    a = (120.0, 5000, "arranque igual")
    b = (999.0, 5000, "arranque igual")
    assert V.es_duplicado(a, b) is True


def test_la_misma_duracion_alcanza_aunque_el_largo_difiera():
    """Dos pasadas de Whisper sobre el mismo video dan la duración del
    contenedor idéntica al microsegundo y largos que difieren en miles de
    caracteres, porque el decodificado no es determinista."""
    assert V.es_duplicado((120.0, 5000, "igual"), (120.005, 8000, "igual")) is True


def test_la_duracion_parecida_pero_fuera_de_tolerancia_no_alcanza():
    assert V.es_duplicado((120.0, 5000, "igual"), (120.5, 8000, "igual")) is False


def test_el_mismo_arranque_solo_no_alcanza():
    """Dos notas de voz distintas pueden empezar con la misma muletilla."""
    assert V.es_duplicado((10.0, 300, "hola como estas"),
                          (90.0, 4000, "hola como estas")) is False


def test_sin_arranque_nunca_es_duplicado():
    """Dos transcripts vacíos tienen cabecera vacía y el mismo largo (0). Sin
    esta guarda, los 62 vacíos del corpus serían 62 duplicados entre sí."""
    assert V.es_duplicado((None, 0, ""), (None, 0, "")) is False


def test_arranques_distintos_no_son_duplicados():
    assert V.es_duplicado((120.0, 5000, "uno"), (120.0, 5000, "otro")) is False


def test_sin_duracion_en_ninguno_y_largos_distintos_no_es_duplicado():
    assert V.es_duplicado((None, 5000, "igual"), (None, 8000, "igual")) is False


@pytest.mark.parametrize("dur", [0, 0.0, -3, None, "ciento veinte"])
def test_la_huella_descarta_las_duraciones_que_no_sirven(dur):
    """Una duración de 0 o negativa no es una duración: si entrara como número,
    dos transcripts sin duración medida se compararían como iguales."""
    assert V.huella({"dur": dur, "chars": 10, "cabecera": "x"})[0] is None
    assert M._huella({"dur": dur, "chars": 10, "cabecera": "x"})[0] is None


def test_la_huella_sin_cabecera_devuelve_cadena_vacia():
    assert V.huella({"dur": 1.0, "chars": 10})[2] == ""


# --- 2) el piso que protege a INVENTARIO.md ---------------------------------


def test_el_piso_deja_pasar_lo_normal():
    """Hoy son 17801 archivos contra 8838 transcripts: el margen es amplio y
    solo salta cuando algo se rompió de verdad."""
    assert M.exigir_piso(17801, 8838, argv=["manifest.py"]) is None


def test_el_piso_deja_pasar_el_empate():
    assert M.exigir_piso(8838, 8838, argv=["manifest.py"]) is None


def test_el_piso_corta_si_el_disco_trajo_menos_que_los_transcripts():
    """El caso real que esto ataja: las junctions apuntan a un disco que no
    está, `os.walk` no falla y devuelve nada, y el inventario saldría vacío."""
    with pytest.raises(SystemExit) as e:
        M.exigir_piso(0, 8838, argv=["manifest.py"])
    assert "ABORTADO" in str(e.value)


def test_el_mensaje_del_piso_dice_que_no_se_escribio_nada():
    """Lo que hace útil al corte es que quien lo lea sepa que el inventario que
    hay sigue siendo el de antes."""
    with pytest.raises(SystemExit) as e:
        M.exigir_piso(3, 8838, argv=["manifest.py"])
    texto = str(e.value)
    assert "No se escribio nada" in texto
    assert "--sin-piso" in texto


def test_sin_piso_lo_desactiva():
    """Para el caso en que el material se haya borrado a propósito y se quiera
    un inventario que lo refleje."""
    assert M.exigir_piso(0, 8838, argv=["manifest.py", "--sin-piso"]) is None


# --- 3) por qué un archivo está sin cubrir ----------------------------------


@pytest.fixture
def sin_fuera(monkeypatch):
    """Sin exclusiones deliberadas, salvo que el test diga otra cosa."""
    monkeypatch.setattr(M, "_stems_fuera_del_corpus", lambda: set())


def fila(path: str, tipo: str = "video", cubierto: str = "NO") -> dict:
    return {"path": path, "tipo": tipo, "cubierto": cubierto}


def test_lo_cubierto_no_lleva_motivo(sin_fuera):
    rows = [fila("a.mp4", cubierto="SI")]
    assert M.clasificar_sin_cubrir(rows) == 0
    assert rows[0]["motivo"] == ""


def test_un_video_sin_pista_de_audio_no_es_un_hueco(monkeypatch, sin_fuera):
    """No hay habla que perder: el archivo se puede borrar sin perder nada."""
    monkeypatch.setattr(M, "_pista_de_audio", lambda _p: "no")
    rows = [fila("mudo.mp4")]
    assert M.clasificar_sin_cubrir(rows) == 0
    assert "no hay habla" in rows[0]["motivo"]
    assert "HUECO" not in rows[0]["motivo"]


def test_un_contenedor_ilegible_si_es_un_hueco(monkeypatch, sin_fuera):
    monkeypatch.setattr(M, "_pista_de_audio", lambda _p: "ilegible")
    rows = [fila("roto.mp4")]
    assert M.clasificar_sin_cubrir(rows) == 1
    assert rows[0]["motivo"].startswith("HUECO")


def test_sin_ffprobe_la_duda_queda_del_lado_conservador(monkeypatch, sin_fuera):
    """"No se pudo verificar" no es "está roto", pero tampoco es "se puede
    borrar": queda como hueco para que el archivo no entre en la lista de lo
    que se borra."""
    monkeypatch.setattr(M, "_pista_de_audio", lambda _p: "sin_probe")
    rows = [fila("dudoso.mp4")]
    assert M.clasificar_sin_cubrir(rows) == 1
    assert "sin verificar" in rows[0]["motivo"]


def test_un_video_con_audio_y_sin_transcript_es_un_hueco(monkeypatch, sin_fuera):
    monkeypatch.setattr(M, "_pista_de_audio", lambda _p: "si")
    rows = [fila("falta.mp4")]
    assert M.clasificar_sin_cubrir(rows) == 1
    assert rows[0]["motivo"] == "HUECO: falta procesarlo"


def test_un_excluido_a_proposito_no_es_un_hueco(monkeypatch):
    """Los CV de Ilan llegaron por WhatsApp y su transcript se movió a
    `_fuera_del_corpus/`. No son material del corpus, así que figurar como
    pérdida sería falso."""
    monkeypatch.setattr(M, "_stems_fuera_del_corpus", lambda: {"cv_ilan"})
    rows = [fila("algo/CV_Ilan.pdf", tipo="doc")]
    assert M.clasificar_sin_cubrir(rows) == 0
    assert "excluido a proposito" in rows[0]["motivo"]


def test_un_doc_sin_cubrir_no_pasa_por_ffprobe(monkeypatch, sin_fuera):
    """Correr ffprobe sobre un PDF es tiempo perdido sobre miles de archivos."""
    llamadas = []
    monkeypatch.setattr(M, "_pista_de_audio", lambda p: llamadas.append(p) or "no")
    rows = [fila("apunte.pdf", tipo="doc")]
    assert M.clasificar_sin_cubrir(rows) == 1
    assert llamadas == []


# --- 4) el matcheo por duración ---------------------------------------------


def pendiente(path: str, kind: str = "video") -> list:
    return [{"path": path, "kind": kind}, None, ""]


def test_matchea_un_archivo_renombrado_por_su_duracion(monkeypatch):
    """El caso real: en disco "78 que es el valor.mp4", en el transcript
    "78 que es el valor-008.mp4". 3.4 GB que el inventario contaba como
    material sin transcribir."""
    monkeypatch.setattr(M, "_dur_ffprobe", lambda _p: 7200.0)
    h = pendiente("78 que es el valor.mp4")
    by_base = {"78 que es el valor-008.mp4": [{"dur": 7200.3, "transcript": "video_78.json"}]}
    assert M.match_por_duracion([h], by_base) == 1
    assert h[2] == "transcript (por duracion)"
    assert h[1]["transcript"] == "video_78.json"


def test_fuera_de_tolerancia_no_matchea(monkeypatch):
    monkeypatch.setattr(M, "_dur_ffprobe", lambda _p: 7200.0)
    h = pendiente("otra.mp4")
    by_base = {"x.mp4": [{"dur": 7205.0, "transcript": "t.json"}]}
    assert M.match_por_duracion([h], by_base) == 0
    assert h[1] is None


def test_elige_el_transcript_mas_cercano(monkeypatch):
    monkeypatch.setattr(M, "_dur_ffprobe", lambda _p: 100.0)
    h = pendiente("a.mp4")
    by_base = {
        "lejos.mp4": [{"dur": 100.9, "transcript": "lejos.json"}],
        "cerca.mp4": [{"dur": 100.1, "transcript": "cerca.json"}],
    }
    assert M.match_por_duracion([h], by_base) == 1
    assert h[1]["transcript"] == "cerca.json"


def test_un_transcript_no_se_usa_dos_veces(monkeypatch):
    """Dos archivos de la misma duración no pueden quedar los dos cubiertos por
    el mismo transcript: eso declararía cubierto un archivo que no lo está."""
    monkeypatch.setattr(M, "_dur_ffprobe", lambda _p: 60.0)
    h1, h2 = pendiente("a.mp4"), pendiente("b.mp4")
    by_base = {"t.mp4": [{"dur": 60.0, "transcript": "t.json"}]}
    assert M.match_por_duracion([h1, h2], by_base) == 1
    assert (h1[1] is None) != (h2[1] is None)


def test_quedarse_sin_transcripts_no_rompe_el_inventario(monkeypatch):
    """Con mas archivos sin cubrir que transcripts sueltos, la lista de
    candidatos se vacia a mitad del barrido. Antes `min()` sobre la lista vacia
    tiraba ValueError y se llevaba puesta la corrida entera de manifest.py: el
    inventario no se escribia y el motivo visible era un traceback que no decia
    nada sobre duraciones."""
    monkeypatch.setattr(M, "_dur_ffprobe", lambda _p: 60.0)
    hits = [pendiente(f"{i}.mp4") for i in range(5)]
    by_base = {"t.mp4": [{"dur": 60.0, "transcript": "t.json"}]}
    assert M.match_por_duracion(hits, by_base) == 1


def test_sin_duracion_medida_no_matchea(monkeypatch):
    """Si ffprobe no pudo leer el archivo, no hay con qué comparar. Inventar un
    match acá es declarar cubierto algo que no se verificó."""
    monkeypatch.setattr(M, "_dur_ffprobe", lambda _p: None)
    h = pendiente("a.mp4")
    by_base = {"t.mp4": [{"dur": 60.0, "transcript": "t.json"}]}
    assert M.match_por_duracion([h], by_base) == 0


def test_solo_prueba_sobre_audio_y_video(monkeypatch):
    """Un PDF no tiene duración. Correr ffprobe sobre los documentos sin cubrir
    sería gastar minutos para nada."""
    llamadas = []
    monkeypatch.setattr(M, "_dur_ffprobe", lambda p: llamadas.append(p) or 60.0)
    h = pendiente("apunte.pdf", kind="doc")
    by_base = {"t.mp4": [{"dur": 60.0, "transcript": "t.json"}]}
    assert M.match_por_duracion([h], by_base) == 0
    assert llamadas == []


def test_lo_ya_cubierto_no_se_vuelve_a_probar(monkeypatch):
    llamadas = []
    monkeypatch.setattr(M, "_dur_ffprobe", lambda p: llamadas.append(p) or 60.0)
    h = [{"path": "a.mp4", "kind": "video"}, {"transcript": "ya.json"}, "transcript"]
    by_base = {"t.mp4": [{"dur": 60.0, "transcript": "t.json"}]}
    assert M.match_por_duracion([h], by_base) == 0
    assert llamadas == []


def test_los_transcripts_sin_duracion_no_entran_al_pase(monkeypatch):
    monkeypatch.setattr(M, "_dur_ffprobe", lambda _p: 60.0)
    h = pendiente("a.mp4")
    by_base = {"t.mp4": [{"dur": 0, "transcript": "t.json"}]}
    assert M.match_por_duracion([h], by_base) == 0


# --- 5) clasificación de extensiones y formato ------------------------------


@pytest.mark.parametrize("ext,esperado", [
    (".mp4", "video"), (".mkv", "video"),
    (".mp3", "audio"), (".opus", "audio"),
    (".pdf", "doc"), (".docx", "doc"),
    (".jpg", "imagen"),
    (".exe", "otro"), ("", "otro"),
])
def test_kind_clasifica_por_extension(ext, esperado):
    assert M.kind(ext) == esperado


def test_gb_informa_con_dos_decimales():
    assert M.gb(1024 ** 3) == "1.00 GB"
    assert M.gb(0) == "0.00 GB"
