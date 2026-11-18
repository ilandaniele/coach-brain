"""Tests de `scripts/reparar_source_path.py`: el único script que reescribe el corpus.

Todos los demás scripts de `scripts/` informan. Este reescribe los transcripts
de `data/processed/transcripts/`, que son el corpus del que se arma el índice, y
lo hace mientras `kb.extract` puede estar leyéndolos. Las cuatro propiedades que
lo hacen seguro no son visibles en el informe que imprime, así que tienen que
estar en tests:

1. **El pase seco no escribe nada.** Es el modo por omisión justamente porque se
   corre sobre el corpus vivo para ver qué haría. Si escribiera, nadie se
   enteraría hasta que el índice devolviera otra cosa.
2. **`--apply` copia antes de tocar.** El backup queda en
   `_transcripts_backup_source_path/` con el contenido original, y el
   `source_path` viejo queda escrito en `metadata.source_path_previo`, así que
   la reparación se puede deshacer y auditar.
3. **La escritura es atómica.** `write_text` no lo es: un lector concurrente
   puede ver un JSON truncado. El script escribe a `.tmp` y hace `os.replace`.
4. **Un nombre ambiguo no se resuelve.** Elegir uno de dos candidatos haría que
   un transcript apunte al archivo equivocado, y eso no se nota: el transcript
   sigue siendo válido, solo deja de corresponder a su fuente.

Y la resolución por nombre con el sufijo del pipeline, que es el caso real:
en disco está "78 que es el valor.mp4" y el transcript apunta a
"78 que es el valor-008.mp4".

`tests/test_reparar_source_path.py` ya cubre las dos decisiones puras del
script —`resolver` y `es_duplicado`— con el caso normal, el ambiguo y el del
sufijo. Lo que falta acá es el otro lado: el índice de disco (y las tres
junctions que apuntan al mismo destino) y todo el camino de escritura de
`main()`, que es donde el script puede dejar el corpus distinto de como lo
encontró.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _cargar():
    spec = importlib.util.spec_from_file_location(
        "reparar_source_path", REPO / "scripts" / "reparar_source_path.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _cargar()


# --- el criterio de duplicado, importado y no copiado -----------------------


def test_el_criterio_de_duplicado_viene_del_verificador():
    """Tres scripts comparan transcripts por contenido. Si cada uno se copiara
    la regla, ajustar la tolerancia en uno dejaría a los otros dos contestando
    distinto sobre el mismo par de archivos."""
    assert R.es_duplicado.__code__.co_filename.endswith("verificar.py")
    assert R.CABECERA_DUPLICADO == 200
    assert R.TOLERANCIA_DUPLICADO == 0.01


def test_la_huella_descarta_una_duracion_que_no_sirve():
    assert R.huella({"full_text": "hola", "metadata": {"duration_seconds": 0}})[0] is None
    assert R.huella({"full_text": "hola"})[0] is None


def test_la_huella_corta_el_arranque_en_el_largo_del_criterio():
    h = R.huella({"full_text": "x" * 500})
    assert h[1] == 500
    assert len(h[2]) == R.CABECERA_DUPLICADO


# --- resolver(): de un source_path roto a un archivo en disco ---------------


def test_resuelve_por_nombre_ignorando_mayusculas():
    idx = {"clase 3.mp4": [Path("D:/raw/videos/Clase 3.mp4")]}
    nueva, como = R.resolver(Path("E:/viejo/CLASE 3.mp4"), idx)
    assert nueva == Path("D:/raw/videos/Clase 3.mp4")
    assert como == "nombre"






def test_al_sacar_el_sufijo_conserva_la_extension():
    idx = {"a.opus": [Path("D:/raw/a.opus")]}
    nueva, _ = R.resolver(Path("E:/a-001.opus"), idx)
    assert nueva == Path("D:/raw/a.opus")


def test_el_nombre_exacto_gana_sobre_el_sufijo():
    """Si los dos archivos existen, el que el transcript nombra es el suyo."""
    idx = {
        "clase-008.mp4": [Path("D:/raw/clase-008.mp4")],
        "clase.mp4": [Path("D:/raw/clase.mp4")],
    }
    nueva, como = R.resolver(Path("E:/clase-008.mp4"), idx)
    assert nueva == Path("D:/raw/clase-008.mp4")
    assert como == "nombre"


@pytest.mark.parametrize("roto", ["clase-08.mp4", "clase-0081.mp4", "clase-abc.mp4"])
def test_solo_se_saca_un_sufijo_de_tres_digitos(roto):
    """El sufijo del pipeline son exactamente tres dígitos. Sacar cualquier cosa
    que se parezca convertiría "Leccion 1-2" o un año en un nombre distinto y
    resolvería contra el archivo equivocado."""
    idx = {"clase.mp4": [Path("D:/raw/clase.mp4")]}
    nueva, como = R.resolver(Path("E:/" + roto), idx)
    assert nueva is None
    assert como == "sin candidato por nombre"


def test_ambiguo_tambien_sin_sufijo():
    idx = {"clase.mp4": [Path("D:/a/clase.mp4"), Path("D:/b/clase.mp4")]}
    nueva, como = R.resolver(Path("E:/clase-008.mp4"), idx)
    assert nueva is None
    assert como == "ambiguo sin sufijo: 2 candidatos"




# --- indexar_disco(): un archivo que está una vez aparece una vez -----------


def test_el_indice_usa_nombres_en_minusculas(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "videos").mkdir(parents=True)
    (raw / "videos" / "Clase 3.MP4").write_bytes(b"\x00")
    monkeypatch.setattr(R, "RAW", raw)
    idx = R.indexar_disco()
    assert list(idx) == ["clase 3.mp4"]


def test_el_mismo_nombre_en_dos_carpetas_son_dos_candidatos(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    for sub in ("a", "b"):
        (raw / sub).mkdir(parents=True)
        (raw / sub / "clase.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(R, "RAW", raw)
    assert len(R.indexar_disco()["clase.mp4"]) == 2


def test_las_tres_junctions_al_mismo_destino_no_triplican_el_corpus(
        tmp_path, monkeypatch):
    """Las tres entradas de `data/raw` son junctions al mismo árbol en
    `Desktop\\Ilan\\Seduccion`. Sin deduplicar por realpath, cada archivo
    aparece tres veces y TODO nombre queda "ambiguo: 3 candidatos", es decir:
    el script no repara nada y el motivo que imprime es falso."""
    raw = tmp_path / "raw"
    for sub in ("videos", "espejo1", "espejo2"):
        (raw / sub).mkdir(parents=True)
        (raw / sub / "clase.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(R, "RAW", raw)

    # Simula los junctions: los tres directorios resuelven al mismo destino.
    def real_con_junctions(p: Path) -> str:
        s = str(p).lower()
        return s.replace("espejo1", "videos").replace("espejo2", "videos")

    monkeypatch.setattr(R, "_real", real_con_junctions)
    idx = R.indexar_disco()
    assert len(idx["clase.mp4"]) == 1


# --- main(): el pase seco y el pase que escribe -----------------------------


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Un corpus de juguete: un transcript roto y un archivo en disco que lo
    resuelve por nombre."""
    raw = tmp_path / "raw" / "videos"
    raw.mkdir(parents=True)
    fuente = raw / "clase 3.mp4"
    fuente.write_bytes(b"\x00")

    ts = tmp_path / "transcripts"
    ts.mkdir()
    roto = ts / "video_clase 3.json"
    roto.write_text(json.dumps({
        "source_id": "video_clase 3",
        "source_path": "E:/disco viejo/clase 3.mp4",
        "full_text": "hola, esto es una clase",
        "metadata": {"duration_seconds": 60.0},
    }, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(R, "RAW", tmp_path / "raw")
    monkeypatch.setattr(R, "TRANSCRIPTS", ts)
    monkeypatch.setattr(R, "BACKUP", tmp_path / "backup")
    return tmp_path, ts, roto, fuente


def correr(*argv: str) -> int:
    import sys
    orig = sys.argv
    sys.argv = ["reparar_source_path.py", *argv]
    try:
        return R.main()
    finally:
        sys.argv = orig


def test_el_pase_seco_no_escribe_nada(corpus, capsys):
    """Es el modo por omisión porque se corre sobre el corpus vivo."""
    base, _ts, roto, _f = corpus
    antes = roto.read_bytes()
    assert correr() == 0
    assert roto.read_bytes() == antes
    assert not (base / "backup").exists()
    assert "pase seco" in capsys.readouterr().out


def test_apply_reescribe_el_source_path(corpus):
    _base, _ts, roto, fuente = corpus
    assert correr("--apply") == 0
    d = json.loads(roto.read_text(encoding="utf-8"))
    assert Path(d["source_path"]) == fuente


def test_apply_guarda_el_source_path_viejo(corpus):
    """Sin esto la reparación no es auditable: no queda registro de a qué
    apuntaba el transcript antes."""
    _base, _ts, roto, _f = corpus
    correr("--apply")
    d = json.loads(roto.read_text(encoding="utf-8"))
    # Path() para no comparar separadores: el script guarda str(Path(...)).
    assert Path(d["metadata"]["source_path_previo"]) == Path("E:/disco viejo/clase 3.mp4")
    assert d["metadata"]["duration_seconds"] == 60.0


def test_apply_deja_un_backup_con_el_contenido_original(corpus):
    base, _ts, roto, _f = corpus
    antes = roto.read_text(encoding="utf-8")
    correr("--apply")
    copia = base / "backup" / roto.name
    assert copia.read_text(encoding="utf-8") == antes


def test_apply_no_deja_archivos_temporales(corpus):
    """La escritura es atómica: `.tmp` y `os.replace`, porque `kb.extract`
    puede estar leyendo estos mismos archivos. Un `.tmp` que queda es un
    archivo que el próximo glob no lee pero que ocupa y confunde."""
    _base, ts, _roto, _f = corpus
    correr("--apply")
    assert [p.name for p in ts.iterdir()] == ["video_clase 3.json"]


def test_el_texto_no_se_toca(corpus):
    """El script arregla rutas. Si tocara el texto, cambiaría lo que el
    extractor saca de ahí y lo que el índice devuelve."""
    _base, _ts, roto, _f = corpus
    correr("--apply")
    d = json.loads(roto.read_text(encoding="utf-8"))
    assert d["full_text"] == "hola, esto es una clase"
    assert d["source_id"] == "video_clase 3"


def test_un_transcript_con_fuente_existente_no_se_toca(corpus):
    _base, ts, _roto, fuente = corpus
    sano = ts / "video_sano.json"
    sano.write_text(json.dumps({
        "source_id": "video_sano", "source_path": str(fuente), "full_text": "ok",
    }), encoding="utf-8")
    antes = sano.read_bytes()
    correr("--apply")
    assert sano.read_bytes() == antes


def test_sin_nada_roto_no_crea_el_backup(corpus):
    base, ts, roto, fuente = corpus
    roto.unlink()
    (ts / "video_sano.json").write_text(json.dumps({
        "source_id": "video_sano", "source_path": str(fuente), "full_text": "ok",
    }), encoding="utf-8")
    assert correr("--apply") == 0
    assert not (base / "backup").exists()


def test_un_json_ilegible_no_detiene_la_corrida(corpus, capsys):
    _base, ts, roto, fuente = corpus
    (ts / "roto_json.json").write_text("{truncado", encoding="utf-8")
    assert correr("--apply") == 0
    assert Path(json.loads(roto.read_text(encoding="utf-8"))["source_path"]) == fuente
    assert "ILEGIBLE" in capsys.readouterr().out


def test_sin_carpeta_de_transcripts_devuelve_error(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "TRANSCRIPTS", tmp_path / "no_existe")
    assert correr() == 1


def test_un_transcript_sin_source_path_no_es_un_roto(corpus, capsys):
    """Los transcripts viejos no anotaban la ruta. No hay nada que reparar ahí,
    y contarlos como rotos infla el informe con trabajo que no existe."""
    _base, ts, roto, _f = corpus
    roto.unlink()
    (ts / "viejo.json").write_text(
        json.dumps({"source_id": "viejo", "full_text": "ok"}), encoding="utf-8")
    assert correr() == 0
    assert "con source_path roto: 0" in capsys.readouterr().out
