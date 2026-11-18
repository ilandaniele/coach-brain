"""Ingesta de documentos: .docx, .epub y el barrido que los encuentra.

El .pptx ya está cubierto en `test_ingest_pptx_pdf.py`. Lo que falta acá es el
.docx (incluidas las tablas), el .epub, el registro que se escribe y sobre todo
`_collect_targets()`: los tres junctions de `data/raw` apuntan al mismo árbol,
así que sin deduplicar por path canónico cada libro se procesaría tres veces.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from coach_brain.ingest import docs as D
from coach_brain.settings import REPO_ROOT, settings

# --- helpers ---------------------------------------------------------------

def hacer_docx(path: Path, parrafos=(), tabla=None) -> Path:
    from docx import Document

    doc = Document()
    for p in parrafos:
        doc.add_paragraph(p)
    if tabla:
        t = doc.add_table(rows=len(tabla), cols=len(tabla[0]))
        for i, fila in enumerate(tabla):
            for j, celda in enumerate(fila):
                t.cell(i, j).text = celda
    doc.save(str(path))
    return path


def hacer_epub(path: Path, capitulos=()) -> Path:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test")
    book.set_title("Libro de prueba")
    book.set_language("es")
    items = []
    for i, html in enumerate(capitulos, start=1):
        c = epub.EpubHtml(title=f"cap {i}", file_name=f"cap{i}.xhtml", lang="es")
        c.content = html
        book.add_item(c)
        items.append(c)
    book.toc = tuple(items)
    book.spine = items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


# --- limpieza y rutas ------------------------------------------------------

def test_los_espacios_repetidos_se_colapsan():
    assert D._clean("hola    mundo\t\tque tal  ") == "hola mundo que tal"


def test_los_saltos_de_linea_no_se_tocan():
    """`_WS_RE` no incluye `\\n`: el .epub separa párrafos con saltos."""
    assert D._clean("uno\ndos") == "uno\ndos"


def test_una_ruta_del_repo_se_guarda_relativa():
    assert D._relpath(REPO_ROOT / "data" / "raw" / "x.docx") in (
        "data\\raw\\x.docx", "data/raw/x.docx")


def test_una_ruta_fuera_del_repo_se_guarda_absoluta(tmp_path):
    assert D._relpath(tmp_path / "x.docx") == str(tmp_path / "x.docx")


# --- .docx -----------------------------------------------------------------

def test_los_parrafos_del_docx_salen_en_orden(tmp_path):
    p = hacer_docx(tmp_path / "a.docx", ["primero", "segundo", "tercero"])
    assert D.extract_docx(p) == ["primero", "segundo", "tercero"]


def test_los_parrafos_vacios_del_docx_se_descartan(tmp_path):
    p = hacer_docx(tmp_path / "a.docx", ["texto", "", "   ", "más texto"])
    assert D.extract_docx(p) == ["texto", "más texto"]


def test_las_celdas_de_una_tabla_entran_separadas_por_barra(tmp_path):
    p = hacer_docx(tmp_path / "a.docx", ["intro"], tabla=[["apertura", "qué decir"]])
    assert D.extract_docx(p) == ["intro", "apertura | qué decir"]


def test_una_celda_vacia_no_deja_una_barra_colgando(tmp_path):
    p = hacer_docx(tmp_path / "a.docx", tabla=[["sola", ""]])
    assert D.extract_docx(p) == ["sola"]


def test_una_fila_entera_vacia_no_agrega_nada(tmp_path):
    p = hacer_docx(tmp_path / "a.docx", ["algo"], tabla=[["", ""]])
    assert D.extract_docx(p) == ["algo"]


def test_un_docx_sin_nada_devuelve_lista_vacia(tmp_path):
    assert D.extract_docx(hacer_docx(tmp_path / "a.docx")) == []


def test_los_espacios_raros_del_docx_se_normalizan(tmp_path):
    p = hacer_docx(tmp_path / "a.docx", ["hola     mundo"])
    assert D.extract_docx(p) == ["hola mundo"]


# --- .epub -----------------------------------------------------------------

def test_cada_capitulo_del_epub_es_un_bloque(tmp_path):
    p = hacer_epub(tmp_path / "a.epub",
                   ["<html><body><p>capítulo uno</p></body></html>",
                    "<html><body><p>capítulo dos</p></body></html>"])
    partes = D.extract_epub(p)
    assert any("capítulo uno" in b for b in partes)
    assert any("capítulo dos" in b for b in partes)


def test_las_etiquetas_html_no_llegan_al_texto(tmp_path):
    p = hacer_epub(tmp_path / "a.epub",
                   ["<html><body><h1>Título</h1><p><em>texto</em> suelto</p></body></html>"])
    todo = "\n".join(D.extract_epub(p))
    assert "<" not in todo and ">" not in todo
    assert "Título" in todo and "texto" in todo


def test_las_entidades_html_se_desescapan(tmp_path):
    p = hacer_epub(tmp_path / "a.epub",
                   ["<html><body><p>uno &amp; dos &quot;tres&quot;</p></body></html>"])
    todo = "\n".join(D.extract_epub(p))
    assert "uno & dos" in todo
    assert "&amp;" not in todo


def test_un_capitulo_sin_texto_no_deja_un_bloque_vacio(tmp_path):
    p = hacer_epub(tmp_path / "a.epub",
                   ["<html><body><div>   </div></body></html>",
                    "<html><body><p>lo único</p></body></html>"])
    partes = D.extract_epub(p)
    assert all(b.strip() for b in partes)
    assert any("lo único" in b for b in partes)


# --- el registro que se escribe -------------------------------------------

def test_el_registro_arma_el_source_id_con_el_tipo_y_el_nombre(tmp_path):
    r = D.build_record(tmp_path / "Mode One.docx", "docx", ["a"])
    assert r["source_id"] == "docx_Mode One"
    assert r["source_type"] == "docx"
    assert r["title"] == "Mode One"


def test_los_segmentos_se_numeran_desde_cero(tmp_path):
    r = D.build_record(tmp_path / "a.epub", "epub", ["uno", "dos", "tres"])
    assert [s["id"] for s in r["segments"]] == [0, 1, 2]
    assert [s["text"] for s in r["segments"]] == ["uno", "dos", "tres"]
    assert r["segments"][0]["start"] is None  # un documento no tiene tiempos


def test_el_texto_completo_separa_los_bloques_con_linea_en_blanco(tmp_path):
    r = D.build_record(tmp_path / "a.epub", "epub", ["uno", "dos"])
    assert r["full_text"] == "uno\n\ndos"


def test_la_metadata_cuenta_los_bloques_y_deja_el_resto_en_none(tmp_path):
    r = D.build_record(tmp_path / "a.docx", "docx", ["uno", "dos"])
    assert r["metadata"] == {"pages": None, "duration_seconds": None, "blocks": 2}


def test_el_idioma_sale_de_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "whisper_language", "pt")
    assert D.build_record(tmp_path / "a.docx", "docx", ["x"])["language"] == "pt"


# --- el barrido ------------------------------------------------------------

@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    docs = settings.raw_path / "docs"
    wa = settings.raw_path / "whatsapp"
    docs.mkdir(parents=True, exist_ok=True)
    wa.mkdir(parents=True, exist_ok=True)
    return {"docs": docs, "wa": wa}


def test_encuentra_las_tres_extensiones(raw):
    hacer_docx(raw["docs"] / "a.docx", ["x"])
    (raw["docs"] / "b.epub").write_bytes(b"x")
    (raw["docs"] / "c.pptx").write_bytes(b"x")
    assert [(p.name, k) for p, k in D._collect_targets()] == [
        ("a.docx", "docx"), ("b.epub", "epub"), ("c.pptx", "pptx")]


def test_ignora_lo_que_no_es_documento(raw):
    (raw["docs"] / "a.mp4").write_bytes(b"x")
    (raw["docs"] / "b.pdf").write_bytes(b"x")
    (raw["docs"] / "c.txt").write_text("x", encoding="utf-8")
    assert D._collect_targets() == []


def test_la_extension_en_mayuscula_tambien_cuenta(raw):
    (raw["docs"] / "a.DOCX").write_bytes(b"x")
    assert [k for _, k in D._collect_targets()] == ["docx"]


def test_busca_en_subcarpetas(raw):
    sub = raw["docs"] / "libros" / "2023"
    sub.mkdir(parents=True)
    (sub / "a.epub").write_bytes(b"x")
    assert [p.name for p, _ in D._collect_targets()] == ["a.epub"]


def test_tambien_mira_los_adjuntos_de_whatsapp(raw):
    """Hubo material que llegó como adjunto y esta etapa nunca lo vio."""
    (raw["wa"] / "guia.docx").write_bytes(b"x")
    assert [p.name for p, _ in D._collect_targets()] == ["guia.docx"]


def test_un_root_que_no_existe_no_rompe_el_barrido(raw):
    import shutil
    shutil.rmtree(raw["wa"])
    (raw["docs"] / "a.epub").write_bytes(b"x")
    assert [p.name for p, _ in D._collect_targets()] == ["a.epub"]


def test_sale_ordenado_por_nombre(raw):
    for n in ("zeta.epub", "alfa.epub", "media.epub"):
        (raw["docs"] / n).write_bytes(b"x")
    assert [p.name for p, _ in D._collect_targets()] == [
        "alfa.epub", "media.epub", "zeta.epub"]


def test_el_mismo_archivo_visto_dos_veces_se_procesa_una(raw):
    """Los junctions de `data/raw` apuntan al mismo árbol: sin deduplicar por
    path canónico, cada libro entraría una vez por root."""
    real = raw["docs"] / "libro.epub"
    real.write_bytes(b"x")
    try:
        (raw["wa"] / "libro.epub").symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("este Windows no deja crear symlinks sin privilegios")
    assert len(D._collect_targets()) == 1


# --- la corrida ------------------------------------------------------------

def test_la_corrida_escribe_un_json_por_documento(raw):
    hacer_docx(raw["docs"] / "guia.docx", ["un consejo"])
    D.run()
    out = settings.transcripts_path / "docx_guia.json"
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["source_id"] == "docx_guia"
    assert d["full_text"] == "un consejo"


def test_sin_documentos_no_escribe_nada(raw):
    D.run()
    assert not settings.transcripts_path.exists() or not list(
        settings.transcripts_path.glob("*.json"))


def test_no_rehace_un_documento_ya_procesado(raw):
    hacer_docx(raw["docs"] / "guia.docx", ["nuevo"])
    settings.transcripts_path.mkdir(parents=True, exist_ok=True)
    out = settings.transcripts_path / "docx_guia.json"
    out.write_text('{"source_id": "viejo"}', encoding="utf-8")
    D.run()
    assert json.loads(out.read_text(encoding="utf-8"))["source_id"] == "viejo"


def test_con_force_lo_rehace(raw):
    hacer_docx(raw["docs"] / "guia.docx", ["nuevo"])
    settings.transcripts_path.mkdir(parents=True, exist_ok=True)
    out = settings.transcripts_path / "docx_guia.json"
    out.write_text('{"source_id": "viejo"}', encoding="utf-8")
    D.run(force=True)
    assert json.loads(out.read_text(encoding="utf-8"))["source_id"] == "docx_guia"


def test_un_documento_sin_texto_no_deja_un_json_vacio(raw):
    """Un archivo de 0 bloques no se escribe: un transcript vacío se vería
    hecho y ninguna etapa posterior lo volvería a mirar."""
    hacer_docx(raw["docs"] / "mudo.docx")
    D.run()
    assert not (settings.transcripts_path / "docx_mudo.json").exists()


def test_un_documento_roto_no_detiene_a_los_demas(raw):
    (raw["docs"] / "roto.epub").write_bytes(b"esto no es un epub")
    hacer_docx(raw["docs"] / "sano.docx", ["sirve"])
    D.run()
    assert (settings.transcripts_path / "docx_sano.json").exists()
    assert not (settings.transcripts_path / "epub_roto.json").exists()


def test_un_pptx_de_verdad_llega_hasta_el_json(raw):
    """Prueba de punta a punta del camino pptx, armando el zip a mano."""
    p = raw["docs"] / "clase.pptx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("ppt/slides/slide1.xml",
                   "<p:sld><a:p><a:t>Apertura</a:t></a:p>"
                   "<a:p><a:t>directa</a:t><a:t> y corta</a:t></a:p></p:sld>")
    D.run()
    d = json.loads((settings.transcripts_path / "pptx_clase.json").read_text(encoding="utf-8"))
    assert d["full_text"] == "Apertura\ndirecta y corta"
    assert d["metadata"]["blocks"] == 1
