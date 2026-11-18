"""Tests de los dos ingestores que se escribieron para cerrar huecos del corpus.

Cubren los dos errores que cuestan material sin avisar: una presentacion cuyas
notas del orador se pegan a la diapositiva equivocada, y un PDF con capa de
texto pobre que nunca llega al OCR y queda registrado como cubierto.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from coach_brain.ingest.docs import _pptx_slide_text, extract_pptx
from coach_brain.ingest import pdfs

SLIDE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody>
      <a:p><a:r><a:t>Cliente</a:t></a:r><a:r><a:t> ideal</a:t></a:r></a:p>
      <a:p><a:r><a:t>Hombres de 20 a 35</a:t></a:r></a:p>
    </p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:sld>
"""

NOTES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
         xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody>
      <a:p><a:r><a:t>Nota del orador</a:t></a:r></a:p>
    </p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:notes>
"""

RELS_TMPL = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
    Target="../notesSlides/notesSlide{n}.xml"/>
</Relationships>
"""


def _make_pptx(path: Path, notes_number: int | None = None) -> Path:
    """Escribe un .pptx minimo pero valido para el extractor.

    `notes_number` es el numero de archivo de la notesSlide. A proposito no
    coincide con el de la diapositiva: PowerPoint no garantiza que coincidan.
    """
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ppt/slides/slide1.xml", SLIDE_XML)
        if notes_number is not None:
            z.writestr(f"ppt/notesSlides/notesSlide{notes_number}.xml", NOTES_XML)
            z.writestr(
                "ppt/slides/_rels/slide1.xml.rels",
                RELS_TMPL.format(n=notes_number),
            )
    return path


def test_pptx_un_parrafo_por_linea_y_corridas_pegadas():
    """Las corridas de un mismo parrafo van juntas; los parrafos, separados."""
    text = _pptx_slide_text(SLIDE_XML)
    assert text.splitlines() == ["Cliente ideal", "Hombres de 20 a 35"]


def test_pptx_notas_se_resuelven_por_rels_no_por_numero(tmp_path: Path):
    """La nota de notesSlide7 tiene que llegar a slide1, que es quien la referencia."""
    pptx = _make_pptx(tmp_path / "x.pptx", notes_number=7)
    parts = extract_pptx(pptx)
    assert len(parts) == 1
    assert parts[0].startswith("Cliente ideal")
    assert "Notas: Nota del orador" in parts[0]


def test_pptx_sin_notas_no_agrega_el_prefijo(tmp_path: Path):
    pptx = _make_pptx(tmp_path / "y.pptx", notes_number=None)
    parts = extract_pptx(pptx)
    assert parts == ["Cliente ideal\nHombres de 20 a 35"]


def _make_pdf(path: Path, textos: list[str]) -> Path:
    """PDF de una pagina por elemento de `textos`."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for t in textos:
        page = doc.new_page()
        if t:
            page.insert_text((72, 72), t, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_pdf_capa_de_texto_pobre_dispara_ocr(tmp_path: Path, monkeypatch):
    """Dos paginas, 20 caracteres: no es cero, pero es pobre. Tiene que rasterizar."""
    pdf = _make_pdf(tmp_path / "flaco.pdf", ["diez chars", "diez chars"])

    llamadas = []

    def fake_ocr(p, dpi=200):
        llamadas.append(p)
        return ["texto OCR de la primera pagina, mucho mas largo", "y de la segunda"]

    monkeypatch.setattr(pdfs, "_ocr_pages", fake_ocr)
    data = pdfs.extract_pdf(pdf, ocr=True)

    assert llamadas, "el OCR no corrio sobre un PDF con capa de texto pobre"
    assert data["metadata"]["engine"] == "texto+rapidocr"
    # Se queda el texto mas largo de cada pagina.
    assert "texto OCR de la primera pagina" in data["segments"][0]["text"]


def test_pdf_con_texto_suficiente_no_dispara_ocr(tmp_path: Path, monkeypatch):
    """Una pagina con texto de sobra no se rasteriza, aunque se pase --ocr."""
    pdf = _make_pdf(tmp_path / "gordo.pdf", ["palabra " * 120])

    def fake_ocr(p, dpi=200):  # pragma: no cover - no deberia correr
        raise AssertionError("el OCR corrio sobre un PDF que ya tenia texto")

    monkeypatch.setattr(pdfs, "_ocr_pages", fake_ocr)
    data = pdfs.extract_pdf(pdf, ocr=True)
    assert data["metadata"]["engine"] == "texto"


def test_pdf_sin_flag_ocr_nunca_rasteriza(tmp_path: Path, monkeypatch):
    """Sin --ocr, un PDF vacio se devuelve vacio: el OCR es opcional por costo."""
    pdf = _make_pdf(tmp_path / "vacio.pdf", ["", ""])

    def fake_ocr(p, dpi=200):  # pragma: no cover - no deberia correr
        raise AssertionError("el OCR corrio sin el flag --ocr")

    monkeypatch.setattr(pdfs, "_ocr_pages", fake_ocr)
    data = pdfs.extract_pdf(pdf, ocr=False)
    assert data["segments"] == []
    assert data["metadata"]["engine"] == "texto"


def test_pdf_sin_texto_con_ocr_queda_como_rapidocr(tmp_path: Path, monkeypatch):
    """Cuando no habia nada de texto, el motor registrado es solo rapidocr."""
    pdf = _make_pdf(tmp_path / "escaneado.pdf", ["", ""])

    monkeypatch.setattr(pdfs, "_ocr_pages", lambda p, dpi=200: ["pagina una", "pagina dos"])
    data = pdfs.extract_pdf(pdf, ocr=True)
    assert data["metadata"]["engine"] == "rapidocr"
    assert [s["text"] for s in data["segments"]] == ["pagina una", "pagina dos"]


# --- proporcion de espacios -------------------------------------------------
#
# pypdf no reconstruye los espacios cuando el PDF los implementa como
# posicionamiento en vez de glifos de espacio. Devuelve la cantidad correcta de
# caracteres, asi que el texto roto pasa cualquier control de volumen, pero es
# una tira pegada: "Estoesparaquevayasdiaadia". Nueve libros del corpus, 2,4
# millones de caracteres, se habian indexado asi.


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    """Reemplaza a pypdf para poder fijar exactamente lo que devuelve."""

    def __init__(self, textos: list[str]) -> None:
        self.pages = [_FakePage(t) for t in textos]


def _con_pypdf(monkeypatch, textos: list[str]) -> None:
    monkeypatch.setattr(pdfs, "PdfReader", lambda path: _FakeReader(textos))


def test_space_ratio_cuenta_sobre_el_total():
    assert pdfs._space_ratio([{"text": "a b c d"}]) == pytest.approx(3 / 7)
    assert pdfs._space_ratio([]) == 0.0
    assert pdfs._space_ratio([{"text": ""}]) == 0.0


def test_pdf_texto_pegado_usa_pymupdf(tmp_path: Path, monkeypatch):
    """pypdf devuelve volumen correcto sin espacios: gana pymupdf."""
    _con_pypdf(monkeypatch, ["Estoesparaquevayasdiaadia" * 20])
    bueno = [{"id": 0, "text": "Esto es para que vayas dia a dia " * 20,
              "start": None, "end": None, "page": 1}]
    monkeypatch.setattr(pdfs, "_pymupdf_segments", lambda path: bueno)

    data = pdfs.extract_pdf(tmp_path / "pegado.pdf")

    assert data["metadata"]["engine"] == "texto-pymupdf"
    assert data["full_text"].startswith("Esto es para que")
    assert pdfs._space_ratio(data["segments"]) >= pdfs.MIN_SPACE_RATIO


def test_pdf_con_espacios_no_consulta_pymupdf(tmp_path: Path, monkeypatch):
    """Si pypdf ya dio prosa, pymupdf no se toca: es el camino caro."""
    _con_pypdf(monkeypatch, ["Esto es prosa de verdad con espacios " * 10])

    def no_llamar(path):  # pragma: no cover - no deberia correr
        raise AssertionError("se llamo a pymupdf sobre texto que ya estaba bien")

    monkeypatch.setattr(pdfs, "_pymupdf_segments", no_llamar)
    data = pdfs.extract_pdf(tmp_path / "bien.pdf")
    assert data["metadata"]["engine"] == "texto"


def test_pdf_pymupdf_peor_no_reemplaza(tmp_path: Path, monkeypatch):
    """Si pymupdf no mejora los espacios, se queda el texto de pypdf.

    Es el caso de un PDF escaneado, donde los dos motores devuelven poco y
    nada: reemplazar seria perder lo poco que pypdf si habia sacado.
    """
    _con_pypdf(monkeypatch, ["Aa Bb" + "Ccccccccccccccccccccccccc" * 10])
    antes = "Aa Bb" + "Ccccccccccccccccccccccccc" * 10
    monkeypatch.setattr(
        pdfs, "_pymupdf_segments",
        lambda path: [{"id": 0, "text": "Ccc", "start": None, "end": None, "page": 1}],
    )

    data = pdfs.extract_pdf(tmp_path / "escaneado.pdf")

    assert data["metadata"]["engine"] == "texto"
    assert data["full_text"] == antes


def test_pdf_sin_texto_cae_a_pymupdf_igual(tmp_path: Path, monkeypatch):
    """El fallback viejo sigue vivo: con 0 segmentos, pymupdf entra."""
    _con_pypdf(monkeypatch, ["", ""])
    monkeypatch.setattr(
        pdfs, "_pymupdf_segments",
        lambda path: [{"id": 0, "text": "pagina con texto", "start": None,
                       "end": None, "page": 1}],
    )

    data = pdfs.extract_pdf(tmp_path / "vacio_para_pymupdf.pdf")

    assert data["metadata"]["engine"] == "texto-pymupdf"
    assert data["segments"][0]["text"] == "pagina con texto"


def test_pdf_engine_registra_pymupdf_y_ocr_juntos(tmp_path: Path, monkeypatch):
    """Si el texto lo rescato pymupdf y encima entro el OCR, se anotan los dos."""
    _con_pypdf(monkeypatch, ["Pegadopegadopegadopegado" * 5])
    monkeypatch.setattr(
        pdfs, "_pymupdf_segments",
        lambda path: [{"id": 0, "text": "poco texto con espacios", "start": None,
                       "end": None, "page": 1}],
    )
    monkeypatch.setattr(
        pdfs, "_ocr_pages",
        lambda p, dpi=200: ["texto OCR bastante mas largo que el de la capa"],
    )

    data = pdfs.extract_pdf(tmp_path / "mixto.pdf", ocr=True)

    assert data["metadata"]["engine"] == "texto-pymupdf+rapidocr"
