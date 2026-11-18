"""Tests del OCR de PDFs y del barrido que los mete al pipeline.

`tests/test_ingest_pptx_pdf.py` cubre la arbitración de `extract_pdf`: cuándo
gana pymupdf sobre pypdf por proporción de espacios y cuándo se dispara el OCR.
Lo que no estaba probado es lo que pasa a los dos costados de esa decisión:

- **`_ocr_pages`**, el rasterizado en sí. Es el camino que recuperó los 32 PDFs
  que extrajeron CERO caracteres, cuesta varios segundos por página y toca tres
  dependencias opcionales (pymupdf, numpy, rapidocr). Un error acá no se ve en
  el informe: el PDF queda sin texto y el barrido lo cuenta como "escaneado".
- **`run`**, el barrido. Recorre **dos** raíces —`raw/pdfs`, que es una junction
  al material, y `raw/whatsapp`, que tiene los PDFs que llegaron adjuntos en los
  chats— y deduplica por path canónico porque las junctions de `data/raw`
  apuntan todas al mismo árbol. Sin la segunda raíz esos adjuntos nunca entraban
  al pipeline; sin la deduplicación cada PDF se procesaría tres veces.

Lo que se fija de `run` es sobre todo su contabilidad: un PDF sin texto no tiene
que dejar un JSON vacío en `transcripts/`, porque de ahí se arma el índice, y un
PDF roto no tiene que detener a los otros 61.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from coach_brain.ingest import pdfs
from coach_brain.settings import settings


def hacer_pdf(path: Path, textos: list[str]) -> Path:
    """PDF real de una página por elemento de `textos`."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for t in textos:
        page = doc.new_page()
        if t:
            page.insert_text((72, 72), t, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def un_pdf(path: Path) -> Path:
    """PDF mínimo cuando el contenido no importa porque `extract_pdf` está falseado."""
    return hacer_pdf(path, ["x"])


# --- _ocr_pages: el rasterizado, con pymupdf y numpy de verdad --------------


class MotorFalso:
    """Reemplaza a RapidOCR. Guarda cada imagen que recibe."""

    def __init__(self, filas_por_pagina=None) -> None:
        self.imagenes: list[np.ndarray] = []
        self.filas = filas_por_pagina

    def __call__(self, img):
        self.imagenes.append(img)
        if self.filas is None:
            return [[None, f"pagina {len(self.imagenes)}", 0.9]], 0.0
        return self.filas[len(self.imagenes) - 1], 0.0


def test_el_ocr_devuelve_una_entrada_por_pagina(tmp_path, monkeypatch):
    pdf = hacer_pdf(tmp_path / "dos.pdf", ["", ""])
    monkeypatch.setattr(pdfs, "_OCR_ENGINE", MotorFalso())
    assert pdfs._ocr_pages(pdf) == ["pagina 1", "pagina 2"]


def test_las_filas_se_unen_con_salto_de_linea(tmp_path, monkeypatch):
    """RapidOCR devuelve una fila por renglón detectado, no una por página."""
    pdf = hacer_pdf(tmp_path / "una.pdf", [""])
    filas = [[[None, "primer renglon", 0.9], [None, "segundo renglon", 0.8]]]
    monkeypatch.setattr(pdfs, "_OCR_ENGINE", MotorFalso(filas))
    assert pdfs._ocr_pages(pdf) == ["primer renglon\nsegundo renglon"]


@pytest.mark.parametrize(
    "resultado",
    [
        [],
        None,
        [[None, "", 0.9]],
        [[None, None, 0.9]],
        [None],
        [[None]],
    ],
    ids=["sin filas", "None", "texto vacio", "texto None", "fila None", "fila corta"],
)
def test_una_pagina_sin_texto_legible_queda_vacia(tmp_path, monkeypatch, resultado):
    """Una página en blanco, o una fila que el motor devolvió a medias, tiene que
    dar cadena vacía y no romper el barrido. Después `extract_pdf` la saltea y se
    queda con la capa de texto que ya tenía."""
    pdf = hacer_pdf(tmp_path / "blanco.pdf", [""])
    monkeypatch.setattr(pdfs, "_OCR_ENGINE", MotorFalso([resultado]))
    assert pdfs._ocr_pages(pdf) == [""]


def test_el_motor_recibe_una_imagen_rgb_de_enteros(tmp_path, monkeypatch):
    """RapidOCR no acepta canal alfa ni floats. Si la imagen llegara mal, el OCR
    devolvería vacío y el PDF se contaría como escaneado."""
    pdf = hacer_pdf(tmp_path / "una.pdf", ["hola"])
    motor = MotorFalso()
    monkeypatch.setattr(pdfs, "_OCR_ENGINE", motor)
    pdfs._ocr_pages(pdf)
    img = motor.imagenes[0]
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_el_dpi_decide_el_tamano_de_la_imagen(tmp_path, monkeypatch):
    """El zoom es `dpi/72`: a 144 dpi la página mide el doble que a 72. Es lo que
    separa un OCR que lee de uno que devuelve ruido, y no hay nada en el informe
    que lo delate."""
    pdf = hacer_pdf(tmp_path / "una.pdf", ["hola"])
    motor = MotorFalso()
    monkeypatch.setattr(pdfs, "_OCR_ENGINE", motor)
    pdfs._ocr_pages(pdf, dpi=72)
    pdfs._ocr_pages(pdf, dpi=144)
    chico, grande = motor.imagenes
    assert grande.shape[0] == pytest.approx(chico.shape[0] * 2, abs=2)
    assert grande.shape[1] == pytest.approx(chico.shape[1] * 2, abs=2)


def test_el_motor_de_ocr_se_arma_una_sola_vez(tmp_path, monkeypatch):
    """`RapidOCR()` carga los modelos ONNX: armarlo por página multiplicaría el
    costo del paso más caro del pipeline por la cantidad de páginas."""
    pdf = hacer_pdf(tmp_path / "una.pdf", [""])
    armados = []

    class FakeRapid:
        def __init__(self) -> None:
            armados.append(1)

        def __call__(self, img):
            return [[None, "x", 0.9]], 0.0

    modulo = type(sys)("rapidocr_onnxruntime")
    modulo.RapidOCR = FakeRapid
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", modulo)
    monkeypatch.setattr(pdfs, "_OCR_ENGINE", None)

    pdfs._ocr_pages(pdf)
    pdfs._ocr_pages(pdf)
    assert armados == [1]


# --- _ocr_pages con un pymupdf falso: lo que no se puede provocar con uno real


class PixFalso:
    def __init__(self, h: int, w: int, n: int) -> None:
        self.height, self.width, self.n = h, w, n
        # Cada pixel numerado por canal, para poder ver cual se recorto.
        self.samples = bytes(range(n)) * (h * w)


class PaginaFalsa:
    def __init__(self, pix: PixFalso) -> None:
        self._pix = pix

    def get_pixmap(self, matrix=None):
        return self._pix


class DocFalso:
    def __init__(self, pixes: list[PixFalso]) -> None:
        self._pags = [PaginaFalsa(p) for p in pixes]
        self.cerrado = False

    def __iter__(self):
        return iter(self._pags)

    def close(self) -> None:
        self.cerrado = True


def fake_pymupdf(monkeypatch, doc, *, al_abrir=None):
    modulo = type(sys)("pymupdf")

    def abrir(path):
        if al_abrir is not None:
            raise al_abrir
        return doc

    modulo.open = abrir
    modulo.Matrix = lambda a, b: (a, b)
    monkeypatch.setitem(sys.modules, "pymupdf", modulo)
    return modulo


def test_una_imagen_rgba_se_recorta_a_rgb(tmp_path, monkeypatch):
    """Un PDF con transparencia da un pixmap de 4 canales. RapidOCR no acepta el
    alfa, así que se recorta: si no, el OCR falla y la página queda sin texto."""
    doc = DocFalso([PixFalso(2, 3, 4)])
    fake_pymupdf(monkeypatch, doc)
    motor = MotorFalso()
    monkeypatch.setattr(pdfs, "_OCR_ENGINE", motor)

    pdfs._ocr_pages(tmp_path / "rgba.pdf")

    img = motor.imagenes[0]
    assert img.shape == (2, 3, 3)
    # Se van los canales de alfa, no los de color.
    assert img[0, 0].tolist() == [0, 1, 2]


def test_una_imagen_rgb_pasa_entera(tmp_path, monkeypatch):
    doc = DocFalso([PixFalso(2, 3, 3)])
    fake_pymupdf(monkeypatch, doc)
    motor = MotorFalso()
    monkeypatch.setattr(pdfs, "_OCR_ENGINE", motor)

    pdfs._ocr_pages(tmp_path / "rgb.pdf")
    assert motor.imagenes[0].shape == (2, 3, 3)


def test_el_pdf_se_cierra_aunque_el_motor_falle(tmp_path, monkeypatch):
    """Un PDF abierto y no cerrado deja el archivo tomado en Windows, y el
    barrido corre sobre 62 archivos seguidos."""
    doc = DocFalso([PixFalso(2, 2, 3)])
    fake_pymupdf(monkeypatch, doc)

    def explotar(img):
        raise RuntimeError("modelo no cargado")

    monkeypatch.setattr(pdfs, "_OCR_ENGINE", explotar)

    with pytest.raises(RuntimeError):
        pdfs._ocr_pages(tmp_path / "x.pdf")
    assert doc.cerrado


# --- _pymupdf_segments: los dos modos de no estar disponible ----------------


def test_sin_pymupdf_devuelve_vacio_y_lo_dice(tmp_path, monkeypatch, capsys):
    """pymupdf es opcional. Si falta, `extract_pdf` se queda con lo de pypdf; lo
    que no puede pasar es que el ingestor entero se caiga."""
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    assert pdfs._pymupdf_segments(tmp_path / "x.pdf") == []
    assert "pymupdf no disponible" in capsys.readouterr().out


def test_si_pymupdf_no_puede_abrir_el_archivo_devuelve_vacio(tmp_path, monkeypatch, capsys):
    """Un PDF cifrado o truncado hace fallar `open`. Tiene que quedar nombrado en
    el aviso: es la única pista de por qué ese PDF salió peor de lo esperado."""
    fake_pymupdf(monkeypatch, None, al_abrir=RuntimeError("no es un pdf"))
    assert pdfs._pymupdf_segments(tmp_path / "roto.pdf") == []
    salida = capsys.readouterr().out
    assert "roto.pdf" in salida
    assert "no es un pdf" in salida


def test_las_paginas_sin_texto_no_entran_como_segmentos(tmp_path, monkeypatch):
    """Un segmento vacío sería un chunk vacío en el índice."""

    class Pag:
        def __init__(self, t):
            self._t = t

        def get_text(self):
            return self._t

    class Doc:
        def __init__(self):
            self._p = [Pag("hola"), Pag(""), Pag("   "), Pag(None), Pag("chau")]

        def __iter__(self):
            return iter(self._p)

        def close(self):
            pass

    fake_pymupdf(monkeypatch, Doc())
    segs = pdfs._pymupdf_segments(tmp_path / "x.pdf")
    assert [s["text"] for s in segs] == ["hola", "chau"]
    # La pagina se numera desde 1 y conserva el indice real del documento.
    assert [(s["id"], s["page"]) for s in segs] == [(0, 1), (4, 5)]


# --- extract_pdf: lo que el OCR no puede empeorar ---------------------------


class _PagFalsa:
    def __init__(self, t: str) -> None:
        self._t = t

    def extract_text(self) -> str:
        return self._t


def con_pypdf(monkeypatch, textos: list[str]) -> None:
    class Reader:
        def __init__(self, path) -> None:
            self.pages = [_PagFalsa(t) for t in textos]

    monkeypatch.setattr(pdfs, "PdfReader", Reader)


def test_una_pagina_que_el_ocr_no_leyo_no_borra_la_capa_de_texto(tmp_path, monkeypatch):
    """El OCR devuelve "" para una página que no pudo leer. Escribirla encima
    perdería el texto que pypdf sí había sacado."""
    con_pypdf(monkeypatch, ["titulo del cuadro", "otro titulo"])
    monkeypatch.setattr(
        pdfs, "_ocr_pages", lambda p, dpi=200: ["", "texto OCR largo de la segunda"]
    )
    data = pdfs.extract_pdf(tmp_path / "canva.pdf", ocr=True)
    assert data["segments"][0]["text"] == "titulo del cuadro"
    assert data["segments"][1]["text"] == "texto OCR largo de la segunda"


def test_un_ocr_mas_corto_que_la_capa_de_texto_no_la_reemplaza(tmp_path, monkeypatch):
    """Se queda el que más texto trae. Y si ninguna página mejoró, el motor sigue
    siendo el que de verdad dio el texto: decir "rapidocr" haría creer que el
    archivo se rasterizó para algo."""
    con_pypdf(monkeypatch, ["titulo que igual es corto"])
    monkeypatch.setattr(pdfs, "_ocr_pages", lambda p, dpi=200: ["corto"])
    data = pdfs.extract_pdf(tmp_path / "canva.pdf", ocr=True)
    assert data["segments"][0]["text"] == "titulo que igual es corto"
    assert data["metadata"]["engine"] == "texto"


def test_si_el_ocr_falla_el_pdf_sale_con_lo_que_habia(tmp_path, monkeypatch, capsys):
    """El OCR es el paso más frágil: carga modelos ONNX y rasteriza. Si se cae,
    el PDF tiene que salir con la capa de texto, no perderse entero."""
    con_pypdf(monkeypatch, ["titulo del cuadro"])

    def explotar(p, dpi=200):
        raise RuntimeError("onnxruntime no esta")

    monkeypatch.setattr(pdfs, "_ocr_pages", explotar)
    data = pdfs.extract_pdf(tmp_path / "canva.pdf", ocr=True)

    assert data["full_text"] == "titulo del cuadro"
    assert data["metadata"]["engine"] == "texto"
    salida = capsys.readouterr().out
    assert "canva.pdf" in salida
    assert "onnxruntime no esta" in salida


# --- run(): el barrido de las dos raices -----------------------------------


@pytest.fixture
def raices(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    pdfs_dir = settings.raw_path / "pdfs"
    wa = settings.raw_path / "whatsapp"
    pdfs_dir.mkdir(parents=True)
    wa.mkdir(parents=True)
    return {"pdfs": pdfs_dir, "wa": wa, "out": settings.transcripts_path}


def extract_falso(monkeypatch, *, vacio: tuple[str, ...] = (), falla: tuple[str, ...] = ()):
    """Reemplaza `extract_pdf` para probar la contabilidad de `run` sin PDFs reales."""
    vistos: list[Path] = []

    def fake(path: Path, ocr: bool = False) -> dict:
        vistos.append(path)
        if path.name in falla:
            raise RuntimeError("pdf ilegible")
        segs = [] if path.name in vacio else [{"id": 0, "text": "hola", "page": 1}]
        return {"source_id": f"pdf_{path.stem}", "segments": segs, "full_text": "hola"}

    monkeypatch.setattr(pdfs, "extract_pdf", fake)
    return vistos


def test_barre_las_dos_raices(raices, monkeypatch):
    """`raw/whatsapp` es la raíz que mete los PDFs que llegaron adjuntos en los
    chats. Sin ella no entraban nunca al pipeline."""
    un_pdf(raices["pdfs"] / "libro.pdf")
    un_pdf(raices["wa"] / "adjunto.pdf")
    extract_falso(monkeypatch)

    pdfs.run()

    assert sorted(p.name for p in raices["out"].iterdir()) == [
        "pdf_adjunto.json",
        "pdf_libro.json",
    ]


def test_tambien_entra_lo_que_esta_en_subcarpetas(raices, monkeypatch):
    hondo = raices["pdfs"] / "Bonus" / "2019"
    hondo.mkdir(parents=True)
    un_pdf(hondo / "clase.pdf")
    extract_falso(monkeypatch)
    pdfs.run()
    assert (raices["out"] / "pdf_clase.json").exists()


def test_la_extension_se_compara_sin_mayusculas(raices, monkeypatch):
    un_pdf(raices["pdfs"] / "grito.PDF")
    (raices["pdfs"] / "notas.txt").write_text("no soy un pdf", encoding="utf-8")
    vistos = extract_falso(monkeypatch)
    pdfs.run()
    assert [p.name for p in vistos] == ["grito.PDF"]


def test_el_mismo_archivo_por_dos_caminos_se_cuenta_una_vez(raices, monkeypatch, capsys):
    """Las tres junctions de `data/raw` apuntan todas a `Seduccion`, así que el
    mismo PDF aparece por varios caminos dentro de la misma raíz. El informe
    tiene que contar archivos, no caminos: con 62 PDFs vistos por tres junctions
    diría 186, y cada corrida parecería tener material nuevo que procesar."""
    espejo = raices["pdfs"] / "espejo"
    espejo.mkdir()
    un_pdf(raices["pdfs"] / "libro.pdf")
    un_pdf(espejo / "libro.pdf")

    real = os.path.realpath

    def con_junction(ruta):
        # `espejo/` se comporta como la junction: el path real es el de arriba.
        return real(ruta).replace(f"espejo{os.sep}", "")

    extract_falso(monkeypatch)

    # Sin canonizar, los dos caminos son dos archivos distintos. Es lo que hace
    # que la deduplicacion sea el unico motivo de que despues de uno.
    pdfs.run()
    assert "Procesando 2 PDF(s)" in capsys.readouterr().out

    monkeypatch.setattr(os.path, "realpath", con_junction)
    pdfs.run(force=True)
    assert "Procesando 1 PDF(s)" in capsys.readouterr().out


def test_no_vuelve_a_procesar_lo_que_ya_esta(raices, monkeypatch):
    """El barrido se corre de nuevo cada vez que entra material nuevo: 62 PDFs,
    varios de ellos libros de cientos de páginas."""
    un_pdf(raices["pdfs"] / "libro.pdf")
    raices["out"].mkdir(parents=True, exist_ok=True)
    hecho = raices["out"] / "pdf_libro.json"
    hecho.write_text('{"ya": "estaba"}', encoding="utf-8")
    vistos = extract_falso(monkeypatch)

    pdfs.run()

    assert vistos == []
    assert json.loads(hecho.read_text(encoding="utf-8")) == {"ya": "estaba"}


def test_con_force_lo_reescribe(raices, monkeypatch):
    un_pdf(raices["pdfs"] / "libro.pdf")
    raices["out"].mkdir(parents=True, exist_ok=True)
    hecho = raices["out"] / "pdf_libro.json"
    hecho.write_text('{"ya": "estaba"}', encoding="utf-8")
    extract_falso(monkeypatch)

    pdfs.run(force=True)

    assert json.loads(hecho.read_text(encoding="utf-8"))["source_id"] == "pdf_libro"


def test_un_pdf_sin_texto_no_deja_un_json(raices, monkeypatch, capsys):
    """De `transcripts/` se arma el índice. Un JSON sin segmentos sería un
    documento que el buscador puede devolver con las manos vacías."""
    un_pdf(raices["pdfs"] / "escaneado.pdf")
    extract_falso(monkeypatch, vacio=("escaneado.pdf",))

    pdfs.run()

    assert not (raices["out"] / "pdf_escaneado.json").exists()
    salida = capsys.readouterr().out
    assert "escaneado.pdf" in salida
    assert "--ocr" in salida


def test_con_ocr_el_aviso_ya_no_sugiere_ocr(raices, monkeypatch, capsys):
    """Si ya se corrió con --ocr y el PDF igual salió vacío, repetir la
    sugerencia haría perder horas rasterizando lo mismo de nuevo."""
    un_pdf(raices["pdfs"] / "escaneado.pdf")
    extract_falso(monkeypatch, vacio=("escaneado.pdf",))

    pdfs.run(ocr=True)

    assert "--ocr" not in capsys.readouterr().out


def test_el_flag_de_ocr_llega_a_la_extraccion(raices, monkeypatch):
    un_pdf(raices["pdfs"] / "libro.pdf")
    recibido = []
    monkeypatch.setattr(
        pdfs,
        "extract_pdf",
        lambda path, ocr=False: (
            recibido.append(ocr),
            {"segments": [{"text": "x"}], "source_id": "pdf_libro"},
        )[1],
    )
    pdfs.run(ocr=True)
    assert recibido == [True]


def test_un_pdf_roto_no_detiene_a_los_demas(raices, monkeypatch, capsys):
    """Son 62 PDFs en una corrida. Que uno cifrado la cortara dejaría los otros
    61 sin procesar y el motivo sería un traceback."""
    un_pdf(raices["pdfs"] / "a_roto.pdf")
    un_pdf(raices["pdfs"] / "b_sano.pdf")
    extract_falso(monkeypatch, falla=("a_roto.pdf",))

    pdfs.run()

    assert (raices["out"] / "pdf_b_sano.json").exists()
    assert not (raices["out"] / "pdf_a_roto.json").exists()
    salida = capsys.readouterr().out
    assert "a_roto.pdf" in salida
    assert "errores=1" in salida


def test_el_informe_cuenta_las_cuatro_categorias(raices, monkeypatch, capsys):
    for n in ("ok.pdf", "vacio.pdf", "roto.pdf", "hecho.pdf"):
        un_pdf(raices["pdfs"] / n)
    raices["out"].mkdir(parents=True, exist_ok=True)
    (raices["out"] / "pdf_hecho.json").write_text("{}", encoding="utf-8")
    extract_falso(monkeypatch, vacio=("vacio.pdf",), falla=("roto.pdf",))

    pdfs.run()

    assert "ok=1 skip=1 sin_texto=1 errores=1" in capsys.readouterr().out


def test_sin_pdfs_avisa_y_no_escribe_nada(raices, monkeypatch, capsys):
    def no_llamar(path, ocr=False):  # pragma: no cover - no deberia correr
        raise AssertionError("se extrajo un PDF que no existe")

    monkeypatch.setattr(pdfs, "extract_pdf", no_llamar)

    pdfs.run()

    assert "No hay PDFs" in capsys.readouterr().out
    assert list(raices["out"].glob("*.json")) == []


def test_una_raiz_que_no_existe_no_rompe_el_barrido(raices, monkeypatch):
    """`raw/pdfs` es una junction: si el disco con el material no está montada,
    la carpeta no existe. Los adjuntos de WhatsApp igual se procesan."""
    import shutil

    shutil.rmtree(raices["pdfs"])
    un_pdf(raices["wa"] / "adjunto.pdf")
    extract_falso(monkeypatch)

    pdfs.run()

    assert (raices["out"] / "pdf_adjunto.json").exists()


def test_el_json_queda_legible_y_con_acentos(raices, monkeypatch):
    """Se escribe con `ensure_ascii=False` e indentado porque estos archivos se
    leen a mano cuando hay que decidir si un PDF sirve."""
    un_pdf(raices["pdfs"] / "libro.pdf")
    monkeypatch.setattr(
        pdfs,
        "extract_pdf",
        lambda path, ocr=False: {
            "source_id": "pdf_libro",
            "segments": [{"id": 0, "text": "seducción", "page": 1}],
            "full_text": "seducción",
        },
    )

    pdfs.run()

    crudo = (raices["out"] / "pdf_libro.json").read_text(encoding="utf-8")
    assert "seducción" in crudo
    assert "\n  " in crudo


def test_el_barrido_de_punta_a_punta_con_un_pdf_real(raices):
    """Sin falsear nada: pypdf, la arbitración y la escritura."""
    hacer_pdf(raices["pdfs"] / "clase.pdf", ["Esto es una clase de verdad con espacios"])

    pdfs.run()

    d = json.loads((raices["out"] / "pdf_clase.json").read_text(encoding="utf-8"))
    assert d["source_type"] == "pdf"
    assert d["metadata"]["pages"] == 1
    assert d["metadata"]["duration_seconds"] is None
    assert "clase de verdad" in d["full_text"]
