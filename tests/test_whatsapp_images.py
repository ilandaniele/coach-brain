"""Las imágenes del export de WhatsApp: vision con Haiku y OCR local.

Los dos módulos escriben **el mismo transcript**
(`transcripts/whatsapp_images_chats.json`), así que el que corre segundo pisa al
primero. Eso está acá como prueba, no como nota al pie.

Nada de esto llama a la API ni carga RapidOCR: el cliente de Anthropic y el
motor de OCR se reemplazan por dobles, y las imágenes se arman con PIL en
`tmp_path`.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import types

import pytest
from PIL import Image

from coach_brain.ingest import whatsapp_images as V
from coach_brain.ingest import whatsapp_images_ocr as O
from coach_brain.settings import settings


@pytest.fixture
def arbol(tmp_path, monkeypatch):
    """Mueve todo el árbol de datos a `tmp_path` y crea las carpetas base."""
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    (tmp_path / "raw" / "whatsapp").mkdir(parents=True)
    (tmp_path / "processed" / "transcripts").mkdir(parents=True)
    return tmp_path


def imagen(path, size=(40, 40), color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


# ===========================================================================
# vision (whatsapp_images.py)
# ===========================================================================

# --- _encode ---------------------------------------------------------------

def test_encode_devuelve_jpeg_en_base64(tmp_path):
    media, b64 = V._encode(imagen(tmp_path / "a.png"))
    assert media == "image/jpeg"
    assert Image.open(io.BytesIO(base64.standard_b64decode(b64))).format == "JPEG"


def test_una_imagen_grande_se_baja_a_1568(tmp_path):
    """Arriba de 1568 px Anthropic no mejora el resultado y cobra más."""
    _, b64 = V._encode(imagen(tmp_path / "grande.png", size=(3000, 1500)))
    img = Image.open(io.BytesIO(base64.standard_b64decode(b64)))
    assert max(img.size) == V.MAX_DIM
    assert img.size[0] / img.size[1] == pytest.approx(2.0, abs=0.01)


def test_una_imagen_chica_no_se_estira(tmp_path):
    _, b64 = V._encode(imagen(tmp_path / "chica.png", size=(100, 50)))
    assert Image.open(io.BytesIO(base64.standard_b64decode(b64))).size == (100, 50)


def test_un_png_con_transparencia_se_convierte_a_rgb(tmp_path):
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (20, 20), (1, 2, 3, 0)).save(p)
    assert V._encode(p) is not None  # JPEG no acepta alpha: hay convert("RGB")


def test_un_archivo_que_no_es_imagen_devuelve_none(tmp_path):
    p = tmp_path / "roto.jpg"
    p.write_bytes(b"no soy una imagen")
    assert V._encode(p) is None


def test_un_archivo_que_no_existe_devuelve_none(tmp_path):
    assert V._encode(tmp_path / "fantasma.png") is None


# --- _parse_json -----------------------------------------------------------

def test_un_json_limpio_se_parsea():
    d = V._parse_json('{"es_chat": true, "transcripcion": "EL: hola", "descripcion": "chat"}')
    assert d["es_chat"] is True
    assert d["transcripcion"] == "EL: hola"


def test_un_json_entre_fences_se_parsea():
    d = V._parse_json('```json\n{"es_chat": false, "descripcion": "meme"}\n```')
    assert d["es_chat"] is False
    assert d["descripcion"] == "meme"


def test_un_json_con_prosa_alrededor_se_rescata():
    d = V._parse_json('Claro, acá va:\n{"es_chat": true, "transcripcion": "ELLA: ok"}\nEso es todo.')
    assert d["es_chat"] is True


def test_una_respuesta_sin_json_cae_al_default():
    d = V._parse_json("no pude leer la imagen")
    assert d == {"es_chat": False, "transcripcion": "", "descripcion": "no pude leer la imagen"}


def test_la_descripcion_del_default_se_corta_en_200():
    d = V._parse_json("x" * 500)
    assert len(d["descripcion"]) == 200


def test_los_acentos_sobreviven_al_parseo():
    d = V._parse_json('{"es_chat": true, "transcripcion": "ELLA: mañana sí"}')
    assert d["transcripcion"] == "ELLA: mañana sí"


# --- _process_one ----------------------------------------------------------

class FakeResp:
    def __init__(self, texto):
        self.content = [types.SimpleNamespace(text=texto)]


class FakeClient:
    """Cliente de Anthropic falso: devuelve respuestas o levanta, en orden."""

    def __init__(self, *respuestas):
        self.respuestas = list(respuestas)
        self.llamadas: list[dict] = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.llamadas.append(kw)
        r = self.respuestas.pop(0) if self.respuestas else FakeResp("{}")
        if isinstance(r, Exception):
            raise r
        return r


def test_una_imagen_ya_procesada_se_saltea(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.json").write_text("{}", encoding="utf-8")
    cli = FakeClient()
    assert V._process_one(cli, "m", imagen(tmp_path / "a.png"), out) == "skip"
    assert cli.llamadas == []  # no se gastó una llamada


def test_una_imagen_ilegible_no_llega_al_modelo(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    p = tmp_path / "roto.png"
    p.write_bytes(b"basura")
    cli = FakeClient()
    assert V._process_one(cli, "m", p, out) == "err_open"
    assert cli.llamadas == []


def test_un_chat_se_guarda_y_se_cuenta_como_chat(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cli = FakeClient(FakeResp('{"es_chat": true, "transcripcion": "EL: hola", "descripcion": "chat"}'))
    assert V._process_one(cli, "modelo-x", imagen(tmp_path / "chat1.png"), out) == "chat"
    d = json.loads((out / "chat1.json").read_text(encoding="utf-8"))
    assert d["es_chat"] is True
    assert d["source_file"] == "chat1.png"  # queda de qué archivo salió


def test_una_imagen_que_no_es_chat_se_cuenta_como_other(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cli = FakeClient(FakeResp('{"es_chat": false, "descripcion": "un meme"}'))
    assert V._process_one(cli, "m", imagen(tmp_path / "meme.png"), out) == "other"


def test_la_llamada_manda_la_imagen_y_el_prompt(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cli = FakeClient(FakeResp("{}"))
    V._process_one(cli, "modelo-x", imagen(tmp_path / "a.png"), out)
    kw = cli.llamadas[0]
    assert kw["model"] == "modelo-x"
    contenido = kw["messages"][0]["content"]
    assert contenido[0]["type"] == "image"
    assert contenido[0]["source"]["media_type"] == "image/jpeg"
    assert contenido[1]["text"] == V.PROMPT


def test_un_rate_limit_se_reintenta(tmp_path, monkeypatch):
    """429 y 529 son transitorios: hay backoff, no se pierde la imagen."""
    monkeypatch.setattr(V.time, "sleep", lambda s: None)
    out = tmp_path / "out"
    out.mkdir()
    cli = FakeClient(
        RuntimeError("429 rate_limit_error"),
        RuntimeError("Overloaded"),
        FakeResp('{"es_chat": true, "transcripcion": "EL: hola"}'),
    )
    assert V._process_one(cli, "m", imagen(tmp_path / "a.png"), out) == "chat"
    assert len(cli.llamadas) == 3


def test_el_backoff_crece_y_tiene_techo(tmp_path, monkeypatch):
    esperas = []
    monkeypatch.setattr(V.time, "sleep", lambda s: esperas.append(s))
    out = tmp_path / "out"
    out.mkdir()
    cli = FakeClient(*[RuntimeError("429") for _ in range(5)])
    assert V._process_one(cli, "m", imagen(tmp_path / "a.png"), out) == "err_ratelimit"
    assert esperas == [2, 4, 8, 16, 30]  # 2**4*2 = 32, con techo de 30


def test_un_error_que_no_es_rate_limit_no_se_reintenta(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cli = FakeClient(ValueError("api key inválida"))
    assert V._process_one(cli, "m", imagen(tmp_path / "a.png"), out) == "err:ValueError"
    assert len(cli.llamadas) == 1
    assert not (out / "a.json").exists()  # un error no deja un JSON a medias


# --- aggregate (vision) ----------------------------------------------------

def guardar(out_dir, nombre, **campos):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / nombre).write_text(json.dumps(campos, ensure_ascii=False), encoding="utf-8")


def test_el_agregado_junta_solo_los_chats(arbol):
    out = arbol / "processed" / "whatsapp_images"
    guardar(out, "a.json", es_chat=True, transcripcion="EL: hola", source_file="a.png")
    guardar(out, "b.json", es_chat=False, transcripcion="", source_file="b.png")
    guardar(out, "c.json", es_chat=True, transcripcion="   ", source_file="c.png")
    V.aggregate()
    d = json.loads((arbol / "processed" / "transcripts" / "whatsapp_images_chats.json").read_text(encoding="utf-8"))
    assert d["metadata"]["chats"] == 1
    assert d["metadata"]["total_imagenes_procesadas"] == 3
    assert len(d["segments"]) == 1


def test_el_agregado_numera_los_segmentos_desde_cero(arbol):
    out = arbol / "processed" / "whatsapp_images"
    for n in ("a", "b", "c"):
        guardar(out, f"{n}.json", es_chat=True, transcripcion=f"EL: {n}", source_file=f"{n}.png")
    V.aggregate()
    d = json.loads((arbol / "processed" / "transcripts" / "whatsapp_images_chats.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in d["segments"]] == [0, 1, 2]
    assert "### Screenshot: a.png" in d["full_text"]
    assert d["full_text"].count("### Screenshot:") == 3


def test_un_json_corrupto_no_detiene_el_agregado(arbol):
    out = arbol / "processed" / "whatsapp_images"
    guardar(out, "ok.json", es_chat=True, transcripcion="EL: hola", source_file="ok.png")
    (out / "roto.json").write_text("{esto no es json", encoding="utf-8")
    V.aggregate()
    d = json.loads((arbol / "processed" / "transcripts" / "whatsapp_images_chats.json").read_text(encoding="utf-8"))
    assert d["metadata"]["chats"] == 1


def test_sin_chats_no_se_escribe_transcript(arbol):
    """Un transcript vacío se vería hecho y ninguna etapa lo volvería a mirar."""
    out = arbol / "processed" / "whatsapp_images"
    guardar(out, "a.json", es_chat=False, transcripcion="", source_file="a.png")
    V.aggregate()
    assert not (arbol / "processed" / "transcripts" / "whatsapp_images_chats.json").exists()


# ===========================================================================
# OCR local (whatsapp_images_ocr.py)
# ===========================================================================

# --- _is_noise -------------------------------------------------------------

@pytest.mark.parametrize("linea", [
    "8:33", "2:11 a.m.", "14:05 p. m.",      # reloj
    "29%", "100 %",                            # batería
    "enviado", "Entregado", "leído", "visto",
    "Escribe un mensaje", "escribí tu mensaje",
    "en línea", "online", "escribiendo...",
    "hoy", "Ayer", "mié", "sáb", "dom",
    "<", ">>", "•", "|", "三",
    "a", " ", "..", "12",
])
def test_el_chrome_de_whatsapp_es_ruido(linea):
    assert O._is_noise(linea) is True


@pytest.mark.parametrize("linea", [
    "hola como estas",
    "jaja si",
    "mañana te escribo",
    "ok",            # dos letras: corto pero con alfabéticas
    "Y vos que haces hoy a la tarde",
])
def test_el_contenido_real_no_es_ruido(linea):
    assert O._is_noise(linea) is False


def test_una_hora_dentro_de_una_frase_no_es_ruido():
    """El patrón está anclado: solo descarta la línea que *es* la hora."""
    assert O._is_noise("nos vemos 8:33 entonces") is False


# --- _ocr_image ------------------------------------------------------------

def caja(x, y, w=50, h=10):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def fake_ocr(lineas):
    """Devuelve un callable con la forma de RapidOCR: (resultado, tiempos)."""
    return lambda ruta: (lineas, None)


def test_el_lado_sale_de_la_x_del_borde_izquierdo(tmp_path):
    p = imagen(tmp_path / "chat.png", size=(100, 100))
    res = O._ocr_image(fake_ocr([
        (caja(10, 10), "hola", 0.9),    # izquierda -> OTRO
        (caja(60, 30), "hola vos", 0.9),  # derecha -> YO
    ]), p)
    assert res["transcripcion"] == "OTRO: hola\nYO: hola vos"


def test_las_lineas_se_ordenan_de_arriba_hacia_abajo(tmp_path):
    p = imagen(tmp_path / "chat.png", size=(100, 100))
    res = O._ocr_image(fake_ocr([
        (caja(10, 90), "la ultima", 0.9),
        (caja(10, 10), "la primera", 0.9),
    ]), p)
    assert res["transcripcion"] == "OTRO: la primera la ultima"


def test_dos_lineas_del_mismo_lado_se_unen_en_un_turno(tmp_path):
    p = imagen(tmp_path / "chat.png", size=(100, 100))
    res = O._ocr_image(fake_ocr([
        (caja(10, 10), "hola como", 0.9),
        (caja(10, 25), "estas", 0.9),
        (caja(60, 40), "bien vos", 0.9),
    ]), p)
    assert res["transcripcion"] == "OTRO: hola como estas\nYO: bien vos"


def test_el_ruido_sale_de_la_transcripcion_pero_queda_en_el_crudo(tmp_path):
    """El texto crudo es lo fiable; el etiquetado de hablante es aproximado."""
    p = imagen(tmp_path / "chat.png", size=(100, 100))
    res = O._ocr_image(fake_ocr([
        (caja(10, 5), "8:33", 0.9),
        (caja(10, 20), "hola que haces", 0.9),
        (caja(10, 95), "Escribe un mensaje", 0.9),
    ]), p)
    assert res["transcripcion"] == "OTRO: hola que haces"
    assert res["texto_crudo"] == "8:33\nhola que haces\nEscribe un mensaje"


def test_hacen_falta_tres_lineas_de_contenido_para_ser_chat(tmp_path):
    p = imagen(tmp_path / "chat.png", size=(100, 100))
    dos = O._ocr_image(fake_ocr([
        (caja(10, 10), "hola", 0.9),
        (caja(60, 30), "que haces", 0.9),
    ]), p)
    assert dos["es_chat"] is False
    assert dos["descripcion"] == "imagen con poco texto (OCR)"

    tres = O._ocr_image(fake_ocr([
        (caja(10, 10), "hola", 0.9),
        (caja(60, 30), "que haces", 0.9),
        (caja(10, 50), "nada y vos", 0.9),
    ]), p)
    assert tres["es_chat"] is True
    assert tres["descripcion"] == "screenshot de chat (OCR)"


def test_una_linea_vacia_se_ignora(tmp_path):
    p = imagen(tmp_path / "chat.png", size=(100, 100))
    res = O._ocr_image(fake_ocr([
        (caja(10, 10), "   ", 0.9),
        (caja(10, 20), None, 0.9),
        (caja(10, 30), "hola", 0.9),
    ]), p)
    assert res["texto_crudo"] == "hola"


def test_sin_texto_detectado_no_es_chat(tmp_path):
    p = imagen(tmp_path / "foto.png")
    res = O._ocr_image(fake_ocr([]), p)
    assert res["es_chat"] is False
    assert res["descripcion"] == "sin texto detectado"


def test_una_imagen_que_no_abre_se_informa_y_no_llama_al_ocr(tmp_path):
    p = tmp_path / "roto.png"
    p.write_bytes(b"basura")

    def explota(ruta):
        raise AssertionError("no tendría que llegar al OCR")

    res = O._ocr_image(explota, p)
    assert res["descripcion"] == "no se pudo abrir"
    assert res["engine"] == "rapidocr"


# --- run (OCR) -------------------------------------------------------------

@pytest.fixture
def motor(monkeypatch):
    """Inyecta un `rapidocr_onnxruntime` falso: la librería no hace falta."""
    vistas: list[str] = []

    class FakeRapidOCR:
        def __call__(self, ruta):
            vistas.append(ruta)
            return ([
                (caja(10, 10), "hola", 0.9),
                (caja(60, 30), "que haces", 0.9),
                (caja(10, 50), "nada y vos", 0.9),
            ], None)

    mod = types.ModuleType("rapidocr_onnxruntime")
    mod.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod)
    return vistas


def test_sin_carpeta_de_whatsapp_no_hace_nada(tmp_path, monkeypatch, motor):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    O.run()
    assert not (tmp_path / "processed" / "whatsapp_images_ocr").exists()


def test_una_imagen_del_export_deja_el_json_con_su_stem(arbol, motor):
    imagen(arbol / "raw" / "whatsapp" / "IMG-0001.jpg")
    O.run()
    out = arbol / "processed" / "whatsapp_images_ocr"
    assert (out / "IMG-0001.json").exists()
    d = json.loads((out / "IMG-0001.json").read_text(encoding="utf-8"))
    assert d["es_chat"] is True
    # El path es relativo a `data/raw`, con el separador del sistema.
    assert d["source_file"].replace("\\", "/") == "whatsapp/IMG-0001.jpg"


def test_una_imagen_de_las_raices_extra_lleva_sufijo(arbol, motor):
    """En `docs/` los nombres se repiten entre carpetas ("1.png"), así que un
    solo stem pisaría a otro."""
    imagen(arbol / "raw" / "docs" / "Interacciones" / "1.png")
    imagen(arbol / "raw" / "docs" / "Mensajes" / "1.png")
    O.run()
    salidas = sorted(p.name for p in (arbol / "processed" / "whatsapp_images_ocr").glob("*.json"))
    assert len(salidas) == 2
    assert all(n.startswith("1__") for n in salidas)
    assert salidas[0] != salidas[1]


def test_de_las_raices_extra_se_guarda_el_path_no_el_nombre(arbol, motor):
    imagen(arbol / "raw" / "docs" / "Interacciones" / "1.png")
    O.run()
    j = next((arbol / "processed" / "whatsapp_images_ocr").glob("*.json"))
    d = json.loads(j.read_text(encoding="utf-8"))
    assert "Interacciones" in d["source_file"]
    assert "1.png" in d["source_file"]


def test_solo_se_miran_las_extensiones_de_imagen(arbol, motor):
    imagen(arbol / "raw" / "whatsapp" / "a.jpg")
    imagen(arbol / "raw" / "whatsapp" / "b.PNG")
    (arbol / "raw" / "whatsapp" / "c.txt").write_text("hola", encoding="utf-8")
    (arbol / "raw" / "whatsapp" / "d.opus").write_bytes(b"x")
    O.run()
    assert len(list((arbol / "processed" / "whatsapp_images_ocr").glob("*.json"))) == 2


def test_una_imagen_ya_procesada_no_se_vuelve_a_ocrear(arbol, motor):
    imagen(arbol / "raw" / "whatsapp" / "a.jpg")
    O.run()
    assert len(motor) == 1
    O.run()
    assert len(motor) == 1  # resumable por existencia del JSON


def test_el_limite_corta_la_corrida(arbol, motor):
    for n in range(5):
        imagen(arbol / "raw" / "whatsapp" / f"a{n}.jpg")
    O.run(limit=2)
    assert len(motor) == 2


def test_una_imagen_que_falla_no_detiene_a_las_demas(arbol, monkeypatch, motor):
    imagen(arbol / "raw" / "whatsapp" / "a.jpg")
    imagen(arbol / "raw" / "whatsapp" / "b.jpg")
    real = O._ocr_image

    def a_veces_explota(ocr, path):
        if path.name == "a.jpg":
            raise RuntimeError("se cayó el motor")
        return real(ocr, path)

    monkeypatch.setattr(O, "_ocr_image", a_veces_explota)
    O.run()
    salidas = [p.name for p in (arbol / "processed" / "whatsapp_images_ocr").glob("*.json")]
    assert salidas == ["b.json"]


# --- aggregate (OCR) -------------------------------------------------------

def test_el_agregado_del_ocr_marca_el_motor(arbol):
    out = arbol / "processed" / "whatsapp_images_ocr"
    guardar(out, "a.json", es_chat=True, transcripcion="OTRO: hola", source_file="whatsapp/a.jpg")
    O.aggregate()
    d = json.loads((arbol / "processed" / "transcripts" / "whatsapp_images_chats.json").read_text(encoding="utf-8"))
    assert d["metadata"]["engine"] == "rapidocr"
    assert d["title"].endswith("OCR local")
    assert d["source_id"] == "whatsapp_images_chats"


def test_el_ocr_y_la_vision_escriben_el_mismo_archivo(arbol):
    """Los dos apuntan a `whatsapp_images_chats.json`: el que corre segundo
    pisa al primero. Está así a propósito (el resto del pipeline no cambia),
    pero significa que no se pueden usar los dos a la vez."""
    guardar(arbol / "processed" / "whatsapp_images", "v.json",
            es_chat=True, transcripcion="EL: vision", source_file="v.png")
    guardar(arbol / "processed" / "whatsapp_images_ocr", "o.json",
            es_chat=True, transcripcion="OTRO: ocr", source_file="o.png")
    destino = arbol / "processed" / "transcripts" / "whatsapp_images_chats.json"

    V.aggregate()
    assert "vision" in destino.read_text(encoding="utf-8")
    O.aggregate()
    texto = destino.read_text(encoding="utf-8")
    assert "ocr" in texto and "vision" not in texto


def test_sin_chats_el_ocr_tampoco_escribe(arbol):
    out = arbol / "processed" / "whatsapp_images_ocr"
    guardar(out, "a.json", es_chat=False, transcripcion="", source_file="a.png")
    O.aggregate()
    assert not (arbol / "processed" / "transcripts" / "whatsapp_images_chats.json").exists()
