"""Tests de la etapa de extracción: chunking, parseo de la respuesta y resumible.

Hasta acá la extracción no tenía una sola prueba, y es la etapa que más daño
silencioso puede hacer, porque es la única que decide qué llega al índice: las
3 colecciones se construyen desde `extracted/`, no desde los transcripts.

Lo que se fija acá son las tres cosas que ya fallaron o que fallarían sin
ruido:

1. **El chunking no puede perder texto.** Un corte mal hecho no tira un error:
   extrae menos y el informe igual dice `OK`. La prueba reconstruye el texto
   original desde los chunks, así que un carácter perdido rompe el test.
2. **La respuesta del modelo llega mal formada seguido** — con fences de
   markdown, con prosa alrededor, con una coma colgante o cortada al medio por
   `max_tokens`. Cada caso que `_extract_json()` no recupere es un chunk que se
   descarta entero y que nadie vuelve a mirar, porque el archivo de salida
   queda escrito igual.
3. **La etapa es resumible por existencia del archivo, no por contenido**
   (gotcha #13). Es el mecanismo exacto que dejó 3.184 extracciones vacías
   pareciendo hechas cuando se agotó el crédito de la API: el archivo estaba,
   así que la corrida siguiente lo salteó. La prueba deja ese comportamiento
   escrito, con su única salida (`force=True`), para que se lea como una
   decisión y no como una sorpresa.
"""

from __future__ import annotations

import json

import pytest

from coach_brain.kb import extract as E
from coach_brain.settings import settings

# --- 1) chunking ------------------------------------------------------------


def test_texto_corto_es_un_solo_chunk():
    assert E._chunk_text("hola", size=100, overlap=20) == ["hola"]
    borde = "x" * 100
    assert E._chunk_text(borde, size=100, overlap=20) == [borde]


def test_los_chunks_reconstruyen_el_texto_entero():
    """La propiedad que importa: el chunking reparte, no descarta.

    Cada chunk arranca `overlap` caracteres antes de donde terminó el anterior,
    así que sacándole ese prefijo al segundo y siguientes tiene que volver a
    salir el texto original, carácter por carácter.
    """
    size, overlap = 100, 20
    texto = ". ".join(f"oracion numero {i} con algo de relleno" for i in range(40))
    chunks = E._chunk_text(texto, size=size, overlap=overlap)

    assert len(chunks) > 1, "con 40 oraciones tiene que partir"
    rearmado = chunks[0] + "".join(c[overlap:] for c in chunks[1:])
    assert rearmado == texto


def test_ningun_chunk_pasa_el_tamano_pedido():
    """El tamaño es lo que mantiene la salida del modelo chica: un chunk más
    grande que el pedido es un OOM de Ollama más adelante, no un detalle."""
    texto = "palabra " * 500
    for c in E._chunk_text(texto, size=100, overlap=20):
        assert len(c) <= 100


def test_texto_sin_separadores_no_cuelga():
    """Sin puntos ni espacios no hay dónde cortar lindo. Tiene que cortar igual
    y terminar: el `rfind` devuelve -1 y el corte queda en el tamaño exacto."""
    chunks = E._chunk_text("x" * 450, size=100, overlap=20)
    assert len(chunks) > 1
    assert "".join([chunks[0]] + [c[20:] for c in chunks[1:]]) == "x" * 450


def test_el_overlap_es_contexto_compartido():
    size, overlap = 100, 20
    texto = ". ".join(f"frase {i} larga para forzar el corte" for i in range(20))
    chunks = E._chunk_text(texto, size=size, overlap=overlap)
    for a, b in zip(chunks, chunks[1:], strict=False):  # el ultimo no tiene siguiente
        assert a[-overlap:] == b[:overlap]


# --- 2) parseo de la respuesta del modelo -----------------------------------


@pytest.mark.parametrize("crudo", [
    '{"principios": [], "situaciones": [], "frases": []}',
    '```json\n{"principios": []}\n```',
    '```\n{"principios": []}\n```',
    'Claro, aca va lo que extraje:\n{"principios": []}\nEspero que sirva.',
    '{"principios": [],}',
])
def test_recupera_json_de_una_respuesta_sucia(crudo):
    assert isinstance(E._extract_json(crudo), dict)


def test_recupera_una_respuesta_cortada_por_max_tokens():
    """El caso más común y el más caro: la salida se corta a mitad de un
    objeto. Si no se repara, el chunk se descarta entero."""
    cortado = '{"principios": [{"id": "p1", "texto": "no dependas de la'
    d = E._extract_json(cortado)
    assert d is not None, "una respuesta truncada tiene que recuperarse"
    assert d["principios"][0]["id"] == "p1"


def test_una_respuesta_sin_json_devuelve_none():
    """None es la señal de que el chunk no se pudo leer. Devolver `{}` acá
    haría que un fallo del modelo se vea igual que un chunk sin nada que
    extraer."""
    assert E._extract_json("No encontre principios en este fragmento.") is None
    assert E._extract_json("") is None


def test_repair_cierra_lo_que_quedo_abierto():
    assert json.loads(E._repair_json('{"a": [1, 2')) == {"a": [1, 2]}
    assert json.loads(E._repair_json('{"a": "sin cerrar')) == {"a": "sin cerrar"}
    assert json.loads(E._repair_json('{"a": 1,')) == {"a": 1}


def test_repair_no_toca_las_llaves_dentro_de_un_string():
    """Si contara los brackets de adentro de un string, cerraría de más y el
    JSON válido terminaría roto."""
    s = '{"texto": "usa {} y [] en el ejemplo"}'
    assert json.loads(E._repair_json(s)) == json.loads(s)


def test_una_comilla_escapada_no_cierra_el_string():
    """El caso más común de todos: una frase citando a alguien. Si el escaneo
    tomara la comilla escapada como fin de string, contaría los brackets de
    adentro del texto y cerraría de más."""
    cortado = r'{"frases": [{"texto": "me dijo \"hola\" y se fue'
    assert E._extract_json(cortado) == {
        "frases": [{"texto": 'me dijo "hola" y se fue'}]
    }


def test_una_barra_escapada_no_escapa_la_comilla_que_sigue():
    r"""`\` es una barra literal, no el escape de lo que viene después. Sin
    contar eso, un `"...\\"` dejaría el escaneo adentro del string para siempre
    y el reparado cerraría todo mal."""
    s = r'{"a": "una barra \\", "b": 1}'
    assert json.loads(E._repair_json(s)) == json.loads(s)
    assert E._extract_json(s) == {"a": "una barra " + chr(92), "b": 1}


def test_un_json_que_ni_reparado_parsea_devuelve_none():
    """No alcanza con que haya llaves: lo que falta acá es el valor, y ninguno
    de los tres intentos lo puede inventar. `extract_chunk` lo convierte en las
    tres listas vacías con un aviso que nombra el chunk."""
    assert E._extract_json('{ "principios": }') is None


# --- 3) un chunk que falla no se lleva el transcript ------------------------


def _transcript(d, stem, texto, source_id=None):
    p = d / (stem + ".json")
    p.write_text(json.dumps({
        "source_id": source_id or stem,
        "source_type": "audio",
        "title": stem,
        "full_text": texto,
    }, ensure_ascii=False), encoding="utf-8")
    return p


def test_transcript_vacio_no_llama_al_modelo(tmp_path, monkeypatch):
    """Los 62 transcripts vacíos del corpus (gotcha #21) no tienen que gastar
    una llamada ni escribir items inventados."""
    llamadas = []
    monkeypatch.setattr(E, "extract_chunk",
                        lambda *a, **k: llamadas.append(a) or {})
    r = E.process_transcript(None, "plantilla", _transcript(tmp_path, "vacio", "   "))
    assert llamadas == []
    assert (r["principios"], r["situaciones"], r["frases"]) == ([], [], [])
    assert r["source_id"] == "vacio"


def test_un_chunk_que_explota_no_pierde_los_otros(tmp_path, monkeypatch):
    """Ollama se muere por OOM de a ratos. Si una excepción cortara el
    transcript, un chunk malo tiraría a la basura las horas de los demás."""
    def fake(client, plantilla, source_id, idx, chunk):
        if idx == 1:
            raise RuntimeError("ollama se cayo")
        return {"principios": [{"id": f"p{idx}"}], "situaciones": [], "frases": []}

    monkeypatch.setattr(E, "extract_chunk", fake)
    monkeypatch.setattr(E, "_chunk_text", lambda t, **k: ["a", "b", "c"])
    r = E.process_transcript(None, "plantilla", _transcript(tmp_path, "algo", "texto"))
    assert [p["id"] for p in r["principios"]] == ["p0", "p2"]


# --- 4) resumible: por existencia del archivo, no por contenido -------------


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Apunta la etapa a un data/ de juguete, con backend local (sin API key)."""
    (tmp_path / "processed" / "transcripts").mkdir(parents=True)
    (tmp_path / "processed" / "extracted").mkdir(parents=True)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "extractor_backend", "ollama")
    monkeypatch.setattr(E, "_load_prompt", lambda: "plantilla {source_id} {chunk}")
    return tmp_path / "processed" / "transcripts", tmp_path / "processed" / "extracted"


def test_una_extraccion_vacia_ya_escrita_se_saltea(entorno, monkeypatch):
    """Gotcha #13, el que dejó 3.184 archivos vacíos pareciendo hechos.

    Se saltea por EXISTENCIA del archivo, no por su contenido. Está probado a
    propósito: es lo que hace la etapa resumible después de días de corrida, y
    lo que obliga a que la re-destilación se haga moviendo los archivos malos
    afuera (o con `force`), no volviendo a correr lo mismo.
    """
    tdir, edir = entorno
    _transcript(tdir, "uno", "hay texto de sobra para extraer algo")
    (edir / "uno.extracted.json").write_text(
        json.dumps({"principios": [], "situaciones": [], "frases": []}),
        encoding="utf-8")

    llamadas = []
    monkeypatch.setattr(E, "process_transcript",
                        lambda *a: llamadas.append(a[-1].name) or {})
    E.run()
    assert llamadas == [], "el archivo vacio existe, asi que se saltea"


def test_force_reescribe_lo_que_ya_estaba(entorno, monkeypatch):
    tdir, edir = entorno
    _transcript(tdir, "uno", "hay texto")
    (edir / "uno.extracted.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(E, "process_transcript", lambda *a: {
        "source_id": "uno", "principios": [{"id": "p1"}],
        "situaciones": [], "frases": [],
    })
    E.run(force=True)
    d = json.loads((edir / "uno.extracted.json").read_text(encoding="utf-8"))
    assert d["principios"] == [{"id": "p1"}]


def test_un_transcript_que_falla_no_corta_la_corrida(entorno, monkeypatch):
    """Una corrida son días. Un transcript roto tiene que quedar sin archivo de
    salida (para que la próxima pasada lo reintente) y no frenar al resto."""
    tdir, edir = entorno
    _transcript(tdir, "a_roto", "texto")
    _transcript(tdir, "b_sano", "texto")

    def fake(client, plantilla, tpath):
        if tpath.stem == "a_roto":
            raise RuntimeError("se cayo")
        return {"source_id": tpath.stem, "principios": [], "situaciones": [], "frases": []}

    monkeypatch.setattr(E, "process_transcript", fake)
    E.run()
    assert not (edir / "a_roto.extracted.json").exists()
    assert (edir / "b_sano.extracted.json").exists()


def test_el_nombre_de_salida_es_el_que_lee_el_indexador(entorno, monkeypatch):
    """`<stem>.extracted.json`, no `<stem>.json`. El indexador globea ese
    patrón exacto, y comparar stems a lo bruto hace que cada archivo parezca
    faltante (se reportaron 8.838 faltantes así una vez)."""
    tdir, edir = entorno
    _transcript(tdir, "wa_audio_0001", "texto")
    monkeypatch.setattr(E, "process_transcript", lambda *a: {
        "source_id": "wa_audio_0001", "principios": [], "situaciones": [], "frases": [],
    })
    E.run()
    assert [p.name for p in edir.iterdir()] == ["wa_audio_0001.extracted.json"]


def test_con_el_backend_de_anthropic_el_cliente_se_arma_una_vez_y_se_reparte(
    entorno, monkeypatch, capsys
):
    """Un cliente por corrida, no uno por transcript: son 8.838 archivos.

    Y el cartel de Ollama no tiene que salir, porque dice "local, gratis" y en
    este camino cada chunk se paga por token.
    """
    tdir, _edir = entorno
    _transcript(tdir, "uno", "texto")
    _transcript(tdir, "dos", "texto")
    monkeypatch.setattr(settings, "extractor_backend", "anthropic")
    monkeypatch.setattr(E, "_client", lambda: "CLIENTE")

    vistos = []

    def fake(client, plantilla, tpath):
        vistos.append(client)
        return {"source_id": tpath.stem, "principios": [], "situaciones": [], "frases": []}

    monkeypatch.setattr(E, "process_transcript", fake)
    E.run()

    assert vistos == ["CLIENTE", "CLIENTE"]
    assert "local, gratis" not in capsys.readouterr().out


def test_sin_transcripts_avisa_y_no_abre_el_cliente(entorno, monkeypatch, capsys):
    """El orden importa: la lista de transcripts se mira ANTES de construir el
    cliente. Con la carpeta vacía y el backend en anthropic, pedir la API key
    primero haría fallar una corrida que no tenía nada que hacer."""
    monkeypatch.setattr(settings, "extractor_backend", "anthropic")
    monkeypatch.setattr(E, "_client", lambda: (_ for _ in ()).throw(
        AssertionError("no tiene que construir el cliente")))

    E.run()

    assert "No hay transcripts" in capsys.readouterr().out
