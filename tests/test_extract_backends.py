"""Tests de los dos backends de extracción y de la auto-recuperación de Ollama.

`tests/test_extract.py` cubre el chunking, el parseo del JSON sucio y la
contabilidad de `run`. Lo que no estaba probado es la capa que habla con los
modelos, que es la que decide si una corrida de 50 horas sobre 8838 transcripts
termina o se frena a la mitad:

- **El reintento de Ollama.** Con `num_ctx` en 8192 el server se moría por OOM a
  mitad de la corrida. La respuesta no fue solo bajar el contexto a 6144: fue que
  `_extract_ollama` reintente y que `_ensure_ollama` relance el server si se
  cayó. Sin eso, un OOM a las seis horas deja el resto sin extraer y el motivo
  visible es un error de conexión.
- **El ruteo por backend.** Con `COACH_EXTRACTOR_BACKEND=ollama` no se puede
  tocar la API de Anthropic: ahí hay un cliente en `None` y, si lo hubiera,
  8838 transcripts contra la API cuestan plata. Y al revés, con el backend por
  omisión no se puede llamar a un Ollama que no está corriendo.
- **La falta de key.** `_client` tiene que decir qué falta y dónde ponerlo, no
  tirar un error de la librería desde adentro del primer chunk.

Varios tests fijan números con el motivo al lado: `num_ctx=6144` contra
`CHUNK_CHARS=5000`, el tope de 4 intentos, el techo de 20 segundos de espera.
Son el acuerdo que hace que la corrida entre en la VRAM y se recupere sola.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from coach_brain.kb import extract as E
from coach_brain.settings import settings

# --- _client: el error tiene que decir qué hacer ---------------------------


def test_sin_key_el_error_dice_que_falta_y_donde(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(RuntimeError) as exc:
        E._client()
    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert ".env" in msg


def test_con_key_devuelve_un_cliente(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-de-prueba")
    assert E._client().api_key == "sk-ant-de-prueba"


# --- _load_prompt ----------------------------------------------------------


def test_el_prompt_sale_de_prompts_extractor_md(tmp_path, monkeypatch):
    """El prompt es el que define las tres colecciones del índice. Que viva en un
    archivo y no en el código es lo que permite cambiarlo sin re-deployar."""
    monkeypatch.setattr(settings, "prompts_dir", str(tmp_path))
    (tmp_path / "extractor.md").write_text("sacá {chunk} de {source_id}", encoding="utf-8")
    assert E._load_prompt() == "sacá {chunk} de {source_id}"


def test_el_prompt_del_repo_tiene_los_dos_marcadores():
    """`extract_chunk` reemplaza `{source_id}` y `{chunk}`. Si el prompt real
    perdiera uno, cada chunk se mandaría sin el texto o sin el id y la corrida
    entera saldría vacía sin un solo error."""
    texto = E._load_prompt()
    assert "{chunk}" in texto
    assert "{source_id}" in texto


# --- _extract_anthropic ----------------------------------------------------


class _Bloque:
    def __init__(self, text: str) -> None:
        self.text = text


class _BloqueSinTexto:
    """Un bloque que no es de texto, como los de `tool_use`."""


class _Mensajes:
    def __init__(self, bloques) -> None:
        self.bloques = bloques
        self.visto: dict = {}

    def create(self, **kw):
        self.visto = kw
        return type("R", (), {"content": self.bloques})()


class _ClienteFalso:
    def __init__(self, bloques) -> None:
        self.messages = _Mensajes(bloques)


def test_la_respuesta_de_anthropic_se_une_en_un_solo_texto():
    c = _ClienteFalso([_Bloque("{"), _Bloque('"principios": []}')])
    assert E._extract_anthropic(c, "dame json") == '{"principios": []}'


def test_un_bloque_sin_texto_no_rompe_la_union():
    c = _ClienteFalso([_Bloque("{}"), _BloqueSinTexto()])
    assert E._extract_anthropic(c, "dame json") == "{}"


def test_la_llamada_a_anthropic_usa_el_modelo_y_la_temperatura_configurados(monkeypatch):
    """`temperature=0.1` y el system prompt son lo que hace que la salida sea
    JSON parseable. Con el backend de Anthropic no hay schema que lo fuerce, así
    que es lo único que lo sostiene."""
    monkeypatch.setattr(settings, "model_extractor", "claude-sonnet-4-6")
    c = _ClienteFalso([_Bloque("{}")])
    E._extract_anthropic(c, "el prompt armado")
    visto = c.messages.visto
    assert visto["model"] == "claude-sonnet-4-6"
    assert visto["temperature"] == 0.1
    assert visto["max_tokens"] == 8192
    assert visto["system"] == E._SYSTEM
    assert visto["messages"] == [{"role": "user", "content": "el prompt armado"}]


# --- httpx falso -----------------------------------------------------------


class RespuestaFalsa:
    def __init__(self, data: dict | None = None, error: Exception | None = None) -> None:
        self._data = data or {}
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            raise self._error

    def json(self) -> dict:
        return self._data


def httpx_falso(monkeypatch, *, get=None, post=None):
    modulo = type(sys)("httpx")
    modulo.get = get or (lambda *a, **k: RespuestaFalsa())
    modulo.post = post or (lambda *a, **k: RespuestaFalsa())
    monkeypatch.setitem(sys.modules, "httpx", modulo)
    return modulo


# --- _ollama_alive ---------------------------------------------------------


def test_ollama_vivo_cuando_contesta_api_tags(monkeypatch):
    vistos = []
    httpx_falso(monkeypatch, get=lambda url, timeout=None: vistos.append((url, timeout)) or RespuestaFalsa())
    monkeypatch.setattr(settings, "ollama_host", "http://localhost:11434")

    assert E._ollama_alive() is True
    assert vistos == [("http://localhost:11434/api/tags", 5)]


def test_ollama_muerto_cuando_la_conexion_falla(monkeypatch):
    """Es el estado en el que queda tras un OOM: el proceso se fue y el puerto no
    contesta. Tiene que dar False, no propagar la excepción."""

    def explotar(url, timeout=None):
        raise OSError("conexion rechazada")

    httpx_falso(monkeypatch, get=explotar)
    assert E._ollama_alive() is False


def test_el_chequeo_tiene_timeout_corto(monkeypatch):
    """5 segundos: es un chequeo de vida que corre entre reintentos, no una
    espera. Sin timeout, un server colgado dejaría la corrida trabada."""
    vistos = []
    httpx_falso(monkeypatch, get=lambda url, timeout=None: vistos.append(timeout) or RespuestaFalsa())
    E._ollama_alive()
    assert vistos == [5]


# --- _ensure_ollama: el relanzado ------------------------------------------


@pytest.fixture
def sin_dormir(monkeypatch):
    monkeypatch.setattr(E.time, "sleep", lambda s: None)


def test_si_ollama_esta_vivo_no_se_relanza_nada(monkeypatch, sin_dormir):
    monkeypatch.setattr(E, "_ollama_alive", lambda: True)

    def no_llamar(*a, **k):  # pragma: no cover - no deberia correr
        raise AssertionError("se relanzo un server que ya estaba corriendo")

    monkeypatch.setattr(E.subprocess, "Popen", no_llamar)
    assert E._ensure_ollama() is True


def test_si_se_cayo_se_relanza_y_se_espera(monkeypatch, sin_dormir):
    """El caso real: el server murió por OOM a mitad de una corrida de 50 horas.
    Relanzarlo acá es lo que hace que el job siga solo en vez de quedar tirando
    errores de conexión hasta que alguien mire el log."""
    estados = [False, False, True]
    monkeypatch.setattr(E, "_ollama_alive", lambda: estados.pop(0))
    lanzados = []
    monkeypatch.setattr(E.subprocess, "Popen", lambda cmd, **kw: lanzados.append((cmd, kw)))

    assert E._ensure_ollama() is True
    (cmd, kw), = lanzados
    assert cmd[0].endswith("ollama.exe")
    assert cmd[1] == "serve"
    # Sin DETACHED_PROCESS el server se iria con el proceso de la extraccion.
    assert kw["creationflags"] == (0x00000008 if E.os.name == "nt" else 0)


def test_el_server_se_busca_en_localappdata(monkeypatch, sin_dormir):
    monkeypatch.setattr(E, "_ollama_alive", lambda: False)
    monkeypatch.setenv("LOCALAPPDATA", str(Path("D:/AppData/Local")))
    lanzados = []
    monkeypatch.setattr(E.subprocess, "Popen", lambda cmd, **kw: lanzados.append(cmd))

    E._ensure_ollama(wait=2)

    assert Path(lanzados[0][0]) == Path("D:/AppData/Local/Programs/Ollama/ollama.exe")


def test_si_no_se_puede_relanzar_avisa_y_sigue_esperando(monkeypatch, sin_dormir, capsys):
    """Que no esté instalado donde se lo busca no es motivo para tirar: el server
    puede estar levantándose por otro lado, y el que llama tiene que recibir un
    booleano, no una excepción desde el medio de un reintento."""
    monkeypatch.setattr(E, "_ollama_alive", lambda: False)

    def explotar(cmd, **kw):
        raise FileNotFoundError("no esta")

    monkeypatch.setattr(E.subprocess, "Popen", explotar)

    assert E._ensure_ollama(wait=2) is False
    assert "No pude relanzar Ollama" in capsys.readouterr().out


def test_si_no_vuelve_en_el_plazo_devuelve_false(monkeypatch, sin_dormir):
    intentos = []
    monkeypatch.setattr(E, "_ollama_alive", lambda: intentos.append(1) and False)
    monkeypatch.setattr(E.subprocess, "Popen", lambda cmd, **kw: None)

    assert E._ensure_ollama(wait=10) is False
    # Un chequeo inicial mas wait//2 esperas de 2 segundos.
    assert len(intentos) == 1 + 5


def test_siempre_hay_al_menos_un_reintento(monkeypatch, sin_dormir):
    """`wait=0` no puede significar "no esperes nada": el server recién lanzado
    siempre tarda un momento en atender."""
    intentos = []
    monkeypatch.setattr(E, "_ollama_alive", lambda: intentos.append(1) and False)
    monkeypatch.setattr(E.subprocess, "Popen", lambda cmd, **kw: None)

    E._ensure_ollama(wait=0)
    assert len(intentos) == 2


def test_cuando_vuelve_lo_dice(monkeypatch, sin_dormir, capsys):
    """El aviso es lo que explica, al leer el log, por qué hubo un hueco de un
    minuto en una corrida que por lo demás no falló."""
    estados = [False, True]
    monkeypatch.setattr(E, "_ollama_alive", lambda: estados.pop(0))
    monkeypatch.setattr(E.subprocess, "Popen", lambda cmd, **kw: None)

    E._ensure_ollama()
    assert "Ollama recuperado" in capsys.readouterr().out


# --- _extract_ollama: el payload y los reintentos --------------------------


def test_el_payload_fuerza_json_y_entra_en_la_vram(monkeypatch):
    """`format` con el schema es lo que garantiza que la salida parsee: sin eso,
    un modelo local de 14B devuelve markdown la mitad de las veces.

    Y `num_ctx=6144` contra `CHUNK_CHARS=5000` es el par que hace que el modelo
    entre entero en la VRAM: con 8192 el server se moría por OOM a mitad de la
    corrida. Los dos números son un acuerdo, no dos constantes sueltas.
    """
    vistos = []
    httpx_falso(
        monkeypatch,
        post=lambda url, json=None, timeout=None: vistos.append((url, json, timeout))
        or RespuestaFalsa({"message": {"content": "{}"}}),
    )
    monkeypatch.setattr(settings, "ollama_host", "http://localhost:11434")
    monkeypatch.setattr(settings, "ollama_model", "qwen2.5:14b-instruct")

    assert E._extract_ollama("el prompt") == "{}"

    url, payload, timeout = vistos[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "qwen2.5:14b-instruct"
    assert payload["format"] == E._OLLAMA_SCHEMA
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.1, "num_ctx": 6144}
    assert payload["messages"][0] == {"role": "system", "content": E._SYSTEM}
    assert payload["messages"][1] == {"role": "user", "content": "el prompt"}
    # Un libro de 400 paginas tarda minutos por chunk: el timeout es de corrida
    # larga, no de request web.
    assert timeout == 600


def test_el_schema_exige_las_tres_colecciones():
    """Las tres colecciones del índice salen de estas tres claves. Que el schema
    las exija es lo que hace que `_extract_json` nunca tenga que adivinar."""
    assert E._OLLAMA_SCHEMA["required"] == ["principios", "situaciones", "frases"]
    for k in ("principios", "situaciones", "frases"):
        assert E._OLLAMA_SCHEMA["properties"][k]["type"] == "array"


def test_un_500_se_reintenta_y_la_segunda_vez_sale(monkeypatch, sin_dormir):
    """Un 500 de Ollama es casi siempre transitorio (el modelo recargándose).
    Reintentar es la diferencia entre un hueco de segundos y un transcript
    perdido."""
    llamadas = []

    def post(url, json=None, timeout=None):
        llamadas.append(1)
        if len(llamadas) == 1:
            return RespuestaFalsa(error=RuntimeError("500 server error"))
        return RespuestaFalsa({"message": {"content": "{}"}})

    httpx_falso(monkeypatch, post=post)
    monkeypatch.setattr(E, "_ensure_ollama", lambda: True)

    assert E._extract_ollama("x") == "{}"
    assert len(llamadas) == 2


def test_cada_fallo_intenta_levantar_el_server(monkeypatch, sin_dormir):
    """Es la auto-recuperación: si el fallo fue porque el server murió por OOM,
    el reintento solo sirve si antes alguien lo levanta."""
    httpx_falso(monkeypatch, post=lambda *a, **k: (_ for _ in ()).throw(OSError("muerto")))
    recuperos = []
    monkeypatch.setattr(E, "_ensure_ollama", lambda: recuperos.append(1))

    with pytest.raises(OSError):
        E._extract_ollama("x")
    assert len(recuperos) == 4


def test_despues_de_cuatro_intentos_se_propaga_el_ultimo_error(monkeypatch, sin_dormir):
    """Se rinde, pero con el error de verdad: `process_transcript` lo escribe en
    el log con el nombre del transcript y la corrida sigue con el siguiente."""
    llamadas = []

    def post(url, json=None, timeout=None):
        llamadas.append(1)
        raise OSError(f"fallo {len(llamadas)}")

    httpx_falso(monkeypatch, post=post)
    monkeypatch.setattr(E, "_ensure_ollama", lambda: True)

    with pytest.raises(OSError, match="fallo 4"):
        E._extract_ollama("x")
    assert len(llamadas) == 4


def test_la_espera_entre_intentos_crece_pero_tiene_techo(monkeypatch):
    """Backoff para no martillar un server que se está levantando, con techo para
    no sumar minutos muertos a una corrida de 8838 transcripts."""
    esperas = []
    monkeypatch.setattr(E.time, "sleep", lambda s: esperas.append(s))
    httpx_falso(monkeypatch, post=lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setattr(E, "_ensure_ollama", lambda: True)

    with pytest.raises(OSError):
        E._extract_ollama("x")
    assert esperas == [3, 6, 12, 20]


# --- extract_chunk: el ruteo y el armado del prompt ------------------------


def test_con_backend_ollama_no_se_toca_anthropic(monkeypatch):
    """Con `COACH_EXTRACTOR_BACKEND=ollama` el cliente es `None`: cualquier
    camino que lo use explota. Y si hubiera cliente, 8838 transcripts contra la
    API cuestan plata."""
    monkeypatch.setattr(settings, "extractor_backend", "ollama")
    monkeypatch.setattr(E, "_extract_ollama", lambda prompt: '{"principios": [{"a": 1}]}')

    def no_llamar(*a, **k):  # pragma: no cover - no deberia correr
        raise AssertionError("se llamo a la API de Anthropic con el backend local")

    monkeypatch.setattr(E, "_extract_anthropic", no_llamar)

    data = E.extract_chunk(None, "{chunk}", "wa_audio_1", 0, "texto")
    assert data == {"principios": [{"a": 1}]}


def test_con_el_backend_por_omision_no_se_toca_ollama(monkeypatch):
    monkeypatch.setattr(settings, "extractor_backend", "anthropic")

    def no_llamar(prompt):  # pragma: no cover - no deberia correr
        raise AssertionError("se llamo a Ollama con el backend de Anthropic")

    monkeypatch.setattr(E, "_extract_ollama", no_llamar)
    monkeypatch.setattr(
        E, "_extract_anthropic", lambda client, prompt: '{"principios": [{"a": 1}]}'
    )

    assert E.extract_chunk(None, "{chunk}", "x", 0, "texto") == {"principios": [{"a": 1}]}


def test_el_prompt_lleva_el_chunk_y_el_id_con_el_numero(monkeypatch):
    """El id del chunk viaja al prompt porque los items extraídos lo citan: es lo
    que después permite volver del vector al pedazo de texto que lo produjo."""
    monkeypatch.setattr(settings, "extractor_backend", "ollama")
    vistos = []
    monkeypatch.setattr(
        E, "_extract_ollama", lambda prompt: vistos.append(prompt) or "{}"
    )

    E.extract_chunk(None, "id={source_id} texto={chunk}", "video_clase", 7, "hola")
    assert vistos == ["id=video_clase_c7 texto=hola"]


def test_una_respuesta_que_no_parsea_devuelve_las_tres_claves_vacias(monkeypatch, capsys):
    """Un chunk ilegible no puede devolver `None`: `process_transcript` hace
    `.get(...)` sobre las tres claves y el transcript entero quedaría sin los
    chunks que sí salieron bien."""
    monkeypatch.setattr(settings, "extractor_backend", "ollama")
    monkeypatch.setattr(E, "_extract_ollama", lambda prompt: "no soy json")

    data = E.extract_chunk(None, "{chunk}", "wa_audio_1", 3, "texto")

    assert data == {"principios": [], "situaciones": [], "frases": []}
    salida = capsys.readouterr().out
    assert "wa_audio_1" in salida
    assert "3" in salida


def test_el_tamano_de_chunk_es_el_que_entra_en_el_contexto():
    """5000 caracteres son ~1500 tokens; con `num_ctx=6144` entran el system, el
    prompt del extractor, el chunk y la salida. Subir uno sin el otro es volver
    al OOM que tumbaba el server."""
    assert E.CHUNK_CHARS == 5_000
    assert E.CHUNK_OVERLAP_CHARS == 400
