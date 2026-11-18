"""Tests de la capa de LLM: ruteo, overrides por env y la cadena de fallback.

Este módulo es el que decide con qué proveedor se contesta, y casi todo lo que
hace mal se ve como "el modelo no contestó", no como un error del código. Lo que
se fija acá:

1. **El nombre del modelo se puede pisar por env var.** Los proveedores
   renombran y deprecan modelos seguido; si el override no funcionara, un modelo
   dado de baja obligaría a tocar código para volver a responder.
2. **Un `max_tokens` chico en un modelo con thinking devuelve contenido vacío**,
   no un error. De ahí el `token_floor` y el `raise` con `finish_reason`: sin
   eso, el síntoma es un string vacío sin explicación.
3. **`complete()` distingue "backend que no existe" de "backend sin key".** El
   primero es un bug de llamada (`ValueError`), el segundo es configuración
   faltante (`RuntimeError` nombrando la env var que falta).
4. **`complete_fast()` nunca levanta.** Es la que usan las tareas chicas
   (detectar un nombre, resumir un perfil) y devuelve `""` si no hay ningún
   backend; si levantara, una tarea opcional cortaría la respuesta principal.
5. **Los dos backends propios mandan lo que tienen que mandar.** `_ollama` con
   `stream: False` y `num_ctx: 10240`, `_anthropic` juntando solo los bloques
   con `.text`. Son los dos con los que contesta el MVP: uno gratis y local, el
   otro pago.

No se hace una sola llamada de red: `httpx.post` es un doble.
"""

from __future__ import annotations

import sys

import pytest

from coach_brain import llm as L
from coach_brain.settings import settings


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text or str(self._payload)

    def json(self):
        return self._payload


def _ok(contenido="listo"):
    return {"choices": [{"message": {"content": contenido},
                         "finish_reason": "stop"}],
            "usage": {"total_tokens": 10}}


@pytest.fixture
def post(monkeypatch):
    """Reemplaza httpx.post y guarda con qué se lo llamó."""
    llamadas: list[dict] = []
    respuesta = {"r": FakeResp(payload=_ok())}

    def fake(url, headers=None, json=None, timeout=None):
        llamadas.append({"url": url, "headers": headers or {}, "json": json or {}})
        r = respuesta["r"]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(L.httpx, "post", fake)
    return {"llamadas": llamadas, "set": lambda r: respuesta.__setitem__("r", r)}


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "k-test")
    return L.PROVIDERS["gemini"]


# --- 1) configuración: settings gana, el env es el respaldo ------------------


def test_cfg_prefiere_settings_sobre_el_entorno(monkeypatch):
    """settings ya parseó el .env; si el entorno ganara, un valor viejo
    exportado en la sesión pisaría el archivo del proyecto."""
    monkeypatch.setattr(settings, "groq_api_key", "de-settings")
    monkeypatch.setenv("GROQ_API_KEY", "del-entorno")
    assert L._cfg("GROQ_API_KEY") == "de-settings"


def test_cfg_cae_al_entorno_si_settings_no_lo_tiene(monkeypatch):
    monkeypatch.setenv("COACH_MODEL_GROQ", "llama-nuevo")
    assert L._cfg("COACH_MODEL_GROQ") == "llama-nuevo"


def test_cfg_ignora_un_valor_en_blanco(monkeypatch):
    """Una key seteada a espacios no es una key: tiene que contar como faltante
    para que el fallback siga de largo en lugar de pedir con Bearer vacío."""
    monkeypatch.setattr(settings, "groq_api_key", "   ")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert L._cfg("GROQ_API_KEY") == ""
    assert L.PROVIDERS["groq"].configured is False


def test_el_modelo_se_pisa_por_env_var(monkeypatch):
    p = L.PROVIDERS["groq"]
    assert p.model == p.default_model
    monkeypatch.setenv("COACH_MODEL_GROQ", "llama-4-turbo")
    assert p.model == "llama-4-turbo"


def test_available_providers_son_los_que_tienen_key(monkeypatch):
    for k in ("gemini_api_key", "groq_api_key", "cerebras_api_key", "openrouter_api_key"):
        monkeypatch.setattr(settings, k, "")
    for v in ("GEMINI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert L.available_providers() == []

    monkeypatch.setattr(settings, "cerebras_api_key", "k")
    assert [p.key for p in L.available_providers()] == ["cerebras"]


# --- 2) el request que sale ------------------------------------------------


def test_el_pedido_lleva_url_key_modelo_y_los_dos_mensajes(post, gemini):
    L._openai_compatible(gemini, "sos un coach", "que hago", 100, 0.3)
    c = post["llamadas"][0]

    assert c["url"] == gemini.base_url + "/chat/completions"
    assert c["headers"]["Authorization"] == "Bearer k-test"
    assert c["json"]["model"] == gemini.default_model
    assert c["json"]["messages"] == [{"role": "system", "content": "sos un coach"},
                                     {"role": "user", "content": "que hago"}]
    assert c["json"]["temperature"] == 0.3


def test_el_piso_de_tokens_levanta_un_max_tokens_chico(post, gemini):
    """Gemini 3 gasta presupuesto razonando ANTES de escribir: con 100 tokens
    devuelve contenido vacío. El piso es lo que evita ese fallo silencioso."""
    assert gemini.token_floor == 2048
    L._openai_compatible(gemini, "s", "u", 100, 0.0)
    assert post["llamadas"][0]["json"]["max_tokens"] == 2048


def test_un_max_tokens_grande_no_se_recorta(post, gemini):
    L._openai_compatible(gemini, "s", "u", 9000, 0.0)
    assert post["llamadas"][0]["json"]["max_tokens"] == 9000


def test_los_params_extra_del_proveedor_viajan(post, gemini):
    """`reasoning_effort: none` devuelve 400 en Gemini 3; "low" es el mínimo."""
    L._openai_compatible(gemini, "s", "u", 100, 0.0)
    assert post["llamadas"][0]["json"]["reasoning_effort"] == "low"


def test_un_proveedor_sin_extras_no_agrega_nada(post, monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "k")
    L._openai_compatible(L.PROVIDERS["groq"], "s", "u", 100, 0.0)
    assert "reasoning_effort" not in post["llamadas"][0]["json"]


# --- 3) los errores dicen qué pasó -----------------------------------------


def test_un_error_http_incluye_el_body(post, gemini):
    """El motivo real (cuota agotada, modelo inexistente, key inválida) viene en
    el body. Sin él, los cuatro proveedores fallan igual y no se distinguen."""
    post["set"](FakeResp(status=429, payload={}, text="quota exceeded for model X"))
    with pytest.raises(RuntimeError, match="429.*quota exceeded"):
        L._openai_compatible(gemini, "s", "u", 100, 0.0)


def test_una_respuesta_con_otra_forma_no_tira_keyerror(post, gemini):
    post["set"](FakeResp(payload={"error": "algo"}))
    with pytest.raises(RuntimeError, match="respuesta inesperada"):
        L._openai_compatible(gemini, "s", "u", 100, 0.0)


def test_una_respuesta_vacia_explica_por_que(post, gemini):
    """El caso del modelo con thinking que se quedó sin presupuesto. Devolver ""
    dejaría una extracción vacía pareciendo hecha."""
    post["set"](FakeResp(payload={"choices": [{"message": {"content": "  "},
                                               "finish_reason": "length"}],
                                  "usage": {"total_tokens": 2048}}))
    with pytest.raises(RuntimeError, match="respuesta vacia.*finish_reason=length"):
        L._openai_compatible(gemini, "s", "u", 100, 0.0)


def test_content_en_null_cuenta_como_vacia(post, gemini):
    post["set"](FakeResp(payload={"choices": [{"message": {"content": None}}]}))
    with pytest.raises(RuntimeError, match="respuesta vacia"):
        L._openai_compatible(gemini, "s", "u", 100, 0.0)


# --- 4) complete(): a dónde va cada backend --------------------------------


def test_un_backend_que_no_existe_es_un_error_de_llamada():
    with pytest.raises(ValueError, match="backend desconocido: inventado"):
        L.complete(system="s", user="u", backend="inventado")


def test_un_proveedor_sin_key_nombra_la_env_var_que_falta(monkeypatch):
    """La diferencia importa: esto no es un bug, es configuración, y el mensaje
    tiene que decir qué variable cargar."""
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        L.complete(system="s", user="u", backend="openrouter")


def test_ollama_no_pasa_por_los_proveedores(monkeypatch):
    monkeypatch.setattr(L, "_ollama", lambda s, u, t: "desde ollama")
    assert L.complete(system="s", user="u", backend="ollama") == "desde ollama"


def test_anthropic_sin_modelo_usa_el_principal(monkeypatch):
    vistos = {}

    def espia(system, user, model, max_tokens, temperature):
        vistos["model"] = model
        return "ok"

    monkeypatch.setattr(L, "_anthropic", espia)
    L.complete(system="s", user="u", backend="anthropic")
    assert vistos["model"] == settings.model_main


def test_anthropic_respeta_el_modelo_pedido(monkeypatch):
    vistos = {}
    monkeypatch.setattr(L, "_anthropic",
                        lambda system, user, model, max_tokens, temperature:
                        vistos.setdefault("model", model) or "ok")
    L.complete(system="s", user="u", backend="anthropic", model="claude-raro")
    assert vistos["model"] == "claude-raro"


def test_un_proveedor_con_key_llega_a_la_llamada(post, gemini):
    assert L.complete(system="s", user="u", backend="gemini") == "listo"


# --- 5) complete_fast(): la cadena, y que nunca levante --------------------


@pytest.fixture
def sin_backends(monkeypatch):
    """Apaga todo: Ollama, los cuatro proveedores y Anthropic."""
    monkeypatch.setattr(settings, "ollama_enabled", False)
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    for k in ("gemini_api_key", "groq_api_key", "cerebras_api_key", "openrouter_api_key"):
        monkeypatch.setattr(settings, k, "")
    for v in ("GEMINI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY",
              "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)


def test_sin_ningun_backend_devuelve_vacio_y_no_levanta(sin_backends):
    """La propiedad que importa: una tarea chica que no se puede hacer no corta
    la respuesta principal."""
    assert L.complete_fast(system="s", user="u") == ""


def test_con_ollama_prendido_arranca_por_ahi(sin_backends, monkeypatch):
    monkeypatch.setattr(settings, "ollama_enabled", True)
    monkeypatch.setattr(L, "_ollama", lambda s, u, t: "local")
    assert L.complete_fast(system="s", user="u") == "local"


def test_si_ollama_no_esta_sigue_con_un_proveedor(sin_backends, monkeypatch, post):
    monkeypatch.setattr(settings, "ollama_enabled", True)
    monkeypatch.setattr(L, "_ollama", lambda s, u, t: (_ for _ in ()).throw(
        RuntimeError("connection refused")))
    monkeypatch.setattr(settings, "groq_api_key", "k")
    assert L.complete_fast(system="s", user="u") == "listo"


def test_un_proveedor_caido_no_detiene_la_cadena(sin_backends, monkeypatch):
    """Los free tier se caen y agotan cuota seguido; probar el siguiente es el
    punto de tener cuatro."""
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "cerebras_api_key", "k")
    intentos: list[str] = []

    def falla_el_primero(provider, system, user, max_tokens, temperature):
        intentos.append(provider.key)
        if len(intentos) == 1:
            raise RuntimeError("429")
        return "del segundo"

    monkeypatch.setattr(L, "_openai_compatible", falla_el_primero)
    assert L.complete_fast(system="s", user="u") == "del segundo"
    assert len(intentos) == 2


def test_el_ultimo_recurso_es_haiku(sin_backends, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "k")
    vistos = {}

    def espia(system, user, model, max_tokens, temperature):
        vistos["model"] = model
        return "de haiku"

    monkeypatch.setattr(L, "_anthropic", espia)
    assert L.complete_fast(system="s", user="u") == "de haiku"
    assert vistos["model"] == settings.model_fast, "la tarea chica no paga Sonnet"


def test_si_hasta_anthropic_falla_devuelve_vacio(sin_backends, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "k")
    monkeypatch.setattr(L, "_anthropic", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("overloaded")))
    assert L.complete_fast(system="s", user="u") == ""


# --- 6) _ollama(): el backend local, el que contesta gratis -----------------


class RespOllama:
    """Lo mínimo que `_ollama` le pide: `raise_for_status` y `json`."""

    def __init__(self, payload=None, error=None):
        self._payload = payload if payload is not None else {
            "message": {"content": "desde ollama"}
        }
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._payload


@pytest.fixture
def post_ollama(monkeypatch):
    """`httpx.post` para la rama de Ollama, que manda la URL por posición."""
    llamadas: list[dict] = []
    respuesta = {"r": RespOllama()}

    def fake(url, json=None, timeout=None, **kw):
        llamadas.append({"url": url, "json": json or {}, "timeout": timeout})
        return respuesta["r"]

    monkeypatch.setattr(L.httpx, "post", fake)
    return {"llamadas": llamadas, "set": lambda r: respuesta.__setitem__("r", r)}


def test_ollama_pega_en_api_chat_del_host_configurado(post_ollama, monkeypatch):
    monkeypatch.setattr(settings, "ollama_host", "http://127.0.0.1:11434")
    L._ollama("s", "u", 0.7)
    assert post_ollama["llamadas"][0]["url"] == "http://127.0.0.1:11434/api/chat"


def test_ollama_manda_el_modelo_de_settings(post_ollama, monkeypatch):
    monkeypatch.setattr(settings, "ollama_model", "qwen2.5:14b-instruct")
    L._ollama("s", "u", 0.7)
    assert post_ollama["llamadas"][0]["json"]["model"] == "qwen2.5:14b-instruct"


def test_ollama_manda_los_dos_mensajes_en_orden(post_ollama):
    L._ollama("sos un coach", "que le digo", 0.7)
    msgs = post_ollama["llamadas"][0]["json"]["messages"]
    assert msgs == [
        {"role": "system", "content": "sos un coach"},
        {"role": "user", "content": "que le digo"},
    ]


def test_ollama_pide_la_respuesta_sin_stream(post_ollama):
    """Con `stream: True` el body son varios JSON pegados y `r.json()` revienta."""
    L._ollama("s", "u", 0.7)
    assert post_ollama["llamadas"][0]["json"]["stream"] is False


def test_ollama_pide_un_contexto_de_10240(post_ollama):
    """Más grande que el 6144 de la extracción (ver `kb/extract.py`): acá entra
    el prompt con los fragmentos recuperados, no un chunk de 5000 caracteres."""
    L._ollama("s", "u", 0.7)
    assert post_ollama["llamadas"][0]["json"]["options"]["num_ctx"] == 10240


def test_ollama_pasa_la_temperatura_que_le_dieron(post_ollama):
    L._ollama("s", "u", 0.25)
    assert post_ollama["llamadas"][0]["json"]["options"]["temperature"] == 0.25


def test_ollama_espera_hasta_300_segundos(post_ollama):
    """Un modelo de 14B en CPU tarda minutos. Con el timeout por omisión de
    httpx (5 s) toda respuesta larga seria un error."""
    L._ollama("s", "u", 0.7)
    assert post_ollama["llamadas"][0]["timeout"] == 300


def test_ollama_devuelve_el_contenido_del_mensaje(post_ollama):
    post_ollama["set"](RespOllama({"message": {"content": "andá de frente"}}))
    assert L._ollama("s", "u", 0.7) == "andá de frente"


def test_ollama_propaga_el_error_http(post_ollama):
    """`complete_fast` lo atrapa y sigue con el próximo backend; que levante es
    la única forma de que esa cadena funcione."""
    post_ollama["set"](RespOllama(error=OSError("conexion rechazada")))
    with pytest.raises(OSError, match="conexion rechazada"):
        L._ollama("s", "u", 0.7)


def test_ollama_con_otra_forma_de_respuesta_levanta(post_ollama):
    """A diferencia de la rama OpenAI-compatible, acá no hay traducción a
    `RuntimeError`: sale el `KeyError` crudo. `complete_fast` lo atrapa igual,
    pero `complete(backend="ollama")` lo deja pasar tal cual."""
    post_ollama["set"](RespOllama({"error": "model not found"}))
    with pytest.raises(KeyError):
        L._ollama("s", "u", 0.7)


# --- 7) _anthropic(): el backend pago --------------------------------------


class BloqueTexto:
    def __init__(self, text):
        self.text = text


class BloqueSinTexto:
    """Un bloque de thinking: viene en `content` y no tiene `.text`."""


class RespAnthropic:
    def __init__(self, content):
        self.content = content


def anthropic_falso(monkeypatch, respuesta, error=None):
    """Reemplaza el `from anthropic import Anthropic` que hace `_anthropic`."""
    registro: dict = {"construido": [], "create": []}

    class Mensajes:
        def create(self, **kw):
            registro["create"].append(kw)
            if error:
                raise error
            return respuesta

    class Cliente:
        def __init__(self, **kw):
            registro["construido"].append(kw)
            self.messages = Mensajes()

    modulo = type(sys)("anthropic")
    modulo.Anthropic = Cliente
    monkeypatch.setitem(sys.modules, "anthropic", modulo)
    return registro


def test_anthropic_construye_el_cliente_con_la_key_de_settings(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-de-prueba")
    reg = anthropic_falso(monkeypatch, RespAnthropic([BloqueTexto("ok")]))

    L._anthropic("s", "u", "claude-x", 100, 0.7)

    assert reg["construido"] == [{"api_key": "sk-ant-de-prueba"}]


def test_anthropic_manda_system_aparte_del_mensaje(monkeypatch):
    """La API de Anthropic no acepta un mensaje con `role: system`: el prompt de
    sistema es un parámetro propio."""
    reg = anthropic_falso(monkeypatch, RespAnthropic([BloqueTexto("ok")]))

    L._anthropic("sos un coach", "que le digo", "claude-x", 100, 0.7)

    kw = reg["create"][0]
    assert kw["system"] == "sos un coach"
    assert kw["messages"] == [{"role": "user", "content": "que le digo"}]


def test_anthropic_pasa_modelo_tokens_y_temperatura(monkeypatch):
    reg = anthropic_falso(monkeypatch, RespAnthropic([BloqueTexto("ok")]))

    L._anthropic("s", "u", "claude-sonnet-4-5", 1234, 0.3)

    kw = reg["create"][0]
    assert kw["model"] == "claude-sonnet-4-5"
    assert kw["max_tokens"] == 1234
    assert kw["temperature"] == 0.3


def test_anthropic_junta_los_bloques_de_texto(monkeypatch):
    anthropic_falso(
        monkeypatch,
        RespAnthropic([BloqueTexto("primero "), BloqueTexto("segundo")]),
    )
    assert L._anthropic("s", "u", "m", 100, 0.7) == "primero segundo"


def test_anthropic_saltea_los_bloques_sin_texto(monkeypatch):
    """Un bloque de thinking no tiene `.text`; sin el filtro, el `join` tira
    `AttributeError` y la respuesta se pierde entera."""
    anthropic_falso(
        monkeypatch,
        RespAnthropic([BloqueSinTexto(), BloqueTexto("la respuesta")]),
    )
    assert L._anthropic("s", "u", "m", 100, 0.7) == "la respuesta"


def test_anthropic_sin_bloques_devuelve_vacio(monkeypatch):
    anthropic_falso(monkeypatch, RespAnthropic([]))
    assert L._anthropic("s", "u", "m", 100, 0.7) == ""


def test_anthropic_propaga_el_error_de_la_api(monkeypatch):
    """Es lo que hace que `complete_fast` caiga al siguiente backend y que
    `complete(backend="anthropic")` muestre el motivo real (cuota, key)."""
    anthropic_falso(
        monkeypatch,
        RespAnthropic([]),
        error=RuntimeError("credit balance is too low"),
    )
    with pytest.raises(RuntimeError, match="credit balance"):
        L._anthropic("s", "u", "m", 100, 0.7)
