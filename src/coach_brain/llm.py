"""Capa unica de LLM: Anthropic, Ollama local, o cualquier proveedor OpenAI-compatible.

Groq, Gemini, OpenRouter y Cerebras exponen todos el mismo endpoint
`POST /chat/completions` con `Authorization: Bearer <key>`, asi que un solo
code path los cubre. Cambiar de proveedor = cambiar base_url + key + modelo.

Los nombres de modelo se pueden pisar por env var porque los proveedores los
renombran y deprecan seguido; asi no hace falta tocar codigo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from coach_brain.settings import settings

TIMEOUT = 180.0  # los free tier suelen ser lentos en hora pico


def _cfg(name: str) -> str:
    """Lee config de settings (que ya parsea .env) y si no, del entorno."""
    val = getattr(settings, name.lower(), None)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return (os.environ.get(name) or "").strip()


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    env_key: str
    default_model: str
    free: bool
    note: str = ""
    # Params extra del payload (ej: reasoning_effort en modelos con thinking).
    extra: dict = field(default_factory=dict)
    # Piso de max_tokens. Los modelos con thinking gastan presupuesto razonando
    # ANTES de escribir: con un tope bajo devuelven contenido vacio.
    token_floor: int = 0

    @property
    def api_key(self) -> str:
        return _cfg(self.env_key)

    @property
    def model(self) -> str:
        """Modelo configurable: COACH_MODEL_<PROVIDER> pisa el default."""
        return _cfg(f"COACH_MODEL_{self.key.upper()}") or self.default_model

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


# Orden = prioridad sugerida en la UI.
PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        key="gemini", label="Gemini 3.6 Flash", free=True,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        env_key="GEMINI_API_KEY", default_model="gemini-3.6-flash",
        note="Free tier amplio. aistudio.google.com/apikey",
        # "none" devuelve 400: "low" es el minimo que acepta Gemini 3.
        extra={"reasoning_effort": "low"},
        token_floor=2048,
    ),
    "groq": Provider(
        key="groq", label="Groq (Llama 3.3 70B)", free=True,
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY", default_model="llama-3.3-70b-versatile",
        note="Muy rapido. console.groq.com/keys",
    ),
    "cerebras": Provider(
        key="cerebras", label="Cerebras (Llama 3.3 70B)", free=True,
        base_url="https://api.cerebras.ai/v1",
        env_key="CEREBRAS_API_KEY", default_model="llama-3.3-70b",
        note="cloud.cerebras.ai",
    ),
    "openrouter": Provider(
        key="openrouter", label="OpenRouter", free=True,
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY", default_model="deepseek/deepseek-chat-v3.1:free",
        note="Agrega muchos modelos; los :free no cobran. openrouter.ai/keys",
    ),
}


def available_providers() -> list[Provider]:
    """Solo los que tienen API key cargada."""
    return [p for p in PROVIDERS.values() if p.configured]


def _openai_compatible(provider: Provider, system: str, user: str,
                       max_tokens: int, temperature: float) -> str:
    r = httpx.post(
        f"{provider.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {provider.api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": provider.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max(max_tokens, provider.token_floor),
            "temperature": temperature,
            **provider.extra,
        },
        timeout=TIMEOUT,
    )
    if r.status_code >= 400:
        # El body trae el motivo real (cuota, modelo inexistente, key invalida).
        raise RuntimeError(f"{provider.label} HTTP {r.status_code}: {r.text[:400]}")
    data = r.json()
    try:
        choice = data["choices"][0]
        content = (choice["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"{provider.label}: respuesta inesperada: {str(data)[:300]}") from e
    if not content:
        # Tipico de modelos con thinking: se quedaron sin presupuesto razonando.
        raise RuntimeError(
            f"{provider.label}: respuesta vacia "
            f"(finish_reason={choice.get('finish_reason')}, usage={data.get('usage')}). "
            "Subi max_tokens o bajá el reasoning_effort.")
    return content


def _ollama(system: str, user: str, temperature: float) -> str:
    r = httpx.post(
        f"{settings.ollama_host}/api/chat",
        json={"model": settings.ollama_model,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}],
              "stream": False,
              "options": {"temperature": temperature, "num_ctx": 10240}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _anthropic(system: str, user: str, model: str,
               max_tokens: int, temperature: float) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def complete(*, system: str, user: str, backend: str, model: str = "",
             max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """Punto unico de entrada.

    backend: "anthropic" | "ollama" | una key de PROVIDERS (gemini/groq/...).
    """
    if backend == "ollama":
        return _ollama(system, user, temperature)
    if backend == "anthropic":
        return _anthropic(system, user, model or settings.model_main, max_tokens, temperature)
    provider = PROVIDERS.get(backend)
    if provider is None:
        raise ValueError(f"backend desconocido: {backend}")
    if not provider.configured:
        raise RuntimeError(f"Falta {provider.env_key} en el entorno para usar {provider.label}.")
    return _openai_compatible(provider, system, user, max_tokens, temperature)


def complete_fast(*, system: str, user: str, max_tokens: int = 400,
                  temperature: float = 0.0) -> str:
    """Para tareas chicas (detectar nombre, resumir perfil).

    Elige solo: Ollama local -> primer proveedor gratis -> Anthropic Haiku.
    Devuelve "" si no hay ningun backend disponible, para que el llamador siga.
    """
    if settings.ollama_enabled:
        try:
            return _ollama(system, user, temperature)
        except Exception:  # noqa: BLE001
            pass
    for provider in available_providers():
        try:
            return _openai_compatible(provider, system, user, max_tokens, temperature)
        except Exception:  # noqa: BLE001
            continue
    if settings.anthropic_api_key:
        try:
            return _anthropic(system, user, settings.model_fast, max_tokens, temperature)
        except Exception:  # noqa: BLE001
            pass
    return ""
