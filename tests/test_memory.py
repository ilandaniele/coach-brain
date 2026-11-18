"""Memoria por contacto: SQLite local, migracion de esquema y perfil corrido.

`DB_PATH` se resuelve al importar el modulo, asi que cada prueba lo reapunta a
`tmp_path` ademas de mover `settings.data_dir` (de donde sale `processed_path`).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from coach_brain import llm as L
from coach_brain.kb import memory as M
from coach_brain.settings import settings


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Una base vacia por prueba, fuera del repo."""
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(M, "DB_PATH", tmp_path / "processed" / "contacts.db")
    monkeypatch.setattr(settings, "ollama_enabled", False)
    return tmp_path / "processed" / "contacts.db"


@pytest.fixture
def sin_llm(monkeypatch):
    """Ningun backend: `_summarize_profile` tiene que devolver el perfil previo."""
    monkeypatch.setattr(settings, "ollama_enabled", False)
    monkeypatch.setattr(L, "complete_fast", lambda **kw: "")
    return monkeypatch


# --- etiquetas -------------------------------------------------------------

def test_la_etiqueta_junta_nombre_y_apellido():
    assert M.label_of({"name": "Mica", "apellido": "Wurch", "tag": ""}) == "Mica Wurch"


def test_el_tag_va_entre_parentesis():
    assert M.label_of({"name": "Mica", "apellido": "", "tag": "gimnasio"}) == "Mica (gimnasio)"


def test_solo_el_nombre_no_agrega_nada():
    assert M.label_of({"name": "Mica"}) == "Mica"


def test_nombre_apellido_y_tag_conviven():
    row = {"name": "Mica", "apellido": "Wurch", "tag": "gimnasio"}
    assert M.label_of(row) == "Mica Wurch (gimnasio)"


# --- crear, leer, listar ---------------------------------------------------

def test_crear_y_leer_devuelve_lo_guardado(db):
    cid = M.create_contact("Mica", "Wurch", "gimnasio")
    c = M.get_contact(cid)
    assert (c["name"], c["apellido"], c["tag"]) == ("Mica", "Wurch", "gimnasio")
    assert c["label"] == "Mica Wurch (gimnasio)"


def test_un_contacto_nuevo_arranca_sin_perfil_ni_historial(db):
    c = M.get_contact(M.create_contact("Mica"))
    assert c["profile"] == ""
    assert c["history"] == []


def test_los_espacios_del_formulario_se_recortan(db):
    c = M.get_contact(M.create_contact("  Mica  ", " Wurch ", " gimnasio "))
    assert (c["name"], c["apellido"], c["tag"]) == ("Mica", "Wurch", "gimnasio")


def test_un_id_que_no_existe_es_none_no_una_excepcion(db):
    assert M.get_contact(9999) is None


def test_dos_micas_son_dos_contactos_distintos(db):
    a = M.create_contact("Mica", tag="gimnasio")
    b = M.create_contact("Mica", tag="facultad")
    assert a != b
    assert {M.get_contact(a)["label"], M.get_contact(b)["label"]} == {
        "Mica (gimnasio)", "Mica (facultad)"}


def test_buscar_por_nombre_encuentra_la_colision(db):
    M.create_contact("Mica", tag="gimnasio")
    M.create_contact("Mica", tag="facultad")
    M.create_contact("Lucia")
    assert len(M.find_by_name("Mica")) == 2


def test_buscar_por_nombre_ignora_mayusculas_y_espacios(db):
    M.create_contact("Mica")
    assert len(M.find_by_name("  mICa ")) == 1


def test_buscar_un_nombre_que_no_esta_devuelve_lista_vacia(db):
    M.create_contact("Mica")
    assert M.find_by_name("Lucia") == []


def test_listar_trae_la_etiqueta_ya_armada(db):
    M.create_contact("Mica", "Wurch")
    assert M.list_contacts()[0]["label"] == "Mica Wurch"


def test_el_mas_recien_tocado_queda_primero(db, sin_llm):
    viejo = M.create_contact("Vieja")
    nuevo = M.create_contact("Nueva")
    M.update_after_analysis(viejo, "charla", "veredicto")
    assert [c["id"] for c in M.list_contacts()][0] == viejo
    assert nuevo in [c["id"] for c in M.list_contacts()]


def test_listar_una_base_vacia_no_rompe(db):
    assert M.list_contacts() == []


# --- historial y perfil ----------------------------------------------------

def test_el_analisis_agrega_una_entrada_al_historial(db, sin_llm):
    cid = M.create_contact("Mica")
    M.update_after_analysis(cid, "hola que hacés", "bien, directo")
    h = M.get_contact(cid)["history"]
    assert len(h) == 1
    assert h[0]["chat"] == "hola que hacés"
    assert h[0]["verdict"] == "bien, directo"
    assert h[0]["ts"]


def test_el_chat_y_el_veredicto_se_recortan_en_el_historial(db, sin_llm):
    cid = M.create_contact("Mica")
    M.update_after_analysis(cid, "x" * 2000, "y" * 1200)
    h = M.get_contact(cid)["history"][0]
    assert len(h["chat"]) == 1500
    assert len(h["verdict"]) == 800


def test_el_historial_se_queda_con_los_ultimos_veinte(db, sin_llm):
    cid = M.create_contact("Mica")
    M._save(cid, "", [{"ts": "t", "chat": f"c{i}", "verdict": "v"} for i in range(25)])
    M.update_after_analysis(cid, "la nueva", "v")
    h = M.get_contact(cid)["history"]
    assert len(h) == 20
    assert h[-1]["chat"] == "la nueva"
    assert h[0]["chat"] == "c6"  # las 6 primeras se cayeron


def test_los_acentos_sobreviven_al_json(db, sin_llm):
    cid = M.create_contact("Mica")
    M.update_after_analysis(cid, "¿cómo andás? ñandú", "v")
    assert M.get_contact(cid)["history"][0]["chat"] == "¿cómo andás? ñandú"
    crudo = sqlite3.connect(str(db)).execute(
        "SELECT history FROM contacts WHERE id=?", (cid,)).fetchone()[0]
    assert "cómo" in crudo  # ensure_ascii=False, no ó


def test_analizar_un_contacto_que_no_existe_devuelve_vacio(db, sin_llm):
    assert M.update_after_analysis(9999, "charla", "veredicto") == ""


def test_el_perfil_que_devuelve_el_modelo_queda_guardado(db, monkeypatch):
    monkeypatch.setattr(M, "_summarize_profile", lambda *a, **k: "**Quién es**: Mica")
    cid = M.create_contact("Mica")
    assert M.update_after_analysis(cid, "c", "v") == "**Quién es**: Mica"
    assert M.get_contact(cid)["profile"] == "**Quién es**: Mica"


def test_el_perfil_previo_entra_al_prompt(db, monkeypatch):
    vistos = {}
    monkeypatch.setattr(M, "_summarize_profile",
                        lambda user, prev, chat, verdict: vistos.update(
                            user=user, prev=prev, chat=chat) or "nuevo")
    cid = M.create_contact("Mica")
    M._save(cid, "perfil viejo", [])
    M.update_after_analysis(cid, "charla nueva", "v", user="Ilan")
    assert vistos == {"user": "Ilan", "prev": "perfil viejo", "chat": "charla nueva"}


# --- el perfil corrido: Ollama, fallback, y el piso ------------------------

@pytest.fixture
def ollama(monkeypatch):
    """Prende Ollama y deja interceptado `httpx.post`."""
    import httpx
    monkeypatch.setattr(settings, "ollama_enabled", True)
    monkeypatch.setattr(settings, "ollama_host", "http://local:11434")
    monkeypatch.setattr(settings, "ollama_model", "qwen-test")
    llamadas: list[dict] = []
    estado = {"r": None}

    class R:
        def __init__(self, contenido="perfil nuevo", status=200):
            self.contenido = contenido
            self.status = status

        def raise_for_status(self):
            if self.status >= 400:
                raise RuntimeError(f"HTTP {self.status}")

        def json(self):
            return {"message": {"content": self.contenido}}

    estado["r"] = R()

    def fake(url, json=None, timeout=None):
        llamadas.append({"url": url, "json": json or {}})
        r = estado["r"]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(httpx, "post", fake)
    return {"llamadas": llamadas, "set": lambda r: estado.__setitem__("r", r), "R": R}


def test_el_pedido_a_ollama_lleva_modelo_y_prompt(ollama):
    M._summarize_profile("Ilan", "lo previo", "la charla", "el veredicto")
    pedido = ollama["llamadas"][0]
    assert pedido["url"] == "http://local:11434/api/chat"
    assert pedido["json"]["model"] == "qwen-test"
    assert pedido["json"]["stream"] is False
    prompt = pedido["json"]["messages"][0]["content"]
    for texto in ("Ilan", "lo previo", "la charla", "el veredicto"):
        assert texto in prompt


def test_sin_perfil_previo_el_prompt_lo_dice(ollama):
    M._summarize_profile("Ilan", "", "c", "v")
    assert "es nueva" in ollama["llamadas"][0]["json"]["messages"][0]["content"]


def test_el_chat_y_el_veredicto_se_recortan_en_el_prompt(ollama):
    # Contar letras sobre todo el prompt mide tambien las de la plantilla
    # ("markdown" tiene una "w"), asi que se mide cada seccion por separado.
    M._summarize_profile("Ilan", "", "x" * 3000, "w" * 2000)
    prompt = ollama["llamadas"][0]["json"]["messages"][0]["content"]
    chat = prompt.split("CHAT:\n", 1)[1].split("\n\nVEREDICTO:", 1)[0]
    veredicto = prompt.split("VEREDICTO:\n", 1)[1].split("\n\nDevolvé", 1)[0]
    assert chat == "x" * 1800
    assert veredicto == "w" * 1000


def test_el_perfil_de_ollama_vuelve_limpio(ollama):
    assert M._summarize_profile("Ilan", "", "c", "v") == "perfil nuevo"


def test_un_perfil_entre_fences_pierde_los_fences(ollama):
    ollama["set"](ollama["R"]("```markdown\n**Quién es**: Mica\n```"))
    assert M._summarize_profile("Ilan", "", "c", "v") == "**Quién es**: Mica"


def test_si_ollama_falla_sigue_el_proveedor(ollama, monkeypatch):
    ollama["set"](RuntimeError("conexión rechazada"))
    monkeypatch.setattr(L, "complete_fast", lambda **kw: "perfil del proveedor")
    assert M._summarize_profile("Ilan", "lo previo", "c", "v") == "perfil del proveedor"


def test_un_http_500_de_ollama_tambien_cae_al_proveedor(ollama, monkeypatch):
    ollama["set"](ollama["R"](status=500))
    monkeypatch.setattr(L, "complete_fast", lambda **kw: "perfil del proveedor")
    assert M._summarize_profile("Ilan", "", "c", "v") == "perfil del proveedor"


def test_el_proveedor_tambien_pierde_los_fences(monkeypatch):
    monkeypatch.setattr(settings, "ollama_enabled", False)
    monkeypatch.setattr(L, "complete_fast", lambda **kw: "```\nperfil\n```")
    assert M._summarize_profile("Ilan", "", "c", "v") == "perfil"


def test_sin_ningun_backend_queda_el_perfil_previo(monkeypatch):
    monkeypatch.setattr(settings, "ollama_enabled", False)
    monkeypatch.setattr(L, "complete_fast", lambda **kw: "")
    assert M._summarize_profile("Ilan", "lo previo", "c", "v") == "lo previo"


def test_si_el_proveedor_explota_queda_el_perfil_previo(monkeypatch):
    def explota(**kw):
        raise RuntimeError("sin credito")

    monkeypatch.setattr(settings, "ollama_enabled", False)
    monkeypatch.setattr(L, "complete_fast", explota)
    assert M._summarize_profile("Ilan", "lo previo", "c", "v") == "lo previo"


def test_un_analisis_sin_modelo_no_pierde_el_perfil_que_ya_habia(db, monkeypatch):
    """El peor caso: el modelo no responde y el perfil guardado no se borra."""
    monkeypatch.setattr(settings, "ollama_enabled", False)
    monkeypatch.setattr(L, "complete_fast", lambda **kw: "")
    cid = M.create_contact("Mica")
    M._save(cid, "perfil que costó", [])
    M.update_after_analysis(cid, "charla", "veredicto")
    c = M.get_contact(cid)
    assert c["profile"] == "perfil que costó"
    assert len(c["history"]) == 1  # el intercambio igual se registra


# --- migracion del esquema viejo ------------------------------------------

def test_el_esquema_viejo_se_migra_conservando_los_datos(db):
    """Base con `name` como PK (sin columna `id`): migra y no pierde nada."""
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE contacts (name TEXT PRIMARY KEY, profile TEXT, "
              "history TEXT, updated_at TEXT)")
    c.execute("INSERT INTO contacts VALUES(?,?,?,?)",
              ("Mica", "perfil viejo", json.dumps([{"chat": "hola"}]), "2026-01-01T00:00:00"))
    c.commit()
    c.close()

    filas = M.list_contacts()
    assert len(filas) == 1
    migrada = M.get_contact(filas[0]["id"])
    assert migrada["name"] == "Mica"
    assert migrada["profile"] == "perfil viejo"
    assert migrada["history"] == [{"chat": "hola"}]
    assert migrada["apellido"] == "" and migrada["tag"] == ""


def test_despues_de_migrar_la_tabla_vieja_no_queda(db):
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE contacts (name TEXT PRIMARY KEY, profile TEXT, "
              "history TEXT, updated_at TEXT)")
    c.commit()
    c.close()
    M.list_contacts()
    tablas = [r[0] for r in sqlite3.connect(str(db)).execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    assert "contacts_old" not in tablas
    assert "contacts" in tablas


def test_una_base_ya_migrada_no_se_vuelve_a_migrar(db):
    cid = M.create_contact("Mica", "Wurch")
    M.list_contacts()
    M.list_contacts()
    assert M.get_contact(cid)["apellido"] == "Wurch"
    assert len(M.list_contacts()) == 1


def test_la_carpeta_de_la_base_se_crea_sola(db):
    assert not db.parent.exists()
    M.create_contact("Mica")
    assert db.exists()
