"""Tests del cuerpo de la UI de `app/streamlit_app.py`, con el runtime de Streamlit.

`tests/test_streamlit_app.py` cubre la mitad de arriba del archivo: lee la
fuente, corta en el marcador `# ============ UI ============` y ejecuta sólo los
helpers. Lo de abajo del marcador no se puede probar así, porque son llamadas a
`st` encadenadas que necesitan un ScriptRunner de verdad para producir algo
observable.

Acá se usa `streamlit.testing.v1.AppTest`, que corre el script entero en un
thread de este mismo proceso y expone los elementos que dibujó. Eso permite
probar lo que hasta ahora no tenía ninguna prueba y es justamente donde vive el
comportamiento que el usuario ve:

- **El panel de modelos.** Si no hay ningún backend configurado la app corta ahí
  con un mensaje; si hay, el orden de las opciones decide con qué contesta por
  defecto (los gratis primero, a propósito).
- **El flujo del Tab 1 de punta a punta**: el texto pegado, el RAG, la memoria
  de la chica, el prompt que se arma y lo que pasa cuando algo de eso falla.
- **La degradación.** Un índice caído o un proveedor sin cuota no pueden dejar
  la pantalla en blanco: tienen que avisar y seguir.
- **Las notas del Tab 3**, que son el único lugar donde la app escribe a disco.
- **El camino de las capturas**, que es como se usa la app en el celular: subir
  el screenshot, leerlo (con el OCR local o con el modelo de visión) y que el
  texto termine en la consulta.

Nada de esto toca Qdrant, bge-m3, Ollama ni Anthropic: todo lo caro se reemplaza
en el módulo de origen, porque `streamlit_app.py` hace `from ... import nombre` y
vuelve a resolver esos nombres en cada rerun. El `Banco` de abajo es a la vez el
doble de todo eso y el registro de con qué lo llamaron.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from coach_brain.settings import settings

FUENTE = Path(__file__).resolve().parents[1] / "src" / "coach_brain" / "app" / "streamlit_app.py"


def widget(elementos, texto: str):
    """Agarra un widget por su etiqueta.

    Los del panel lateral no tienen `key`, así que no se pueden pedir por clave,
    y el orden del listado plano mezcla el panel con los tabs.
    """
    for e in elementos:
        if texto in e.label:
            return e
    raise AssertionError(f"no hay widget con '{texto}': {[e.label for e in elementos]}")


def textos(elementos) -> str:
    """Todo lo que dibujó una familia de elementos, junto, para buscar adentro."""
    return "\n".join(e.value for e in elementos)


class Res:
    """Lo que devuelve `retrieve`: tres listas y el formateo para el prompt."""

    def __init__(self):
        self.principles = [{"id": "p1", "texto": "no persigas"}]
        self.situations = [{"id": "s1", "texto": "quedo fria"}]
        self.style = [{"id": "f1", "texto": "dale igual"}]

    def format_for_prompt(self) -> dict:
        return {
            "principles_retrieved": "PRINC-RECUPERADO",
            "situations_retrieved": "SIT-RECUPERADA",
            "style_retrieved": "FRASE-RECUPERADA",
        }


class Banco:
    """El doble de todo lo caro, y el registro de con qué lo llamaron.

    Cada test ajusta los atributos antes de arrancar la app, así que el estado
    del mundo (qué contactos hay, si el índice responde, si el modelo explota)
    se declara en el test y no en un fixture por caso.
    """

    def __init__(self):
        self.llm: list[dict] = []
        self.fast: list[dict] = []
        self.consultas: list[str] = []
        self.creados: list[tuple] = []
        self.memoria: list[tuple] = []
        self.respuesta = 'Diagnóstico: quedó fría.\n\nLíneas posibles:\n1. "dale igual"\n'
        self.contactos: list[dict] = []
        self.por_nombre: list[dict] = []
        self.perfiles: dict[str, dict] = {}
        self.nombre_detectado = "desconocido"
        self.rag_explota: Exception | None = None
        self.llm_explota: Exception | None = None
        self.providers: list = []

    # --- los dobles ---
    def retrieve(self, q: str) -> Res:
        self.consultas.append(q)
        if self.rag_explota:
            raise self.rag_explota
        return Res()

    def complete(self, **kw) -> str:
        self.llm.append(kw)
        if self.llm_explota:
            raise self.llm_explota
        return self.respuesta

    def complete_fast(self, **kw) -> str:
        self.fast.append(kw)
        return self.nombre_detectado

    def create_contact(self, *a) -> str:
        self.creados.append(a)
        return "cid-nuevo"

    def update_after_analysis(self, cid, chat, answer, user="Ilan") -> str:
        self.memoria.append((cid, chat, answer, user))
        return "PERFIL-ACTUALIZADO"

    # --- el arrancador ---
    def arrancar(self, **estado) -> AppTest:
        at = AppTest.from_file(str(FUENTE), default_timeout=90)
        for k, v in estado.items():
            at.session_state[k] = v
        at.run()
        return at


@pytest.fixture
def banco(monkeypatch, tmp_path):
    """Deja la app corrible sin GPU, sin red y sin tocar el disco del proyecto."""
    import coach_brain.kb.embed as E
    import coach_brain.kb.index as I
    import coach_brain.kb.memory as M
    import coach_brain.kb.retrieve as R
    import coach_brain.llm as L

    # Los caches son del proceso: sin limpiar, el primer test le presta su
    # `warmup` y su diccionario de intentos de login a todos los demás.
    st.cache_resource.clear()
    st.cache_data.clear()

    b = Banco()

    monkeypatch.setattr(I, "get_qdrant", lambda: "qdrant-falso")
    monkeypatch.setattr(E, "get_model", lambda: "modelo-falso")
    monkeypatch.setattr(E, "embed_one", lambda t: [0.0] * 4)
    monkeypatch.setattr(R, "retrieve", b.retrieve)
    monkeypatch.setattr(L, "complete", b.complete)
    monkeypatch.setattr(L, "complete_fast", b.complete_fast)
    monkeypatch.setattr(L, "available_providers", lambda: b.providers)
    monkeypatch.setattr(M, "list_contacts", lambda: b.contactos)
    monkeypatch.setattr(M, "find_by_name", lambda n: b.por_nombre)
    monkeypatch.setattr(M, "get_contact", lambda i: b.perfiles.get(i))
    monkeypatch.setattr(M, "create_contact", b.create_contact)
    monkeypatch.setattr(M, "update_after_analysis", b.update_after_analysis)

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "app_password", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "ollama_enabled", True)
    monkeypatch.setattr(settings, "qdrant_url", "")

    yield b

    st.cache_resource.clear()
    st.cache_data.clear()


def analizar(at, texto: str = "[ella] hola\n[yo] hey", **widgets):
    """Pega un chat en el Tab 1, ajusta lo que pidan y aprieta Analizar."""
    if texto is not None:
        at.text_area(key="chat_manual_tab1").set_value(texto)
    for k, v in widgets.items():
        at.text_input(key=k).set_value(v)
    at.button(key="go_tab1").click().run()
    return at


# ===========================================================================
# El arranque y el panel de modelos
# ===========================================================================


def test_la_app_arranca_entera_sin_excepciones(banco):
    """La prueba más barata y la que más vale: un `NameError` en el cuerpo de la
    UI no lo agarra ningún test de helpers, y en Fly.io se ve como una pantalla
    en blanco."""
    at = banco.arrancar()

    assert not at.exception, [e.message for e in at.exception]
    assert at.title[0].value == "🎯 Coach Asistente"


def test_las_tres_tabs_estan_con_etiquetas_cortas(banco):
    """Con los nombres largos las tabs medían 407 px y no entraban en los 349
    útiles de un celular de 375 px."""
    at = banco.arrancar()

    assert [t.label for t in at.tabs] == ["💬 Chat", "🎯 Líneas", "📝 Frases"]


def test_sin_ningun_modelo_configurado_la_app_corta_con_un_aviso(banco, monkeypatch):
    """Sin backend no hay nada que hacer, y dejar los tabs a la vista haría que
    el error aparezca recién después de escribir toda la consulta."""
    monkeypatch.setattr(settings, "ollama_enabled", False)

    at = banco.arrancar()

    assert "No hay ningún modelo configurado" in at.error[0].value
    assert not at.tabs, "no se dibujó la UI después del corte"


def test_los_gratis_van_antes_que_los_pagos(banco, monkeypatch):
    """El orden es el default: `index=0` es con lo que contesta si nadie toca
    nada, y no puede ser el que cobra por token."""
    import types

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-de-prueba")
    banco.providers = [types.SimpleNamespace(label="Gemini", key="gemini", model="flash")]

    at = banco.arrancar()
    opciones = widget(at.radio, "¿Con qué genero la respuesta?").options

    assert "Ollama" in opciones[0]
    assert "Gemini" in opciones[1]
    assert "💳" in opciones[2] and "💳" in opciones[3], "los pagos quedan al final"


def test_sin_key_de_anthropic_no_se_ofrecen_los_pagos(banco):
    at = banco.arrancar()
    opciones = widget(at.radio, "¿Con qué genero la respuesta?").options

    assert len(opciones) == 1
    assert "💳" not in opciones[0]


def test_sin_key_tampoco_se_ofrece_la_vision_paga(banco, monkeypatch):
    """La visión por API es sólo de Anthropic: ofrecerla sin key sería un error
    recién al momento de leer la imagen, con el screenshot ya subido."""
    at = banco.arrancar()
    assert widget(at.radio, "¿Cómo leo las imágenes?").options == ["🟢 OCR local (gratis)"]

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-de-prueba")
    at = banco.arrancar()
    assert len(widget(at.radio, "¿Cómo leo las imágenes?").options) == 2


def test_sin_pin_configurado_no_hay_boton_de_salir(banco):
    """Corriendo local no hay sesión que cerrar."""
    at = banco.arrancar()
    assert not [b for b in at.button if "Cerrar sesión" in b.label]


def test_con_pin_el_boton_de_salir_vuelve_a_la_pantalla_del_pin(banco, monkeypatch):
    monkeypatch.setattr(settings, "app_password", "1234")

    at = banco.arrancar(_authed=True)
    widget(at.button, "Cerrar sesión").click().run()

    assert not at.tabs, "la UI no se dibuja después de salir"
    assert at.title[0].value == "🔒 Coach Asistente", "volvió el formulario del PIN"
    assert at.text_input(key="_pin_input") is not None


# ===========================================================================
# TAB 1 — analizar un chat
# ===========================================================================


def test_sin_imagen_ni_texto_avisa_y_no_gasta_tokens(banco):
    at = analizar(banco.arrancar(), texto="")

    assert "Subí una imagen o pegá texto." in textos(at.warning)
    assert banco.llm == [], "no se llamó al modelo con las manos vacías"


def test_el_texto_pegado_llega_al_modelo_con_el_backend_elegido(banco):
    at = analizar(banco.arrancar())

    assert len(banco.llm) == 1
    assert "[ella] hola" in banco.llm[0]["user"]
    assert banco.llm[0]["backend"] == "ollama"
    assert banco.llm[0]["model"] == settings.ollama_model
    assert banco.respuesta.splitlines()[0] in textos(at.markdown)


def test_el_material_recuperado_entra_en_el_prompt(banco):
    """Es el RAG entero: sin esto la app contesta con el prompt base y los
    38.343 vectores del índice no aportan nada."""
    at = analizar(banco.arrancar())

    assert banco.consultas, "no se consultó el índice"
    assert "PRINC-RECUPERADO" in banco.llm[0]["system"]
    assert "SIT-RECUPERADA" in banco.llm[0]["system"]
    assert "FRASE-RECUPERADA" in banco.llm[0]["system"]
    assert [e for e in at.expander if "1 ppios" in e.label], [e.label for e in at.expander]


def test_con_el_rag_apagado_no_se_consulta_el_indice(banco):
    """El checkbox existe para poder comparar respuestas con y sin material, y
    para contestar aunque Qdrant esté caído."""
    at = banco.arrancar()
    widget(at.checkbox, "Usar RAG").uncheck().run()
    analizar(at)

    assert banco.consultas == []
    assert "(RAG desactivado)" in banco.llm[0]["system"]


def test_si_el_indice_falla_la_consulta_sigue_sin_material(banco):
    """Qdrant embebido se cuelga si otro proceso tiene el lock (gotcha #11).
    Eso no puede dejar al usuario sin respuesta."""
    banco.rag_explota = RuntimeError("storage locked")

    at = analizar(banco.arrancar())

    assert "RAG no disponible aún" in textos(at.warning)
    assert len(banco.llm) == 1, "contestó igual"


def test_si_el_modelo_falla_se_muestra_el_error_traducido(banco):
    """El volcado crudo del proveedor no dice que la salida es elegir otro
    modelo en el panel lateral."""
    banco.llm_explota = RuntimeError("Your credit balance is too low")

    at = analizar(banco.arrancar())

    assert "no tiene créditos" in at.error[0].value
    assert not at.exception, "el error se muestra, no se propaga"


def test_el_modo_elegido_viaja_en_el_prompt(banco):
    at = banco.arrancar()
    at.radio(key="mode_tab1").set_value("🔥 Picante").run()
    analizar(at)

    assert "Subí la tensión" in banco.llm[0]["user"]
    assert "banter" in banco.llm[0]["user"], "el empujón de humor va siempre"


def test_la_pregunta_opcional_viaja_y_si_falta_va_la_de_siempre(banco):
    analizar(banco.arrancar(), userq_tab1="¿la invito?")
    assert "¿la invito?" in banco.llm[0]["user"]

    banco.llm.clear()
    analizar(banco.arrancar())
    assert "¿qué le respondo?" in banco.llm[0]["user"]


def test_el_nombre_del_usuario_entra_en_el_system(banco):
    at = banco.arrancar()
    at.text_input(key="user_name").set_value("Ilan").run()
    analizar(at)

    assert "Ilan" in banco.llm[0]["system"]


def test_la_respuesta_queda_pegada_al_tocar_otro_widget(banco):
    """Sin persistirla en `session_state`, cualquier rerun (tocar un radio,
    tipear) borraba el análisis y había que pagarlo otra vez."""
    at = analizar(banco.arrancar())
    at.text_input(key="userq_tab1").set_value("otra cosa").run()

    assert "↻ Último análisis" in textos(at.caption)
    assert banco.respuesta.splitlines()[0] in textos(at.markdown)
    assert len(banco.llm) == 1, "no se volvió a llamar al modelo"


# --- la memoria por contacto ------------------------------------------------


def test_con_un_nombre_nuevo_se_crea_el_contacto_y_se_guarda_el_contexto(banco):
    analizar(banco.arrancar(), name_tab1="Mica")

    assert banco.creados == [("Mica", "", "")]
    assert banco.memoria and banco.memoria[0][0] == "cid-nuevo"
    assert "[ella] hola" in banco.memoria[0][1]


def test_el_apellido_y_el_tag_se_guardan_con_el_nombre(banco):
    at = banco.arrancar()
    at.text_input(key="name_tab1").set_value("Mica")
    at.text_input(key="ape_tab1").set_value("Pérez")
    at.text_input(key="tag_tab1").set_value("gimnasio").run()
    analizar(at)

    assert banco.creados == [("Mica", "Pérez", "gimnasio")]


def test_un_nombre_repetido_avisa_antes_de_analizar(banco):
    """Dos 'Mica' sin distinguir mezclan los perfiles, y el perfil es lo que
    después entra en el prompt de todas las consultas."""
    banco.por_nombre = [{"label": "Mica (gym)"}, {"label": "Mica (IG)"}]

    at = banco.arrancar()
    at.text_input(key="name_tab1").set_value("Mica").run()

    assert "Ya hay 2 'Mica'" in textos(at.warning)
    assert "apellido o tag" in textos(at.warning)


def test_con_apellido_el_aviso_de_repetido_no_molesta(banco):
    banco.por_nombre = [{"label": "Mica (gym)"}, {"label": "Mica (IG)"}]

    at = banco.arrancar()
    at.text_input(key="name_tab1").set_value("Mica")
    at.text_input(key="ape_tab1").set_value("Pérez").run()

    assert not [w for w in at.warning if "Ya hay" in w.value]


def test_elegir_una_guardada_mete_su_perfil_en_el_prompt(banco):
    banco.contactos = [{"id": "c1", "label": "Mica (gym)"}]
    banco.perfiles = {"c1": {"label": "Mica (gym)", "profile": "PERFIL-DE-MICA", "history": []}}

    at = banco.arrancar()
    at.selectbox(key="contact_sel_tab1").select("Mica (gym)").run()
    assert "PERFIL-DE-MICA" in textos(at.markdown), "el perfil se puede leer antes de analizar"

    analizar(at)
    assert "PERFIL-DE-MICA" in banco.llm[0]["system"]
    assert banco.creados == [], "no se creó un contacto nuevo"


def test_sin_memoria_el_system_lo_dice_explicito(banco):
    """El hueco tiene que estar nombrado: con el placeholder sin reemplazar, el
    prompt le pedía al modelo datos de una variable literal."""
    analizar(banco.arrancar())

    assert "(sin historial de esta persona)" in banco.llm[0]["system"]


def test_la_autodeteccion_usa_la_memoria_de_la_unica_coincidencia(banco):
    """Es el caso cómodo: pegar el screenshot y que la app sepa de quién es."""
    banco.nombre_detectado = "Mica"
    banco.por_nombre = [{"id": "c1", "label": "Mica (gym)"}]
    banco.perfiles = {"c1": {"label": "Mica (gym)", "profile": "PERFIL-DE-MICA", "history": []}}

    at = analizar(banco.arrancar())

    assert banco.fast, "no se usó el modelo rápido para detectar"
    assert "Detecté **Mica**" in textos(at.info)
    assert "PERFIL-DE-MICA" in banco.llm[0]["system"]


def test_si_la_deteccion_matchea_varias_no_mezcla_perfiles(banco):
    banco.nombre_detectado = "Mica"
    banco.por_nombre = [{"id": "c1", "label": "Mica (gym)"}, {"id": "c2", "label": "Mica (IG)"}]

    at = analizar(banco.arrancar())

    assert "Elegila arriba para no mezclar" in textos(at.warning)
    assert "(sin historial de esta persona)" in banco.llm[0]["system"]


def test_un_nombre_detectado_que_no_existe_arranca_su_memoria(banco):
    banco.nombre_detectado = "Mica"
    banco.por_nombre = []

    at = analizar(banco.arrancar())

    assert "(nueva)" in textos(at.info)
    assert banco.creados == [("Mica", "", "")]


def test_si_no_detecta_nada_lo_dice_y_sigue(banco):
    """`desconocido` es la respuesta prevista del extractor cuando el chat no
    trae nombre: no es un error, es que no hay memoria para esta consulta."""
    at = analizar(banco.arrancar())

    assert "No detecté el nombre" in textos(at.caption)
    assert banco.creados == []
    assert len(banco.llm) == 1


def test_con_la_autodeteccion_apagada_no_se_gasta_una_llamada(banco):
    at = banco.arrancar()
    at.checkbox(key="autodetect_tab1").uncheck().run()
    analizar(at)

    assert banco.fast == []
    assert "Tip: elegí o creá la chica" in textos(at.caption)


def test_si_la_memoria_no_se_puede_actualizar_la_respuesta_igual_queda(banco, monkeypatch):
    """Actualizar el perfil es otra llamada al modelo: si falla, lo que ya se
    pagó (el análisis) no se puede perder."""
    import coach_brain.kb.memory as M

    monkeypatch.setattr(M, "update_after_analysis", lambda *a, **k: 1 / 0)

    at = analizar(banco.arrancar(), name_tab1="Mica")

    assert "No pude actualizar memoria" in textos(at.caption)
    assert banco.respuesta.splitlines()[0] in textos(at.markdown)
    assert not at.exception


# ===========================================================================
# TAB 2 — líneas situacionales
# ===========================================================================


def generar(at, situacion: str = "en el gym, primera vez que la veo"):
    if situacion is not None:
        at.text_area(key="situation_input").set_value(situacion)
    at.button(key="go_sit").click().run()
    return at


def test_sin_situacion_descrita_avisa(banco):
    at = generar(banco.arrancar(), situacion="")

    assert "Describí la situación primero." in textos(at.warning)
    assert banco.llm == []


def test_la_situacion_y_lo_que_se_necesita_llegan_al_modelo(banco):
    generar(banco.arrancar())

    assert "en el gym" in banco.llm[0]["user"]
    assert "Opener" in banco.llm[0]["user"], "el tipo de necesidad va con su descripción"
    assert "(sin información guardada)" in banco.llm[0]["user"]


def test_el_tipo_de_necesidad_elegido_es_el_que_viaja(banco):
    at = banco.arrancar()
    at.radio(key="need_type").set_value("🔥 Sexualizar").run()
    generar(at)

    assert "territorio sexual" in banco.llm[0]["user"]


def test_las_lineas_tambien_usan_el_material_del_coach(banco):
    generar(banco.arrancar())

    assert banco.consultas, "no se consultó el índice"
    assert "PRINC-RECUPERADO" in banco.llm[0]["user"]


def test_el_perfil_de_la_elegida_entra_en_las_lineas(banco):
    banco.contactos = [{"id": "c1", "label": "Mica (gym)"}]
    banco.perfiles = {"c1": {"label": "Mica (gym)", "profile": "PERFIL-DE-MICA"}}

    at = banco.arrancar()
    at.selectbox(key="contact_sit").select("Mica (gym)").run()
    generar(at)

    assert "PERFIL-DE-MICA" in banco.llm[0]["user"]


def test_las_frases_personales_entran_y_se_avisa(banco, tmp_path):
    """Es la razón de ser del Tab 3: lo que Ilan guarda ahí tiene que aparecer
    acá, y el aviso es lo que lo hace evidente."""
    (tmp_path / "personal_notes.txt").write_text("MI-FRASE-DE-SIEMPRE", encoding="utf-8")

    at = generar(banco.arrancar())

    assert "MI-FRASE-DE-SIEMPRE" in banco.llm[0]["user"]
    assert "Tus frases personales" in textos(at.caption)


def test_sin_frases_personales_no_se_promete_nada(banco):
    at = generar(banco.arrancar())

    assert "Tus frases personales" not in textos(at.caption)


def test_si_el_modelo_falla_generando_lineas_se_avisa(banco):
    banco.llm_explota = RuntimeError("429 quota exceeded")

    at = generar(banco.arrancar())

    assert "límite de su cuota" in at.error[0].value
    assert not at.exception


def test_las_lineas_quedan_pegadas_al_tocar_otro_widget(banco):
    at = generar(banco.arrancar())
    at.radio(key="sit_mode").set_value("🌫️ Misterio").run()

    assert "↻ Últimas generadas" in textos(at.caption)
    assert len(banco.llm) == 1


# ===========================================================================
# TAB 3 — las notas personales (el único disco que toca la app)
# ===========================================================================


def guardar(at, texto: str):
    at.text_area(key="new_note_input").set_value(texto)
    at.button(key="save_note").click().run()
    return at


def test_sin_notas_se_invita_a_agregar(banco):
    at = banco.arrancar()
    assert "No hay notas guardadas todavía" in textos(at.info)


def test_una_nota_se_escribe_en_disco(banco, tmp_path):
    at = guardar(banco.arrancar(), "no llenar el espacio")

    assert (tmp_path / "personal_notes.txt").read_text(encoding="utf-8") == "no llenar el espacio"
    assert "Total: 20 caracteres" in textos(at.caption)


def test_guardar_vacio_avisa_y_no_crea_el_archivo(banco, tmp_path):
    at = guardar(banco.arrancar(), "   ")

    assert "Escribí algo primero." in textos(at.warning)
    assert not (tmp_path / "personal_notes.txt").exists()


def test_la_segunda_nota_se_agrega_sin_pisar_la_primera(banco, tmp_path):
    """Es un apéndice, no un reemplazo: lo que se pega acá viene de iCloud
    Notes de a una nota por vez."""
    at = guardar(banco.arrancar(), "la primera")
    guardar(at, "la segunda")

    guardado = (tmp_path / "personal_notes.txt").read_text(encoding="utf-8")
    assert "la primera" in guardado
    assert "la segunda" in guardado
    assert "---" in guardado, "el separador es lo que las deja legibles"


def test_las_notas_guardadas_se_muestran_en_solo_lectura(banco, tmp_path):
    (tmp_path / "personal_notes.txt").write_text("MI-FRASE", encoding="utf-8")

    at = banco.arrancar()
    caja = at.text_area(key="notes_display")

    assert caja.value == "MI-FRASE"
    assert caja.disabled, "editar acá daría la ilusión de guardar"


def test_borrar_las_notas_las_vacia(banco, tmp_path):
    (tmp_path / "personal_notes.txt").write_text("MI-FRASE", encoding="utf-8")

    at = banco.arrancar()
    widget(at.button, "Borrar todas las notas").click().run()

    assert (tmp_path / "personal_notes.txt").read_text(encoding="utf-8") == ""
    assert "No hay notas guardadas todavía" in textos(at.info)


def test_sin_notas_no_hay_boton_de_borrar(banco):
    """No hay nada que borrar, y el botón al lado del de guardar se aprieta
    solo."""
    at = banco.arrancar()
    assert not [b for b in at.button if "Borrar" in b.label]


def test_si_el_indice_falla_las_lineas_se_generan_igual(banco):
    """Mismo trato que en el Tab 1: el índice caído degrada la calidad de la
    respuesta, no la cancela."""
    banco.rag_explota = RuntimeError("storage locked")

    at = generar(banco.arrancar())

    assert "RAG no disponible" in textos(at.warning)
    assert len(banco.llm) == 1, "generó igual"


# ===========================================================================
# TAB 1 — las capturas de pantalla
# ===========================================================================


def png(w: int = 100, h: int = 100) -> bytes:
    """Un PNG de verdad: `Image.open` corre sobre lo que se sube, sin parchear."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def caja(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def poner_ocr(monkeypatch, cajas):
    """Reemplaza el motor de OCR en el paquete, que es de donde `_get_ocr` lo
    importa: dentro del script de la app no se puede parchear nada, porque se
    vuelve a ejecutar entero en cada rerun."""
    import rapidocr_onnxruntime

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", lambda: (lambda _arr: (cajas, None)))


def subir(at, *archivos):
    at.file_uploader(key="upload_tab1").set_value(
        [(nombre, datos, "image/png") for nombre, datos in archivos]
    )
    at.button(key="go_tab1").click().run()
    return at


def test_una_captura_leida_con_ocr_llega_al_modelo(banco, monkeypatch):
    """El camino principal de la app: sacar el screenshot y subirlo. Hasta acá
    no tenía ninguna prueba de punta a punta."""
    poner_ocr(monkeypatch, [
        (caja(5, 40, 50, 50), "hola", 0.9),
        (caja(55, 60, 95, 70), "todo bien?", 0.9),
    ])

    at = subir(banco.arrancar(), ("chat.png", png()))

    assert "[ella] hola" in banco.llm[0]["user"]
    assert [e for e in at.expander if "1 captura/s" in e.label]
    assert "hola" in textos(at.code)


def test_varias_capturas_se_pegan_en_el_orden_en_que_se_subieron(banco, monkeypatch):
    """Una conversación larga no entra en una pantalla: el orden es el del
    scroll, y al revés el diagnóstico sale al revés."""
    poner_ocr(monkeypatch, [(caja(5, 40, 50, 50), "hola", 0.9)])

    at = subir(banco.arrancar(), ("1.png", png()), ("2.png", png()))

    assert banco.llm[0]["user"].count("[ella] hola") == 2
    assert [e for e in at.expander if "2 captura/s" in e.label]


def test_una_captura_sin_texto_no_manda_una_consulta_vacia(banco, monkeypatch):
    """Una foto que no es un chat (un paisaje, una pantalla negra) no puede
    gastar una llamada al modelo con el prompt vacío."""
    poner_ocr(monkeypatch, [])

    at = subir(banco.arrancar(), ("nada.png", png()))

    assert "(no se detectó texto)" in textos(at.code)
    assert banco.llm == []


def test_el_texto_de_la_captura_tambien_alimenta_la_autodeteccion(banco, monkeypatch):
    """El nombre suele estar en el encabezado, que no es un mensaje: por eso lo
    que se le pasa al detector es el crudo del OCR, no el chat ya armado."""
    poner_ocr(monkeypatch, [
        (caja(20, 2, 60, 10), "Mica en línea", 0.9),
        (caja(5, 40, 50, 50), "hola", 0.9),
    ])
    banco.nombre_detectado = "Mica"
    banco.por_nombre = [{"id": "c1", "label": "Mica (gym)"}]
    banco.perfiles = {"c1": {"label": "Mica (gym)", "profile": "PERFIL-DE-MICA", "history": []}}

    subir(banco.arrancar(), ("chat.png", png()))

    assert "Mica" in banco.fast[0]["user"], "el encabezado viaja al detector"
    assert "PERFIL-DE-MICA" in banco.llm[0]["system"]


def test_con_la_vision_paga_se_lee_con_el_modelo_y_no_con_el_ocr(banco, monkeypatch):
    """La opción de pago existe para los screenshots que el OCR no agarra
    (fondos con imagen, stickers). Lo que importa es que no termine llamando a
    los dos."""
    import anthropic

    visto = {}

    class Mensajes:
        def create(self, **kw):
            visto.update(kw)
            texto = '{"messages":[{"sender":"ella","text":"hola"}],"notas":"la vi escribiendo"}'
            return type("R", (), {"content": [type("B", (), {"text": texto})()]})()

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda **kw: type("C", (), {"messages": Mensajes()})()
    )
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-de-prueba")
    poner_ocr(monkeypatch, [(caja(5, 40, 50, 50), "ESTO-NO-TIENE-QUE-APARECER", 0.9)])

    at = banco.arrancar()
    widget(at.radio, "¿Cómo leo las imágenes?").set_value("💳 Haiku vision (pago)").run()
    subir(at, ("chat.png", png()))

    assert visto["model"] == settings.model_fast, "lee con el barato, no con el grande"
    assert "[ella] hola" in banco.llm[0]["user"]
    assert "la vi escribiendo" in banco.llm[0]["user"], "las notas del extractor viajan"
    assert "ESTO-NO-TIENE-QUE-APARECER" not in banco.llm[0]["user"]
