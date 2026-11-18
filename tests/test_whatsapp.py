"""Parseo del export de WhatsApp (iOS, español).

Es la fuente más grande del corpus y la única que trae roles: si el parser se
come un mensaje del coach, el extractor no tiene de dónde sacar el consejo.
Todo corre sobre un `_chat.txt` escrito en `tmp_path`.
"""

from __future__ import annotations

import json

import pytest

from coach_brain.ingest import whatsapp as W
from coach_brain.settings import settings


def chat(tmp_path, *lineas: str):
    """Escribe un `_chat.txt` y lo parsea."""
    p = tmp_path / "_chat.txt"
    p.write_text("\n".join(lineas), encoding="utf-8")
    return W.parse_chat(p)


# --- una línea normal ------------------------------------------------------

def test_una_linea_da_remitente_rol_y_texto(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 7:27:32 p. m.] Ema Pereyra: dale, probá eso")
    assert len(msgs) == 1
    assert msgs[0]["sender"] == "Ema Pereyra"
    assert msgs[0]["role"] == "coach"
    assert msgs[0]["text"] == "dale, probá eso"
    assert msgs[0]["attachment"] is None


def test_el_alumno_tiene_su_propio_rol(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 7:27:32 p. m.] Ilan Daniele: listo")
    assert msgs[0]["role"] == "alumno"


def test_un_remitente_desconocido_es_otro(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 7:27:32 p. m.] Mica: hola")
    assert msgs[0]["role"] == "otro"
    assert msgs[0]["sender"] == "Mica"


def test_un_mensaje_vacio_se_descarta(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 7:27:32 p. m.] Ema Pereyra: ")
    assert msgs == []


def test_el_texto_puede_tener_dos_puntos(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 7:27:32 p. m.] Ema Pereyra: mirá: esto va así")
    assert msgs[0]["text"] == "mirá: esto va así"


def test_una_linea_que_no_es_un_mensaje_y_no_hay_anterior_se_tira(tmp_path):
    assert chat(tmp_path, "basura sin timestamp") == []


def test_las_lineas_en_blanco_no_cuentan(tmp_path):
    msgs = chat(tmp_path, "", "   ", "[6/5/23, 7:27:32 p. m.] Mica: hola", "")
    assert len(msgs) == 1


# --- la hora: el p. m. del iOS español ------------------------------------

def test_la_tarde_pasa_a_24_horas(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 7:27:32 p. m.] Mica: x")
    assert msgs[0]["ts"] == "2023-05-06T19:27:32"


def test_las_doce_de_la_noche_son_cero(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 12:05:00 a. m.] Mica: x")
    assert msgs[0]["ts"] == "2023-05-06T00:05:00"


def test_el_mediodia_sigue_siendo_las_doce(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 12:05:00 p. m.] Mica: x")
    assert msgs[0]["ts"] == "2023-05-06T12:05:00"


def test_la_manana_queda_igual(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 9:05:00 a. m.] Mica: x")
    assert msgs[0]["ts"] == "2023-05-06T09:05:00"


def test_sin_am_pm_se_lee_como_24_horas(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 19:27:32] Mica: x")
    assert msgs[0]["ts"] == "2023-05-06T19:27:32"


def test_un_ano_de_dos_digitos_es_del_dos_mil(tmp_path):
    msgs = chat(tmp_path, "[1/1/24, 10:00:00] Mica: x")
    assert msgs[0]["ts"].startswith("2024-01-01")


def test_un_ano_de_cuatro_digitos_se_respeta(tmp_path):
    msgs = chat(tmp_path, "[1/1/2024, 10:00:00] Mica: x")
    assert msgs[0]["ts"].startswith("2024-01-01")


def test_el_dia_va_antes_del_mes(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 10:00:00] Mica: x")
    assert msgs[0]["ts"].startswith("2023-05-06")  # 6 de mayo, no 5 de junio


def test_una_fecha_imposible_deja_el_ts_en_none_pero_no_pierde_el_mensaje(tmp_path):
    msgs = chat(tmp_path, "[31/2/23, 10:00:00] Mica: igual lo dijo")
    assert len(msgs) == 1
    assert msgs[0]["ts"] is None
    assert msgs[0]["text"] == "igual lo dijo"


# --- multilínea ------------------------------------------------------------

def test_la_continuacion_se_pega_al_mensaje_anterior(tmp_path):
    msgs = chat(tmp_path,
                "[6/5/23, 10:00:00] Ema Pereyra: primera",
                "segunda línea",
                "tercera línea")
    assert len(msgs) == 1
    assert msgs[0]["text"] == "primera\nsegunda línea\ntercera línea"


def test_un_mensaje_nuevo_corta_la_continuacion(tmp_path):
    msgs = chat(tmp_path,
                "[6/5/23, 10:00:00] Ema Pereyra: uno",
                "sigue uno",
                "[6/5/23, 10:01:00] Ilan Daniele: dos")
    assert [m["text"] for m in msgs] == ["uno\nsigue uno", "dos"]


# --- adjuntos --------------------------------------------------------------

@pytest.mark.parametrize("cuerpo,kind,etiqueta", [
    ("imagen omitida", "imagen", "[imagen]"),
    ("audio omitido", "audio", "[nota de voz]"),
    ("video omitido", "video", "[video]"),
    ("documento omitido", "documento", "[documento]"),
    ("sticker omitido", "sticker", "[sticker]"),
    ("GIF omitido", "gif", "[gif]"),
    ("Contacto omitido", "contacto", "[contacto]"),
])
def test_cada_tipo_de_adjunto_tiene_su_etiqueta(tmp_path, cuerpo, kind, etiqueta):
    msgs = chat(tmp_path, f"[6/5/23, 10:00:00] Mica: {cuerpo}")
    assert msgs[0]["attachment"] == kind
    assert msgs[0]["text"] == etiqueta


def test_el_nombre_del_archivo_se_conserva_junto_a_la_etiqueta(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 10:00:00] Mica: 10 de julio.mp4 documento omitido")
    assert msgs[0]["attachment"] == "documento"
    assert msgs[0]["text"] == "10 de julio.mp4 [documento]"


def test_la_nota_de_voz_no_se_llama_audio_en_el_texto(tmp_path):
    """El transcript lo lee el extractor: '[nota de voz]' dice más que '[audio]'."""
    msgs = chat(tmp_path, "[6/5/23, 10:00:00] Ema Pereyra: audio omitido")
    assert msgs[0]["text"] == "[nota de voz]"


def test_un_adjunto_en_minuscula_tambien_se_detecta(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 10:00:00] Mica: gif omitido")
    assert msgs[0]["attachment"] == "gif"


def test_un_adjunto_sin_texto_igual_entra(tmp_path):
    """Sin esto se perderían los 8k audios: su cuerpo es solo el marcador."""
    msgs = chat(tmp_path, "[6/5/23, 10:00:00] Mica: imagen omitida")
    assert len(msgs) == 1


# --- ruido de sistema ------------------------------------------------------

@pytest.mark.parametrize("cuerpo", [
    "Los mensajes y las llamadas están cifrados de extremo a extremo.",
    "Las llamadas y los mensajes están cifrados.",
    "se eliminó este mensaje",
    "Eliminaste este mensaje",
    "cambió su número de teléfono",
    "<Multimedia omitido>",
])
def test_las_frases_de_sistema_no_son_conversacion(tmp_path, cuerpo):
    assert chat(tmp_path, f"[6/5/23, 10:00:00] Mica: {cuerpo}") == []


def test_el_marcador_de_editado_se_saca_del_texto(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 10:00:00] Mica: ya está <se editó este mensaje>")
    assert msgs[0]["text"] == "ya está"


def test_el_marcador_en_ingles_tambien_se_saca(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 10:00:00] Mica: ok <This message was edited>")
    assert msgs[0]["text"] == "ok"


def test_un_mensaje_que_era_solo_el_marcador_se_descarta(tmp_path):
    assert chat(tmp_path, "[6/5/23, 10:00:00] Mica: <se editó este mensaje>") == []


def test_los_invisibles_de_ios_no_quedan_en_el_texto(tmp_path):
    """U+200E delante del corchete y U+202F antes del 'p. m.' rompen el regex."""
    msgs = chat(tmp_path, "‎[6/5/23, 7:27:32 p. m.] Ema Pereyra: dale")
    assert len(msgs) == 1
    assert msgs[0]["ts"] == "2023-05-06T19:27:32"
    assert msgs[0]["text"] == "dale"


def test_el_nbsp_dentro_del_texto_se_normaliza(tmp_path):
    msgs = chat(tmp_path, "[6/5/23, 10:00:00] Mica: hola mundo")
    assert msgs[0]["text"] == "hola mundo"


# --- el diálogo que lee el extractor --------------------------------------

def test_el_dialogo_nombra_los_roles():
    texto = W._render_full_text([
        {"role": "coach", "sender": "Ema Pereyra", "text": "probá esto"},
        {"role": "alumno", "sender": "Ilan Daniele", "text": "listo"},
    ])
    assert texto == "Coach (Ema): probá esto\nAlumno (Ilan): listo"


def test_un_rol_desconocido_se_rinde_al_nombre():
    texto = W._render_full_text([{"role": "otro", "sender": "Mica", "text": "hola"}])
    assert texto == "Mica: hola"


def test_un_dialogo_vacio_es_un_texto_vacio():
    assert W._render_full_text([]) == ""


# --- la corrida completa ---------------------------------------------------

@pytest.fixture
def datos(tmp_path, monkeypatch):
    """`data_dir` en tmp, con el export ya puesto donde `run()` lo busca."""
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    wa = settings.raw_path / "whatsapp"
    wa.mkdir(parents=True, exist_ok=True)
    return {"chat": wa / "_chat.txt", "out": settings.transcripts_path / "whatsapp_ema_coaching.json"}


def test_la_corrida_escribe_el_transcript(datos):
    datos["chat"].write_text("\n".join([
        "[6/5/23, 10:00:00] Ema Pereyra: probá esto",
        "[6/5/23, 10:01:00] Ilan Daniele: listo",
        "[6/5/23, 10:02:00] Ilan Daniele: audio omitido",
    ]), encoding="utf-8")
    W.run()
    d = json.loads(datos["out"].read_text(encoding="utf-8"))
    assert d["source_id"] == "whatsapp_ema_coaching"
    assert d["source_type"] == "whatsapp"
    assert d["language"] == "es"
    assert d["metadata"]["messages"] == 3
    assert d["metadata"]["coach_messages"] == 1
    assert d["metadata"]["alumno_messages"] == 2
    assert d["metadata"]["attachments"] == {"audio": 1}
    assert d["metadata"]["first_ts"] == "2023-05-06T10:00:00"
    assert d["metadata"]["last_ts"] == "2023-05-06T10:02:00"
    assert [s["id"] for s in d["segments"]] == [0, 1, 2]
    assert "Coach (Ema): probá esto" in d["full_text"]


def test_los_ts_sin_fecha_no_tapan_el_primero(datos):
    """`first_ts` tiene que saltear los mensajes cuya fecha no se pudo leer."""
    datos["chat"].write_text("\n".join([
        "[31/2/23, 10:00:00] Mica: fecha imposible",
        "[6/5/23, 10:01:00] Mica: esta sí",
    ]), encoding="utf-8")
    W.run()
    d = json.loads(datos["out"].read_text(encoding="utf-8"))
    assert d["metadata"]["first_ts"] == "2023-05-06T10:01:00"
    assert d["metadata"]["last_ts"] == "2023-05-06T10:01:00"


def test_sin_export_no_escribe_nada(datos):
    W.run()
    assert not datos["out"].exists()


def test_no_rehace_un_transcript_que_ya_esta(datos):
    datos["chat"].write_text("[6/5/23, 10:00:00] Mica: nuevo", encoding="utf-8")
    datos["out"].parent.mkdir(parents=True, exist_ok=True)
    datos["out"].write_text('{"source_id": "viejo"}', encoding="utf-8")
    W.run()
    assert json.loads(datos["out"].read_text(encoding="utf-8"))["source_id"] == "viejo"


def test_con_force_lo_rehace(datos):
    datos["chat"].write_text("[6/5/23, 10:00:00] Mica: nuevo", encoding="utf-8")
    datos["out"].parent.mkdir(parents=True, exist_ok=True)
    datos["out"].write_text('{"source_id": "viejo"}', encoding="utf-8")
    W.run(force=True)
    assert json.loads(datos["out"].read_text(encoding="utf-8"))["source_id"] == "whatsapp_ema_coaching"


def test_los_acentos_quedan_legibles_en_el_json(datos):
    datos["chat"].write_text("[6/5/23, 10:00:00] Mica: ¿cómo andás? ñandú", encoding="utf-8")
    W.run()
    crudo = datos["out"].read_text(encoding="utf-8")
    assert "cómo andás" in crudo  # ensure_ascii=False


def test_una_ruta_fuera_del_repo_se_guarda_absoluta(tmp_path):
    assert W._relpath(tmp_path / "x.txt") == str(tmp_path / "x.txt")


def test_una_ruta_del_repo_se_guarda_relativa():
    from coach_brain.settings import REPO_ROOT
    assert W._relpath(REPO_ROOT / "data" / "raw" / "x.txt") in (
        "data\\raw\\x.txt", "data/raw/x.txt")
