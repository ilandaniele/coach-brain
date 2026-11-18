"""Tests del detector de alucinaciones de `retranscribir_sin_vad.py`.

Whisper sin filtro de voz no devuelve silencio cuando no hay habla: devuelve
frases de los subtítulos de YouTube con los que se entrenó. Las tres medidas en
este corpus son "Gracias.", "Gracias por ver el video." y "La ciudad de Bogotá"
repetida nueve veces.

El detector existe para que revisar los resultados sea leer con atención los
que pueden ser habla real, en vez de los veintidós. Por eso estos tests cargan
en los dos lados:

- Que marque lo que es basura, porque si no la marca alguien puede promoverla.
- Que **no** marque habla real, porque un falso positivo manda a descartar
  material recuperado, y el original corto se queda como está.

El segundo lado es el que importa más, y es el que tiene más casos acá: el
detector marca, no borra, así que un falso negativo lo agarra la lectura, pero
un falso positivo hace perder justamente lo que se vino a buscar.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parent.parent / "scripts" / "retranscribir_sin_vad.py"


def _cargar():
    spec = importlib.util.spec_from_file_location("retranscribir_novad_aluc", RUTA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _cargar()


# --- lo que tiene que marcar ------------------------------------------------

@pytest.mark.parametrize("texto", [
    "Gracias.",
    "gracias",
    "Gracias por ver el video.",
    "Gracias por ver el vídeo.",
    "¡Muchas gracias!",
    "Suscríbete al canal",
    "Subtítulos realizados por la comunidad de Amara.org",
    "Música",
    "[Música]",
    "Hasta la próxima.",
    "Nos vemos en el próximo video",
])
def test_frases_de_youtube_se_marcan(texto):
    """Medidas en este corpus o conocidas del corpus de entrenamiento."""
    assert M.parece_alucinacion(texto) is not None


def test_bucle_del_decoder_se_marca():
    """El caso real: "La ciudad de Bogota" nueve veces, sobre 248 segundos."""
    t = " ".join(["La ciudad de Bogotá."] * 9)
    motivo = M.parece_alucinacion(t)
    assert motivo is not None
    assert "bucle" in motivo


def test_bucle_con_una_frase_distinta_se_marca():
    """Casi todo repetido sigue siendo un bucle, no habla.

    Umbral: 80% de las frases iguales. Nueve repeticiones mas una frase suelta
    no es una conversacion.
    """
    t = " ".join(["Sí, claro."] * 9 + ["Bueno, te cuento."])
    assert M.parece_alucinacion(t) is not None


def test_vacio_se_marca():
    assert M.parece_alucinacion("") == "vacio"
    assert M.parece_alucinacion("   ") == "vacio"


# --- lo que NO tiene que marcar (el lado caro de equivocarse) ---------------

def test_habla_real_no_se_marca():
    """El caso que justifica todo el script: 8036 caracteres recuperados.

    Un fragmento del transcript que paso de 256 a 8036 caracteres al sacar el
    filtro de voz. Si el detector marcara esto, se perderia lo unico que la
    auditoria encontro.
    """
    t = ("Mirá, lo que pasó fue que yo estaba en la barra y ella se acercó a "
         "pedir un trago. Le dije algo sobre la música y se empezó a reír. "
         "Después me contó que estudiaba arquitectura y que estaba por rendir "
         "finales. Estuvimos como veinte minutos hablando de eso.")
    assert M.parece_alucinacion(t) is None


def test_gracias_dentro_de_una_frase_mas_larga_no_se_marca():
    """"Gracias" es alucinacion cuando es TODO el texto, no cuando aparece.

    Comparar por substring marcaria cualquier conversacion donde alguien
    agradece, que en notas de voz es la mitad.
    """
    t = ("Gracias por el dato, lo voy a probar el finde. Igual me parece que "
         "el lugar ese está muy lleno los sábados.")
    assert M.parece_alucinacion(t) is None


def test_una_sola_frase_corta_real_no_se_marca():
    """Sin repeticion no hay bucle, aunque sea corto."""
    assert M.parece_alucinacion("Dale, nos vemos a las ocho entonces.") is None


def test_dos_repeticiones_no_alcanzan():
    """Repetir algo dos veces es enfasis, no un bucle del decoder.

    El piso son 3 repeticiones justamente para no marcar el enfasis.
    """
    t = "No, no."
    assert M.parece_alucinacion(t) is None


def test_enfasis_repetido_dentro_de_habla_real_no_se_marca():
    """Tres "sí" seguidos dentro de una conversacion no son un bucle.

    Lo que marca el bucle es que el texto COLAPSE a esa frase, no que la
    contenga.
    """
    t = ("Sí. Sí. Sí. Pero lo que te quiero decir es que esa estrategia no "
         "funciona si la chica ya te vio dudar antes. Ahí ya perdiste el "
         "marco y no lo recuperás hablando mas fuerte.")
    assert M.parece_alucinacion(t) is None


def test_dialogo_con_frases_distintas_no_se_marca():
    """Muchas frases cortas y distintas es una conversacion, no un bucle."""
    t = "Hola. ¿Cómo andás? Todo bien. ¿Y vos? Acá andamos. Bueno, dale. Chau."
    assert M.parece_alucinacion(t) is None


# --- el veredicto -----------------------------------------------------------

def test_veredicto_descarta_la_alucinacion():
    v = M._veredicto("Gracias por ver el video.", 0)
    assert v.startswith("DESCARTAR")


def test_veredicto_descarta_lo_que_no_recupero_nada():
    """Mas corto o igual que el original no aporta, aunque sea habla real."""
    v = M._veredicto("Dale, nos vemos.", 500)
    assert v.startswith("DESCARTAR")
    assert "no recupero nada" in v


def test_veredicto_manda_a_revisar_lo_que_crecio():
    """El caso bueno: mucho mas texto y sin señal de alucinacion."""
    t = ("Mirá, lo que pasó fue que yo estaba en la barra y ella se acercó. "
         "Le dije algo sobre la música y se empezó a reír. Después me contó "
         "que estudiaba arquitectura.") * 3
    v = M._veredicto(t, 256)
    assert v.startswith("REVISAR")
    assert "+" in v


def test_veredicto_nunca_promueve_solo():
    """Ningun veredicto dice "promover": el texto lo tiene que leer alguien.

    Si este test falla es porque alguien le dio al script autoridad para meter
    texto al corpus, y eso es exactamente lo que no tiene que poder hacer.
    """
    for args in (("Gracias.", 0), ("Hola, todo bien por acá, te cuento.", 5),
                 ("", 100)):
        v = M._veredicto(*args)
        assert v.startswith(("DESCARTAR", "REVISAR")), v


# --- los cuatro que se escaparon en la corrida real -------------------------
#
# Estos cuatro aparecieron en "A REVISAR" cuando no debian. Son fixtures reales:
# el texto es el que devolvio Whisper, recortado. Los cuatro fallaban por el
# mismo motivo de fondo, que contar frases partidas por `.!?` no sirve cuando
# Whisper en bucle no pone puntuacion.

def test_bucle_sin_puntuacion_se_marca():
    """`wa_audio_00023679`: "La ciudad de Bogota" nueve veces, sin un punto.

    Para cualquier criterio que parta por `.!?` esto es UNA frase, asi que la
    primera version lo dejaba pasar. Es el caso que motivo el n-grama.
    """
    t = " ".join(["La ciudad de Bogotá"] * 9)
    motivo = M.parece_alucinacion(t)
    assert motivo is not None
    assert "bucle" in motivo


def test_bucle_de_frase_larga_sin_puntuacion_se_marca():
    """`wa_audio_00030488`: "de la serie de videos" en bucle, sin puntuacion."""
    t = ("Bienvenidos a un nuevo video de la serie de videos de la serie de "
         "videos de la serie de videos de la serie de la serie de videos de "
         "la serie de la serie de videos de la serie")
    assert M.parece_alucinacion(t) is not None


def test_frases_inventadas_mas_una_url_se_marca():
    """`wa_audio_00033832`: tres "Gracias por ver el video." y un dominio.

    Eran 3 de 4 frases, justo abajo del 80%, asi que el criterio de proporcion
    no alcanzaba. Lo que lo resuelve es preguntar que QUEDA al sacar las frases
    inventadas: nada.
    """
    t = ("Más información www.cdc.gov.ar Gracias por ver el video. "
         "Gracias por ver el video. Gracias por ver el video.")
    assert M.parece_alucinacion(t) is not None


def test_frase_inventada_repetida_dos_veces_se_marca():
    """`wa_video_00002551`: dos veces alcanza si es TODO lo que hay.

    El piso de 3 repeticiones existe para no marcar el enfasis real, pero el
    enfasis real tiene texto alrededor. Aca no hay nada alrededor.
    """
    t = "Gracias por ver el video. Gracias por ver el video."
    assert M.parece_alucinacion(t) is not None


def test_el_caso_bueno_sigue_pasando_con_las_señales_nuevas():
    """`wa_audio_00032678`: 6445 caracteres de dialogo real. NO se marca.

    Es el que justifica todo el script, y contiene justo lo que las señales
    nuevas buscan: "Muchas gracias" quince veces y frases repetidas. Pasa
    porque esas repeticiones son una fraccion chica de un texto largo, que es
    exactamente la diferencia entre una muletilla y un bucle.

    Si este test falla, las señales se volvieron demasiado anchas y la auditoria
    descarta lo unico que encontro.
    """
    t = (
        "Hola chicos Hola chicos Hola chicos "
        "¿No es tu hermano el que esta ahí mirando? "
        "No, no es mi hermano No es mi hermano No es mi hermano "
        "De donde era vos? De donde era vos? De Navarra "
        + "¡Muchas gracias! " * 15
        + "Saca las fechas. No puedo, no puedo sacar las fechas. "
        "¿Cumplió la semana que viene? Me dijiste, es verdad, me dijiste que "
        "cumplió la semana que viene, y yo te dije que no podía ir porque "
        "estaba trabajando esa noche en el bar de la esquina. "
        "Después hablamos de otra cosa, de lo que había pasado el sábado "
        "anterior cuando salimos todos juntos y terminamos en ese lugar "
        "horrible que ni me acuerdo como se llamaba. "
        "Ella me contaba que estudiaba arquitectura y que le faltaban dos "
        "materias para recibirse, y que su hermana vivia en Madrid. "
        "Yo le pregunte si pensaba irse tambien y me dijo que no, que acá "
        "tenia todo armado y que no le veia sentido empezar de nuevo. "
    ) * 2
    assert M.parece_alucinacion(t) is None, M.parece_alucinacion(t)


def test_muletilla_frecuente_en_charla_larga_no_se_marca():
    """Una muletilla repetida mucho dentro de texto largo no es bucle.

    La diferencia es la cobertura: lo que marca el bucle es que la secuencia
    ocupe casi todo el texto, no que aparezca seguido.
    """
    t = ("o sea " * 10) + (
        "lo que yo te quiero explicar es que la conversacion no se sostiene "
        "sola, hace falta que vos tengas algo para decir despues de la "
        "primera pregunta, porque si no se cae y queda ese silencio incomodo "
        "que no hay manera de levantar. Eso lo aprendi a los golpes, "
        "practicando en la calle durante meses hasta que me salio natural.")
    assert M.parece_alucinacion(t) is None
