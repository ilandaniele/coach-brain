"""Tests del reparador de `source_path`.

Las dos piezas que importan de `scripts/reparar_source_path.py` son decisiones,
no transformaciones: `resolver` decide si un nombre roto tiene un unico destino
posible en disco, y `es_duplicado` decide si dos transcripts son el mismo
contenido. Las dos pueden fallar en silencio y de la peor manera: apuntar un
transcript al archivo equivocado, o declarar duplicado algo que no lo es y hacer
que alguien borre material unico.

Por eso estos tests fijan sobre todo los casos en los que las funciones tienen
que NEGARSE a decidir. Un nombre ambiguo no se adivina, y un arranque de texto
igual no alcanza para llamar duplicado a nada.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cargar_reparador():
    """Importa scripts/reparar_source_path.py, que no es parte del paquete."""
    spec = importlib.util.spec_from_file_location(
        "reparar_source_path_script", REPO / "scripts" / "reparar_source_path.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = _cargar_reparador()


# --- resolver: encontrar el archivo que se movio -------------------------------


def test_resuelve_cuando_hay_un_solo_candidato_por_nombre():
    """El caso normal: el archivo existe, una carpeta mas abajo.

    Es lo que paso con los 154 videos de clase: el material se reorganizo dentro
    de `Seduccion\\Bonus\\` despues de la ingesta, asi que el nombre sigue siendo
    unico y el `source_path` viejo es lo unico que quedo mal.
    """
    destino = Path("C:/m/Bonus/clase.mp4")
    idx = {"clase.mp4": [destino]}
    nueva, motivo = rep.resolver(Path("C:/m/clase.mp4"), idx)
    assert nueva == destino
    assert motivo == "nombre"


def test_se_niega_a_elegir_entre_dos_candidatos_con_el_mismo_nombre():
    """Dos archivos con el mismo nombre en carpetas distintas no se adivinan.

    Elegir uno al azar escribiria un `source_path` que apunta a contenido
    distinto del que produjo el transcript, y nada volveria a avisar del error.
    """
    idx = {"clase.mp4": [Path("C:/m/a/clase.mp4"), Path("C:/m/b/clase.mp4")]}
    nueva, motivo = rep.resolver(Path("C:/m/clase.mp4"), idx)
    assert nueva is None
    assert motivo.startswith("ambiguo")


def test_reintenta_sin_el_sufijo_que_agrega_el_pipeline():
    """El pipeline numera las fuentes con `-NNN`; el archivo en disco no.

    `78 que es el valor-008.mp4` es un nombre que solo existe dentro del
    pipeline: en disco el archivo se llama `78 que es el valor.mp4`.
    """
    destino = Path("C:/m/78 que es el valor.mp4")
    idx = {"78 que es el valor.mp4": [destino]}
    nueva, motivo = rep.resolver(Path("C:/m/78 que es el valor-008.mp4"), idx)
    assert nueva == destino
    assert motivo == "nombre sin sufijo del pipeline"


def test_no_inventa_nada_cuando_el_nombre_no_esta_en_disco():
    idx = {"otra cosa.mp4": [Path("C:/m/otra cosa.mp4")]}
    nueva, motivo = rep.resolver(Path("C:/m/no existe.mp4"), idx)
    assert nueva is None
    assert motivo == "sin candidato por nombre"


# --- es_duplicado: decidir si sobra texto, no si falta -------------------------


def test_duplicado_con_misma_duracion_y_largos_distintos():
    """Dos pasadas de Whisper sobre el mismo video: mismo audio, texto distinto.

    La duracion del contenedor es identica al microsegundo porque sale del mismo
    archivo, pero el decodificado no es determinista y los largos difieren en
    algunos miles de caracteres. Por eso la duracion sola decide.
    """
    cab = "a" * rep.CABECERA_DUPLICADO
    a = (5948.7201875, 83248, cab)
    b = (5948.7201875, 84606, cab)
    assert rep.es_duplicado(a, b)


def test_duplicado_con_mismo_largo_y_sin_duracion():
    """Dos PDF identicos con nombres distintos: no tienen duracion, pero el
    texto extraido sale caracter por caracter igual."""
    cab = "b" * rep.CABECERA_DUPLICADO
    a = (None, 56636, cab)
    b = (None, 56636, cab)
    assert rep.es_duplicado(a, b)


def test_no_es_duplicado_si_el_arranque_difiere():
    """Mismo largo y misma duracion no alcanzan si el texto arranca distinto."""
    a = (100.0, 5000, "arranque uno" + "x" * 200)
    b = (100.0, 5000, "arranque dos" + "x" * 200)
    assert not rep.es_duplicado(a, b)


def test_no_es_duplicado_por_duracion_parecida_con_largos_distintos():
    """El caso que casi hizo sobre-reportar duplicados.

    Agrupando por duracion redondeada aparecen 159 grupos, y casi todos son
    notas de voz cortas cuyas duraciones caen cerca por casualidad. Sin arranque
    identico no hay duplicado, y con duraciones separadas por mas de la
    tolerancia tampoco.
    """
    cab = "c" * rep.CABECERA_DUPLICADO
    a = (28.0, 300, cab)
    b = (29.0, 310, cab)
    assert not rep.es_duplicado(a, b)


def test_no_es_duplicado_si_no_hay_texto():
    """Dos transcripts vacios no son el mismo contenido: no hay contenido."""
    a = (30.0, 0, "")
    b = (30.0, 0, "")
    assert not rep.es_duplicado(a, b)
