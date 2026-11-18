"""Tests de los dos arreglos que salieron de leer el grupo "A REVISAR".

Los dos fallos son reales, encontrados en los 21 resultados de la primera
corrida:

1. `_resto_sin_frases_inventadas` solo sacaba una frase cuando la frase entera
   era una alucinación conocida. Whisper muchas veces las escupe sin
   puntuación, así que se fusionan con lo de al lado y dejan de ser una frase
   entera: en `wa_audio_00029933` la frase `"paloma dahi paloma de tennis
   gracias por ver el video"` contó sus 53 caracteres como habla real.

2. Hay resultados que son mitad habla y mitad bucle del decoder. El veredicto
   los manda a revisar, que es lo correcto, pero no decía dónde mirar. En
   `wa_audio_00029933`, `"¡Vaya que se cae!"` repetido 9 veces cubre el 49% del
   texto: por debajo del 60% que hace falta para descartar.

Los avisos NO cambian el veredicto. Lo dice el test
`test_los_avisos_no_cambian_el_veredicto`: si algún día lo cambiaran, el script
estaría descartando trabajo de GPU por su cuenta.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cargar():
    spec = importlib.util.spec_from_file_location(
        "novad_avisos", REPO / "scripts" / "retranscribir_sin_vad.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _cargar()


# Los dos resultados reales que motivan estos tests. Van textuales porque las
# proporciones importan: al acortar el de 29933 el bucle pasa a cubrir el 74%
# del texto y el detector lo descarta, que es justo lo contrario del caso que
# hay que fijar.
_CASO_29933 = (
    "Mirá... ¿Qué pasa con esta arma ahí boludo? ¿Te pones frente y hablás de "
    "ahí? ¡Jajajajaja! ¡Vaya guay boludo! "
    + "¡Vaya que se cae! " * 9
    + "¡Ojo! Mirá la pelota! Paloma de Expansión ¡Eh! Paloma dahi Paloma de "
      "tennis Gracias por ver el video"
)

_CASO_33841 = (
    "Tengo que ir a otro Vamos ¿Ves? Chica, ¿cómo va? ¿Tú bien? Es muy loco "
    "esto Pero te vi, me pensé que tenías muy buena energía Y digo, la, la, la "
    "Ah, bueno, no pasa nada Estamos aquí"
)


# --- 1) las frases inventadas se sacan de adentro de una frase más larga ----


def test_la_frase_inventada_pegada_a_otro_texto_se_resta():
    """El caso exacto de `wa_audio_00029933`, sin el punto que la separaría."""
    t = "paloma dahi paloma de tennis gracias por ver el video"
    assert M._resto_sin_frases_inventadas(t) < 0.6


def test_antes_contaba_entera():
    """Guardia contra volver a comparar la frase completa.

    Si alguien reescribe la función con `if f in ALUCINACIONES`, este texto
    vuelve a conservar el 100% y el test falla.
    """
    t = "paloma dahi gracias por ver el video"
    assert M._resto_sin_frases_inventadas(t) < 1.0


def test_el_limite_de_palabra_no_es_decorativo():
    """`fin` está en la lista. Sin `\\b` saldría de media docena de palabras.

    Pasó de verdad en este corpus: "definitivamente" aparece en el habla real.
    """
    t = "definitivamente afinar el final"
    assert M._resto_sin_frases_inventadas(t) == 1.0


def test_fin_suelto_si_se_saca():
    """Pero `fin` solo, como palabra, es una de las que Whisper inventa."""
    assert M._INVENTADAS.sub("", "y fin").strip() == "y"


def test_el_caso_bueno_sobrevive_al_cambio():
    """El transcript de 6445 caracteres trae "Muchas gracias" quince veces.

    Sacarlas como subcadena le baja el resto de 0.96 a 0.87, que sigue muy por
    encima de RESTO_MINIMO. Si este test falla, el cambio empezó a descartar el
    único resultado que recuperó diálogo real.
    """
    real = ("Hola chicos. No es mi hermano. De donde era vos? De Navarra. "
            "¡Muchas gracias! " * 15)
    real += ("Saca las fechas. No puedo sacar las fechas. Cumplio la semana "
             "que viene, me dijiste, es verdad. ") * 20
    assert M._resto_sin_frases_inventadas(real) > M.RESTO_MINIMO
    assert M.parece_alucinacion(real) is None


# --- 2) los avisos ----------------------------------------------------------


def test_el_bucle_que_no_alcanza_para_descartar_se_avisa():
    """`wa_audio_00029933`: 9 repeticiones que cubren el 49%."""
    t = _CASO_29933
    assert M.parece_alucinacion(t) is None, "no tiene que alcanzar para descartar"
    a = M.avisos(t)
    assert any("vaya que se cae" in x for x in a), a


def test_el_texto_mitad_inventado_se_avisa():
    t = "che mira esto gracias por ver el video"
    assert any("Whisper inventa" in x for x in M.avisos(t))


def test_el_habla_real_no_junta_avisos():
    """Los tres resultados del grupo A REVISAR que parecen habla de verdad.

    Si empiezan a juntar avisos, el aviso deja de señalar nada.
    """
    assert M.avisos(_CASO_33841) == []


def test_texto_vacio_no_avisa():
    assert M.avisos("") == []
    assert M.avisos(None) == []


def test_los_avisos_no_cambian_el_veredicto():
    """La propiedad que hace que los avisos sean seguros de agregar.

    Un aviso de más cuesta una línea en el reporte. Un veredicto de más
    descarta trabajo de GPU sin que nadie lo haya leído.
    """
    assert M.avisos(_CASO_29933), "el texto tiene que dar avisos para que el test sirva"
    assert M._veredicto(_CASO_29933, 108).startswith("REVISAR")


# --- 3) el denominador se mide sobre las mismas frases que el numerador -----


def test_el_texto_largo_y_limpio_no_pierde_por_la_puntuacion():
    """El PDF de 95941 caracteres daba 11% "inventado" con 30 caracteres de lista.

    `total` se medía sobre el texto entero y `queda` sobre las frases ya
    partidas, así que cada punto y cada salto de línea que se come el split
    contaba como texto borrado. En un texto con mucha puntuación eso solo
    depende del largo: no dice nada de alucinaciones.
    """
    limpio = ". ".join("La frase numero %d no tiene nada raro adentro" % i
                       for i in range(200)) + "."
    assert M._resto_sin_frases_inventadas(limpio) > 0.99
    # Solo el aviso de frases inventadas. Este texto sintetico repite ocho
    # palabras en cada frase, asi que el aviso de cobertura salta con razon:
    # eso es un bucle, y no es lo que mide este test.
    assert not any("Whisper inventa" in a for a in M.avisos(limpio))


def test_lo_que_mide_ahora_es_solo_lo_que_se_saca():
    """Media frase de lista en una frase, con el resto limpio.

    "gracias por ver el video" son 24 de los 48 caracteres de la frase, así
    que el resto tiene que dar cerca de la mitad y no depender del largo del
    texto que la rodea.
    """
    una = "che mira esto gracias por ver el video"
    diez = (una + ". ") * 10
    assert abs(M._resto_sin_frases_inventadas(una)
               - M._resto_sin_frases_inventadas(diez)) < 0.02
