"""Tests de `ingest/_guard.py`: la guarda contra pisar un transcript con uno más chico.

El caso que la motivó es real y está en disco. `whatsapp_images.py` (vision con
Haiku) y `whatsapp_images_ocr.py` (OCR local) escriben el mismo archivo,
`transcripts/whatsapp_images_chats.json`, con el mismo `source_id`. Eso es a
propósito: el resto del pipeline no tiene que saber con qué motor se sacó el
texto.

Lo que no es a propósito es el estado en el que quedaron las dos corridas. La de
vision se abandonó a medias —4074 imágenes de 8978, 2873 chats— porque fallaba
y encima cuesta plata por imagen. La de OCR terminó entera: 8953 imágenes, 6336
chats, 2,6 MB de texto. Un `python -m coach_brain.ingest.whatsapp_images
aggregate` tecleado por costumbre pisaba el transcript bueno con el parcial, sin
un error y sin backup, y después `kb.extract` lo re-extraía y `kb.index --reset`
lo publicaba: la pérdida llegaba hasta el índice que sirve las respuestas.

`write_transcript` corta eso. Lo que se fija acá:

- **Se mide en segmentos, no en bytes.** Es la unidad que se vuelve chunks y
  después vectores. Un transcript con menos segmentos aporta menos al índice,
  por más que su JSON pese más por el formato.
- **Solo bloquea el achique.** Igual o más grande escribe: el pipeline tiene que
  poder re-agregar con material nuevo sin flags.
- **`force=True` escribe igual.** Si la decisión es a propósito, no hay que
  pelear con la herramienta.
- **Lo ilegible no bloquea.** Un JSON truncado en el destino no tiene segmentos
  que perder; tratarlo como valioso dejaría el pipeline trabado sin nada que
  proteger.
"""

from __future__ import annotations

import json

import pytest

from coach_brain.ingest._guard import write_transcript


def transcript(n: int, marca: str = "x") -> dict:
    return {
        "source_id": "whatsapp_images_chats",
        "segments": [{"id": i, "text": f"{marca} {i}"} for i in range(n)],
        "full_text": f"{marca} " * n,
    }


def escrito(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- el caso que la motivó -------------------------------------------------


def test_el_parcial_no_pisa_al_completo(tmp_path, capsys):
    """Los números son los de disco: 6336 chats del OCR contra 2873 de la
    corrida de vision que quedó a medias."""
    destino = tmp_path / "whatsapp_images_chats.json"
    destino.write_text(json.dumps(transcript(6336, "ocr")), encoding="utf-8")

    assert write_transcript(destino, transcript(2873, "vision")) is False
    assert escrito(destino)["segments"][0]["text"] == "ocr 0"
    assert len(escrito(destino)["segments"]) == 6336


def test_cuando_no_escribe_dice_los_dos_numeros(tmp_path, capsys):
    """El aviso tiene que alcanzar para decidir sin abrir los archivos: cuántos
    hay, cuántos iban a quedar y cuántos se habrían perdido."""
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps(transcript(6336)), encoding="utf-8")

    write_transcript(destino, transcript(2873))

    salida = capsys.readouterr().out
    assert "6336" in salida
    assert "2873" in salida
    assert "3463" in salida  # lo que se habria perdido
    assert "force" in salida


# --- qué bloquea y qué no --------------------------------------------------


def test_uno_mas_grande_escribe_sin_flags(tmp_path):
    """Re-agregar con material nuevo es la operación normal del pipeline."""
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps(transcript(10, "viejo")), encoding="utf-8")

    assert write_transcript(destino, transcript(20, "nuevo")) is True
    assert len(escrito(destino)["segments"]) == 20


def test_uno_del_mismo_tamano_escribe(tmp_path):
    """Mismo tamaño con contenido distinto es una re-corrida del mismo material,
    no una pérdida. Bloquearla obligaría a `force` en el caso más común."""
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps(transcript(5, "viejo")), encoding="utf-8")

    assert write_transcript(destino, transcript(5, "nuevo")) is True
    assert escrito(destino)["segments"][0]["text"] == "nuevo 0"


def test_un_destino_que_no_existe_se_escribe(tmp_path):
    destino = tmp_path / "t.json"
    assert write_transcript(destino, transcript(3)) is True
    assert len(escrito(destino)["segments"]) == 3


def test_perder_un_solo_segmento_ya_bloquea(tmp_path):
    """El borde: 6336 contra 6335. No hay umbral de tolerancia, porque un
    segmento es un chat entero de alguien."""
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps(transcript(6336)), encoding="utf-8")
    assert write_transcript(destino, transcript(6335)) is False


# --- force ----------------------------------------------------------------


def test_con_force_se_pisa_igual(tmp_path):
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps(transcript(6336, "ocr")), encoding="utf-8")

    assert write_transcript(destino, transcript(2, "vision"), force=True) is True
    assert len(escrito(destino)["segments"]) == 2


def test_con_force_no_se_lee_el_destino(tmp_path, monkeypatch):
    """Con `force` no hace falta abrir el archivo viejo, así que un destino
    ilegible o enorme no puede hacer fallar la escritura."""
    destino = tmp_path / "t.json"
    destino.write_text("{esto no es json", encoding="utf-8")
    assert write_transcript(destino, transcript(1), force=True) is True
    assert len(escrito(destino)["segments"]) == 1


# --- los casos raros del destino -------------------------------------------


def test_un_destino_ilegible_no_bloquea(tmp_path):
    """Un JSON truncado no tiene segmentos que perder. Tratarlo como valioso
    dejaría el pipeline trabado protegiendo basura."""
    destino = tmp_path / "t.json"
    destino.write_text("{truncado", encoding="utf-8")
    assert write_transcript(destino, transcript(1)) is True


def test_un_destino_sin_segmentos_no_bloquea(tmp_path):
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps({"source_id": "x"}), encoding="utf-8")
    assert write_transcript(destino, transcript(1)) is True


def test_un_destino_con_segments_en_null_no_bloquea(tmp_path):
    """`segments: null` es lo que deja un agregado que no encontró nada."""
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps({"segments": None}), encoding="utf-8")
    assert write_transcript(destino, transcript(1)) is True


def test_escribir_nada_sobre_algo_tambien_se_bloquea(tmp_path):
    """El peor caso posible: un agregado vacío contra el corpus entero."""
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps(transcript(6336)), encoding="utf-8")
    assert write_transcript(destino, {"segments": []}) is False


# --- el formato de lo que escribe ------------------------------------------


def test_los_acentos_se_guardan_como_acentos(tmp_path):
    """`ensure_ascii=False`: el texto es español y el transcript se lee a mano."""
    destino = tmp_path / "t.json"
    write_transcript(destino, {"segments": [{"text": "qué decís"}]})
    crudo = destino.read_text(encoding="utf-8")
    assert "qué decís" in crudo
    assert "\\u" not in crudo


def test_se_guarda_indentado(tmp_path):
    """Los transcripts se abren a mano y se diffean entre corridas."""
    destino = tmp_path / "t.json"
    write_transcript(destino, transcript(2))
    assert "\n  " in destino.read_text(encoding="utf-8")


@pytest.mark.parametrize("n", [0, 1, 2])
def test_el_resultado_dice_si_escribio(tmp_path, n):
    """El booleano es lo que usan los dos `aggregate()` para no imprimir un OK
    sobre una escritura que no pasó."""
    destino = tmp_path / "t.json"
    destino.write_text(json.dumps(transcript(1)), encoding="utf-8")
    assert write_transcript(destino, transcript(n)) is (n >= 1)
