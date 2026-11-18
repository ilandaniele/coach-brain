"""Tests de la deteccion de pista de audio y del flag de VAD.

Cubren el arreglo de los dos videos de WhatsApp que hacian caer a Whisper con
`tuple index out of range`: no tienen pista de audio, y sin la comprobacion
previa quedaban para siempre en la lista de "sin cubrir" como si fueran un
hueco del corpus.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from coach_brain.ingest import transcribe


class _Corrida:
    """Reemplaza a subprocess.run y guarda con que se lo llamo."""

    def __init__(self, stdout: str = "", excepcion: Exception | None = None) -> None:
        self.stdout = stdout
        self.excepcion = excepcion
        self.llamadas: list[list[str]] = []

    def __call__(self, cmd, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        self.llamadas.append(list(cmd))
        if self.excepcion is not None:
            raise self.excepcion
        return subprocess.CompletedProcess(cmd, 0, self.stdout, "")


def test_has_audio_stream_con_pista(monkeypatch):
    corrida = _Corrida(stdout="0\n")
    monkeypatch.setattr(transcribe.subprocess, "run", corrida)
    assert transcribe.has_audio_stream(Path("video.mp4")) is True
    assert corrida.llamadas[0][0] == "ffprobe"


def test_has_audio_stream_sin_pista(monkeypatch):
    monkeypatch.setattr(transcribe.subprocess, "run", _Corrida(stdout="\n"))
    assert transcribe.has_audio_stream(Path("video.mp4")) is False


def test_has_audio_stream_sin_ffprobe_asume_que_si(monkeypatch):
    """Si no se pudo verificar, devuelve True: que lo intente ffmpeg.

    Lo importante es que un ffprobe ausente NO haga saltear el archivo, porque
    eso seria perder un video que si tenia audio.
    """
    monkeypatch.setattr(
        transcribe.subprocess, "run", _Corrida(excepcion=OSError("no ffprobe"))
    )
    assert transcribe.has_audio_stream(Path("video.mp4")) is True


def test_has_audio_stream_timeout_asume_que_si(monkeypatch):
    monkeypatch.setattr(
        transcribe.subprocess,
        "run",
        _Corrida(excepcion=subprocess.TimeoutExpired("ffprobe", 120)),
    )
    assert transcribe.has_audio_stream(Path("video.mp4")) is True


def test_extract_audio_sin_pista_levanta_error(monkeypatch, tmp_path):
    """Sin pista de audio no se invoca ffmpeg: se corta con un error claro."""
    monkeypatch.setattr(transcribe, "has_audio_stream", lambda p: False)
    llamadas: list[list[str]] = []
    monkeypatch.setattr(
        transcribe.subprocess, "run", lambda cmd, **kw: llamadas.append(list(cmd))
    )
    with pytest.raises(ValueError, match="no tiene pista de audio"):
        transcribe.extract_audio(tmp_path / "v.mp4", tmp_path / "a.wav")
    assert llamadas == []


def test_extract_audio_con_pista_invoca_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "has_audio_stream", lambda p: True)
    corrida = _Corrida()
    monkeypatch.setattr(transcribe.subprocess, "run", corrida)
    transcribe.extract_audio(tmp_path / "v.mp4", tmp_path / "a.wav")
    cmd = corrida.llamadas[0]
    assert cmd[0] == "ffmpeg"
    for flag in ("-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le"):
        assert flag in cmd


class _Seg:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start = start
        self.end = end


class _Info:
    language = "es"
    language_probability = 0.99
    duration = 12.5


class _ModeloFalso:
    """Modelo de Whisper de mentira: registra los kwargs recibidos."""

    def __init__(self, segs: list[_Seg] | None = None) -> None:
        self.segs = segs if segs is not None else []
        self.kwargs: dict = {}

    def transcribe(self, path, **kwargs):  # noqa: ANN001, ANN003
        self.kwargs = kwargs
        return iter(self.segs), _Info()


def test_transcribe_file_pasa_el_vad(tmp_path):
    """`vad=False` tiene que llegar a model.transcribe como vad_filter=False.

    Es el flag con el que se auditan los transcripts vacios, asi que si no se
    propaga, la auditoria devuelve el mismo vacio y parece confirmarlo.
    """
    modelo = _ModeloFalso()
    transcribe.transcribe_file(
        modelo, tmp_path / "a.wav", "wa_audio", tmp_path / "a.ogg", vad=False
    )
    assert modelo.kwargs["vad_filter"] is False

    modelo2 = _ModeloFalso()
    transcribe.transcribe_file(modelo2, tmp_path / "a.wav", "wa_audio", tmp_path / "a.ogg")
    assert modelo2.kwargs["vad_filter"] is True


def test_transcribe_file_anota_el_vad_en_metadata(tmp_path):
    """El transcript tiene que decir si se hizo sin VAD: sin el filtro, Whisper
    alucina, y un texto asi no se puede creer sin leerlo."""
    data = transcribe.transcribe_file(
        _ModeloFalso(), tmp_path / "a.wav", "wa_audio", tmp_path / "a.ogg", vad=False
    )
    assert data["metadata"]["vad_filter"] is False


def test_transcribe_file_arma_el_formato_unificado(tmp_path):
    segs = [_Seg(" hola ", 0.0, 1.0), _Seg("que tal", 1.0, 2.0)]
    data = transcribe.transcribe_file(
        _ModeloFalso(segs), tmp_path / "a.wav", "wa_audio", tmp_path / "nota.ogg"
    )
    assert data["source_id"] == "wa_audio_nota"
    assert data["full_text"] == "hola que tal"
    assert [s["id"] for s in data["segments"]] == [0, 1]
    assert data["segments"][0]["text"] == "hola"
    assert data["segments"][0]["page"] is None
    assert data["metadata"]["duration_seconds"] == 12.5
    assert data["metadata"]["pages"] is None
