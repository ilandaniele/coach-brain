"""Tests del barrido y de la corrida del transcriptor, con un Whisper falso.

Lo que ya estaba probado de `ingest/transcribe.py` es la parte chica: la
detección de pista de audio, la extracción con ffmpeg y el formato que arma
`transcribe_file`. Lo que faltaba es lo que decide *qué* se transcribe y lo que
pasa cuando algo sale mal a mitad de una corrida de días, que es justamente
donde este módulo puede hacer daño silencioso:

1. **El orden del barrido no es alfabético, es por prioridad.** Primero las
   clases (videos y audios) y después el media de WhatsApp, porque las 8 mil
   notas de voz de la coach tardan días y las clases son el material que
   importa. Si alguien cambia el orden de la lista, la corrida sigue andando y
   nadie se entera hasta que a los tres días todavía no hay ni una clase.
2. **La corrida es resumible por existencia del JSON.** Es el mismo mecanismo
   que en la extracción (gotcha #13) y la razón por la que un transcript vacío
   no se reintenta nunca solo.
3. **`--limit` cuenta archivos nuevos, no archivos vistos.** Si contara los
   salteados, el segundo batch no haría nada: terminaría el tope antes de
   llegar a lo que falta.
4. **Un archivo que explota no puede cortar la corrida**, y el WAV temporal de
   los videos tiene que borrarse igual, porque son cientos de archivos de
   varios cientos de MB cada uno sobre el mismo disco.

El modelo es un doble: cargar faster-whisper de verdad necesita GPU y las DLLs
de CUDA, así que se inyecta un módulo falso en `sys.modules`.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from coach_brain.ingest import transcribe as T
from coach_brain.settings import REPO_ROOT, settings

# --- 1) rutas relativas al repo ---------------------------------------------


def test_relpath_deja_las_rutas_de_adentro_del_repo_relativas():
    """El JSON guarda `source_path` relativo para que el corpus siga siendo
    verificable si el repo se mueve de carpeta."""
    p = REPO_ROOT / "data" / "raw" / "videos" / "clase.mp4"
    assert T._relpath(p) == str(Path("data/raw/videos/clase.mp4"))


def test_relpath_deja_absolutas_las_de_afuera(tmp_path):
    """Las junctions de `data/raw` apuntan afuera del repo (al disco con los
    360 GB). Esas no se pueden relativizar y tienen que quedar absolutas."""
    p = tmp_path / "Seduccion" / "clase.mp4"
    assert T._relpath(p) == str(p)


# --- 2) el parche de DLLs de CUDA -------------------------------------------


def test_fuera_de_windows_el_parche_de_dlls_no_hace_nada(monkeypatch):
    """En Linux (CI, Fly.io) no hay DLLs que copiar. Tiene que salir antes de
    tocar el disco, no fallar."""
    monkeypatch.setattr(T.sys, "platform", "linux")
    llamadas = []
    monkeypatch.setattr(T.sysconfig, "get_paths", lambda: llamadas.append(1) or {})
    T._register_cuda_dlls()
    assert llamadas == []


def test_sin_las_carpetas_esperadas_el_parche_no_falla(monkeypatch, tmp_path):
    """Si nvidia-cublas-cu12 o ctranslate2 no están instalados, el parche se
    saltea. Un ImportError acá dejaría sin transcriptor a una máquina que
    igual podría correr en CPU."""
    monkeypatch.setattr(T.sys, "platform", "win32")
    monkeypatch.setattr(T.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    T._register_cuda_dlls()  # no levanta


@pytest.fixture
def site_packages(tmp_path, monkeypatch):
    """Un site-packages de juguete con las dos carpetas que el parche espera."""
    monkeypatch.setattr(T.sys, "platform", "win32")
    purelib = tmp_path / "site-packages"
    cublas = purelib / "nvidia" / "cublas" / "bin"
    ct2 = purelib / "ctranslate2"
    cublas.mkdir(parents=True)
    ct2.mkdir(parents=True)
    monkeypatch.setattr(T.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})
    return {"cublas": cublas, "ct2": ct2}


def test_las_dos_dlls_de_cublas_se_copian_al_lado_de_ctranslate2(site_packages):
    """Es el parche entero: CTranslate2 carga esas DLLs por nombre con un
    LoadLibrary que ignora `os.add_dll_directory`, así que la única forma de que
    las encuentre es tenerlas en su propio directorio. Sin esto, la GPU no
    transcribe nada en esta máquina."""
    for name in ("cublas64_12.dll", "cublasLt64_12.dll"):
        (site_packages["cublas"] / name).write_bytes(b"dll " + name.encode())

    T._register_cuda_dlls()

    for name in ("cublas64_12.dll", "cublasLt64_12.dll"):
        copiada = site_packages["ct2"] / name
        assert copiada.exists(), f"falta {name}"
        assert copiada.read_bytes() == b"dll " + name.encode()


def test_una_dll_que_ya_esta_no_se_vuelve_a_copiar(site_packages):
    """`not dst.exists()`: la copia es de 100 MB y corre en cada arranque del
    transcriptor. Pisarla además rompería una DLL que CTranslate2 ya tenga
    abierta en otro proceso."""
    (site_packages["cublas"] / "cublas64_12.dll").write_bytes(b"nueva")
    ya = site_packages["ct2"] / "cublas64_12.dll"
    ya.write_bytes(b"la que ya estaba")

    T._register_cuda_dlls()

    assert ya.read_bytes() == b"la que ya estaba"


def test_si_falta_una_dll_en_el_origen_se_copia_la_otra(site_packages):
    """`nvidia-cublas-cu12` podría cambiar de contenido entre versiones. Faltar
    una no tiene que impedir que la otra llegue ni levantar un error."""
    (site_packages["cublas"] / "cublasLt64_12.dll").write_bytes(b"lt")

    T._register_cuda_dlls()

    assert (site_packages["ct2"] / "cublasLt64_12.dll").exists()
    assert not (site_packages["ct2"] / "cublas64_12.dll").exists()


def test_cudnn_no_se_copia_nunca(site_packages):
    """La aporta torch al importarse. Copiar la de CUDA 12 acá chocaría con la
    de CUDA 13 que torch ya cargó, y eso es un crash del proceso, no un
    fallback a CPU."""
    (site_packages["cublas"] / "cublas64_12.dll").write_bytes(b"x")
    (site_packages["cublas"] / "cudnn64_9.dll").write_bytes(b"no me copies")

    T._register_cuda_dlls()

    assert not (site_packages["ct2"] / "cudnn64_9.dll").exists()
    assert (site_packages["ct2"] / "cublas64_12.dll").exists(), "cublas si"


def test_sin_ctranslate2_no_se_copia_nada(tmp_path, monkeypatch):
    """Instalación en CPU: está nvidia-cublas-cu12 por alguna otra dependencia
    pero no ctranslate2. No hay destino al que copiar."""
    monkeypatch.setattr(T.sys, "platform", "win32")
    purelib = tmp_path / "site-packages"
    cublas = purelib / "nvidia" / "cublas" / "bin"
    cublas.mkdir(parents=True)
    (cublas / "cublas64_12.dll").write_bytes(b"x")
    monkeypatch.setattr(T.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})

    T._register_cuda_dlls()  # no levanta

    assert not (purelib / "ctranslate2").exists()


# --- 3) el barrido ----------------------------------------------------------


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """Un `data/` de juguete con las cuatro carpetas que mira el barrido."""
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    for sub in ("videos", "audios", "whatsapp"):
        (tmp_path / "raw" / sub).mkdir(parents=True)
    return tmp_path / "raw"


def poner(base: Path, nombre: str) -> Path:
    p = base / nombre
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")
    return p


def test_las_clases_van_antes_que_el_media_de_whatsapp(raw):
    """El orden es prioridad, no alfabético: las 8 mil notas de voz tardan días
    y las clases son el material que importa."""
    poner(raw / "whatsapp", "aaa.opus")
    poner(raw / "whatsapp", "aaa.mp4")
    poner(raw / "audios", "zzz.mp3")
    poner(raw / "videos", "zzz.mp4")

    assert [(p.name, t) for p, t in T._collect_targets()] == [
        ("zzz.mp4", "video"),
        ("zzz.mp3", "audio"),
        ("aaa.mp4", "wa_video"),
        ("aaa.opus", "wa_audio"),
    ]


def test_dentro_de_cada_grupo_ordena_por_nombre(raw):
    """Que el orden sea estable es lo que hace que una corrida interrumpida y
    reanudada no vuelva a barajar el material."""
    for n in ("c.mp3", "a.mp3", "b.mp3"):
        poner(raw / "audios", n)
    assert [p.name for p, _ in T._collect_targets()] == ["a.mp3", "b.mp3", "c.mp3"]


def test_busca_en_subcarpetas(raw):
    """El material está en carpetas por curso, varios niveles abajo."""
    poner(raw / "videos", "Curso 1/Modulo 2/clase.mp4")
    assert [p.name for p, _ in T._collect_targets()] == ["clase.mp4"]


def test_ignora_lo_que_no_es_audio_ni_video(raw):
    poner(raw / "videos", "apuntes.pdf")
    poner(raw / "videos", "clase.mp4")
    assert [p.name for p, _ in T._collect_targets()] == ["clase.mp4"]


def test_una_carpeta_que_no_existe_no_rompe_el_barrido(raw):
    """`data/raw/videos` es una junction al disco externo. Si no está montado
    el barrido tiene que seguir con el resto, no cortar."""
    (raw / "videos").rmdir()
    poner(raw / "audios", "algo.mp3")
    assert [p.name for p, _ in T._collect_targets()] == ["algo.mp3"]


def test_sin_nada_en_raw_devuelve_lista_vacia(raw):
    assert T._collect_targets() == []


# --- 4) la corrida ----------------------------------------------------------


class _Modelo:
    """El doble de WhisperModel: registra con qué se lo construyó."""

    def __init__(self, *a, **kw) -> None:
        self.args = a
        self.kwargs = kw


@pytest.fixture
def corrida(raw, monkeypatch):
    """Prepara `run()` con un faster-whisper falso y un `transcribe_file` espía.

    Devuelve la lista de llamadas: `(nombre del archivo que recibió Whisper,
    source_type, nombre del archivo de origen, vad)`.
    """
    monkeypatch.setattr(T, "_register_cuda_dlls", lambda: None)
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _Modelo
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(T, "has_audio_stream", lambda _p: True)

    vistas: list[tuple[str, str, str, bool]] = []

    def fake_transcribe(model, audio_path, source_type, source_path, vad=True):
        vistas.append((Path(audio_path).name, source_type, Path(source_path).name, vad))
        return {
            "source_id": f"{source_type}_{Path(source_path).stem}",
            "segments": [{"id": 0, "text": "hola"}],
            "metadata": {"duration_seconds": 1.0, "vad_filter": vad},
            "full_text": "hola",
        }

    monkeypatch.setattr(T, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(T, "extract_audio", lambda _v, a: Path(a).write_bytes(b"\x00"))
    return vistas


def test_el_nombre_del_json_es_tipo_mas_stem(corrida, raw):
    """`wa_audio_0001.json`, no `0001.json`: el tipo adelante es lo que evita
    que una nota de voz y un video con el mismo nombre se pisen."""
    poner(raw / "whatsapp", "0001.opus")
    T.run()
    assert [p.name for p in settings.transcripts_path.iterdir()] == ["wa_audio_0001.json"]


def test_el_json_se_escribe_con_el_contenido_que_devolvio_whisper(corrida, raw):
    poner(raw / "audios", "clase.mp3")
    T.run()
    d = json.loads((settings.transcripts_path / "audio_clase.json").read_text(encoding="utf-8"))
    assert d["full_text"] == "hola"
    assert d["metadata"]["vad_filter"] is True


def test_un_json_que_ya_existe_se_saltea(corrida, raw):
    """Resumable por existencia del archivo, igual que la extracción (gotcha
    #13). Es lo que permite cortar y seguir una corrida de días."""
    poner(raw / "audios", "hecho.mp3")
    settings.transcripts_path.mkdir(parents=True)
    (settings.transcripts_path / "audio_hecho.json").write_text("{}", encoding="utf-8")
    T.run()
    assert corrida == []


def test_force_rehace_lo_que_ya_estaba(corrida, raw):
    poner(raw / "audios", "hecho.mp3")
    settings.transcripts_path.mkdir(parents=True)
    (settings.transcripts_path / "audio_hecho.json").write_text("{}", encoding="utf-8")
    T.run(force=True)
    assert [v[2] for v in corrida] == ["hecho.mp3"]


def test_el_limite_cuenta_archivos_nuevos_no_vistos(corrida, raw):
    """La sutileza que hace que los batches sirvan de algo.

    Si el tope contara también los salteados, el segundo batch terminaría antes
    de llegar a lo que falta y la corrida no avanzaría nunca.
    """
    for n in ("a.mp3", "b.mp3", "c.mp3", "d.mp3"):
        poner(raw / "audios", n)
    settings.transcripts_path.mkdir(parents=True)
    for n in ("a", "b"):
        (settings.transcripts_path / f"audio_{n}.json").write_text("{}", encoding="utf-8")

    T.run(limit=2)
    assert [v[2] for v in corrida] == ["c.mp3", "d.mp3"]


def test_el_limite_corta_de_verdad(corrida, raw):
    for n in ("a.mp3", "b.mp3", "c.mp3"):
        poner(raw / "audios", n)
    T.run(limit=2)
    assert [v[2] for v in corrida] == ["a.mp3", "b.mp3"]


def test_un_video_pasa_por_la_extraccion_de_audio(corrida, raw):
    """Whisper recibe el WAV temporal, pero el `source_path` del JSON tiene que
    seguir siendo el video: es la ruta que verifica el corpus."""
    poner(raw / "videos", "clase.mp4")
    T.run()
    audio, tipo, origen, _ = corrida[0]
    assert audio == "_cb_clase.wav"
    assert (tipo, origen) == ("video", "clase.mp4")


def test_el_wav_temporal_se_borra_aunque_whisper_falle(corrida, raw, monkeypatch):
    """Son cientos de archivos de varios cientos de MB sobre el mismo disco: un
    WAV que queda es espacio que no vuelve."""
    poner(raw / "videos", "clase.mp4")
    tmp = Path(T.tempfile.gettempdir()) / "_cb_clase.wav"

    def explota(*a, **kw):
        raise RuntimeError("se cayo Whisper")

    monkeypatch.setattr(T, "transcribe_file", explota)
    T.run()
    assert not tmp.exists()
    assert not (settings.transcripts_path / "video_clase.json").exists()


def test_un_audio_va_directo_sin_wav_temporal(corrida, raw):
    poner(raw / "audios", "clase.mp3")
    T.run()
    assert corrida[0][0] == "clase.mp3"


def test_un_video_sin_pista_de_audio_se_saltea_sin_json(corrida, raw, monkeypatch):
    """No es un hueco del corpus: no hay nada que transcribir. Pero tampoco
    puede quedar un JSON vacío, porque entonces nunca se reintentaría."""
    poner(raw / "videos", "mudo.mp4")
    monkeypatch.setattr(T, "has_audio_stream", lambda _p: False)
    T.run()
    assert corrida == []
    assert not settings.transcripts_path.exists() or \
        list(settings.transcripts_path.iterdir()) == []


def test_un_archivo_que_explota_no_corta_la_corrida(corrida, raw, monkeypatch):
    """Una corrida son días. Un archivo roto tiene que quedar sin JSON (para que
    la próxima pasada lo reintente) y no frenar a los que siguen."""
    poner(raw / "audios", "a_roto.mp3")
    poner(raw / "audios", "b_sano.mp3")
    original = T.transcribe_file

    def falla_el_primero(model, audio_path, source_type, source_path, vad=True):
        if Path(source_path).stem == "a_roto":
            raise RuntimeError("se cayo")
        return original(model, audio_path, source_type, source_path, vad=vad)

    monkeypatch.setattr(T, "transcribe_file", falla_el_primero)
    T.run()
    assert not (settings.transcripts_path / "audio_a_roto.json").exists()
    assert (settings.transcripts_path / "audio_b_sano.json").exists()


def test_sin_vad_llega_hasta_whisper(corrida, raw):
    """El flag de auditoría. Sin VAD, Whisper alucina sobre música y ruido
    (gotcha #22), así que tiene que quedar anotado en el JSON — y para eso
    primero tiene que llegar."""
    poner(raw / "audios", "vacio.mp3")
    T.run(sin_vad=True)
    assert corrida[0][3] is False
    d = json.loads((settings.transcripts_path / "audio_vacio.json").read_text(encoding="utf-8"))
    assert d["metadata"]["vad_filter"] is False


def test_sin_material_no_carga_el_modelo(raw, monkeypatch):
    """Cargar Whisper large-v3 son varios GB a la VRAM. Sin nada que transcribir
    no tiene que llegar ni a importar faster-whisper."""
    monkeypatch.setattr(T, "_register_cuda_dlls", lambda: None)
    cargas = []

    class _Explota:
        def __init__(self, *a, **kw):
            cargas.append(kw)
            raise AssertionError("no tendria que cargar el modelo")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _Explota
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    T.run()
    assert cargas == []


def test_el_modelo_se_construye_con_lo_que_dice_settings(corrida, raw, monkeypatch):
    """El device y el compute_type son lo que decide si corre en GPU o se
    arrastra en CPU, y vienen de settings, no hardcodeados."""
    poner(raw / "audios", "clase.mp3")
    monkeypatch.setattr(settings, "whisper_model", "large-v3")
    monkeypatch.setattr(settings, "whisper_device", "cuda")
    monkeypatch.setattr(settings, "whisper_compute_type", "float16")

    vistos: list[_Modelo] = []
    fake = sys.modules["faster_whisper"]

    def espia(*a, **kw):
        m = _Modelo(*a, **kw)
        vistos.append(m)
        return m

    monkeypatch.setattr(fake, "WhisperModel", espia)
    T.run()
    assert vistos[0].args == ("large-v3",)
    assert vistos[0].kwargs == {"device": "cuda", "compute_type": "float16"}
