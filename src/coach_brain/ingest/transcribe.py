"""Transcriptor de audios/videos con faster-whisper GPU."""

from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

from coach_brain.settings import REPO_ROOT, settings

console = Console()

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _register_cuda_dlls() -> None:
    """Hace que CTranslate2 encuentre las DLLs de CUDA 12 en Windows.

    Problema (RTX Blackwell sm_120): torch usa CUDA 13 (cublas64_13) pero
    CTranslate2 4.x necesita CUDA 12 (cublas64_12 + cublasLt64_12), provistas
    por nvidia-cublas-cu12 en site-packages/nvidia/cublas/bin. Como CTranslate2
    carga esas DLLs por nombre con un LoadLibrary que ignora os.add_dll_directory,
    la única forma fiable es tenerlas junto a ctranslate2.dll (Windows siempre
    busca primero en el directorio del módulo que las carga). cudnn64_9 la aporta
    torch al importarse, así que NO se copia (evita chocar con la de CUDA 13).
    """
    if sys.platform != "win32":
        return
    purelib = Path(sysconfig.get_paths()["purelib"])
    cublas_bin = purelib / "nvidia" / "cublas" / "bin"
    ct2_dir = purelib / "ctranslate2"
    if not (cublas_bin.exists() and ct2_dir.exists()):
        return
    import shutil

    for name in ("cublas64_12.dll", "cublasLt64_12.dll"):
        src = cublas_bin / name
        dst = ct2_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def has_audio_stream(path: Path) -> bool:
    """True si el contenedor declara al menos una pista de audio.

    Hay videos de WhatsApp grabados con el micrófono apagado: no tienen pista
    de audio, y sin esta comprobación ffmpeg falla con un error que no dice
    nada y el archivo queda para siempre en la lista de "sin cubrir" como si
    fuera un hueco del corpus. No lo es: no hay nada que transcribir.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return True  # No se pudo verificar: que lo intente ffmpeg.
    return bool((r.stdout or "").strip())


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extrae audio a WAV 16kHz mono — formato preferido por Whisper."""
    if not has_audio_stream(video_path):
        raise ValueError("el video no tiene pista de audio: no hay nada que transcribir")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )


def transcribe_file(
    model, audio_path: Path, source_type: str, source_path: Path, vad: bool = True
) -> dict:
    """Transcribe un archivo y devuelve dict con el formato unificado.

    `vad=False` desactiva el filtro de voz. Sirve para auditar: el VAD decide
    que no hay habla y devuelve 0 segmentos, y como la etapa es resumable por
    existencia del JSON, ese vacío nunca se reintenta. Correr sin VAD muestra
    qué había realmente en el audio. Ojo: sin VAD, Whisper alucina sobre música
    y ruido — "Gracias por ver el video" o "¡Suscríbete!" repetidos son basura
    de su corpus de entrenamiento, no contenido. Hay que leer el resultado
    antes de confiar en él.
    """
    segments, info = model.transcribe(
        str(audio_path),
        language=settings.whisper_language,
        vad_filter=vad,
        word_timestamps=False,
    )

    seg_list: list[dict] = []
    full_text_parts: list[str] = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        seg_list.append({
            "id": i,
            "text": text,
            "start": seg.start,
            "end": seg.end,
            "page": None,
        })
        full_text_parts.append(text)

    return {
        "source_id": f"{source_type}_{source_path.stem}",
        "source_type": source_type,
        "source_path": _relpath(source_path),
        "title": source_path.stem,
        "created_at": datetime.now(UTC).isoformat(),
        "language": info.language,
        "metadata": {
            "pages": None,
            "duration_seconds": info.duration,
            "language_probability": info.language_probability,
            # Queda anotado porque cambia cuánto se le puede creer al texto:
            # sin VAD, Whisper alucina sobre música y ruido.
            "vad_filter": vad,
        },
        "segments": seg_list,
        "full_text": " ".join(full_text_parts),
    }


def _collect_targets() -> list[tuple[Path, str]]:
    """Targets en orden de prioridad: primero las clases (videos/audios), después
    el media de WhatsApp (8k notas de voz de la coach). Cada grupo ordenado por
    nombre; resumable (la corrida saltea los que ya tienen JSON)."""
    targets: list[tuple[Path, str]] = []
    for source_type, exts, subdir in [
        ("video", VIDEO_EXT, "videos"),
        ("audio", AUDIO_EXT, "audios"),
        ("wa_video", VIDEO_EXT, "whatsapp"),
        ("wa_audio", AUDIO_EXT, "whatsapp"),
    ]:
        base = settings.raw_path / subdir
        if not base.exists():
            continue
        group: list[tuple[Path, str]] = []
        for ext in exts:
            group.extend((p, source_type) for p in base.rglob(f"*{ext}"))
        targets.extend(sorted(group, key=lambda t: t[0].name))
    return targets


def run(force: bool = False, limit: int = 0, sin_vad: bool = False) -> None:
    """Transcribe videos/audios.

    limit > 0 procesa como mucho `limit` archivos NUEVOS (los ya hechos no
    cuentan) y luego corta. Pensado para correr de a batches y no trabar la
    maquina con las 155 clases de una. Es resumable: re-llamar sigue donde quedo.

    sin_vad desactiva el filtro de voz; usar junto con --force y --limit para
    revisar un transcript que quedo vacio. Ver la nota en transcribe_file sobre
    las alucinaciones de Whisper cuando no hay habla.
    """
    _register_cuda_dlls()
    from faster_whisper import WhisperModel

    targets = _collect_targets()
    if not targets:
        console.print("[yellow]No hay videos ni audios en data/raw/[/yellow]")
        return

    console.print(
        f"[cyan]Cargando Whisper {settings.whisper_model} en {settings.whisper_device} "
        f"({settings.whisper_compute_type})...[/cyan]"
    )
    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )

    out_dir = settings.transcripts_path
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Transcribiendo {len(targets)} archivo(s)...[/cyan]")
    if limit > 0:
        console.print(f"[cyan]Batch: tope de {limit} archivo(s) nuevos esta corrida.[/cyan]")
    ok = skip = err = sin_audio = 0

    for src_path, source_type in track(targets, description="Transcribiendo"):
        out_path = out_dir / f"{source_type}_{src_path.stem}.json"
        if out_path.exists() and not force:
            skip += 1
            continue

        if limit > 0 and ok >= limit:
            console.print(f"[cyan]Tope de batch alcanzado ({limit}). Corto aca.[/cyan]")
            break

        try:
            if src_path.suffix.lower() in VIDEO_EXT and not has_audio_stream(src_path):
                console.print(f"[yellow]Sin pista de audio, salteo[/yellow] {src_path.name}")
                sin_audio += 1
                continue
            if source_type == "video":
                tmp_audio = Path(tempfile.gettempdir()) / f"_cb_{src_path.stem}.wav"
                try:
                    extract_audio(src_path, tmp_audio)
                    data = transcribe_file(
                        model, tmp_audio, source_type, src_path, vad=not sin_vad
                    )
                finally:
                    tmp_audio.unlink(missing_ok=True)
            else:
                data = transcribe_file(
                    model, src_path, source_type, src_path, vad=not sin_vad
                )

            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
            console.print(
                f"[green]OK[/green] {src_path.name} "
                f"({len(data['segments'])} segmentos, "
                f"{data['metadata']['duration_seconds']:.0f}s)"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error en {src_path.name}: {e}[/red]")
            err += 1

    console.print(
        f"[green]Listo[/green] — ok={ok} skip={skip} "
        f"sin_audio={sin_audio} errores={err}"
    )


if __name__ == "__main__":
    typer.run(run)
