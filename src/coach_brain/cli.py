"""CLI principal — entrypoint via `uv run coach <comando>` o `python -m coach_brain.cli`."""

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="coach-brain CLI")
console = Console()


@app.command()
def ingest_pdfs():
    """Procesa todos los PDFs en data/raw/pdfs/ y guarda transcripts."""
    from coach_brain.ingest.pdfs import run

    run()


@app.command()
def ingest_docs():
    """Procesa .docx/.epub en data/raw/docs/ y guarda transcripts."""
    from coach_brain.ingest.docs import run

    run()


@app.command()
def ingest_whatsapp(force: bool = False):
    """Parsea el export de WhatsApp en data/raw/whatsapp/ (_chat.txt)."""
    from coach_brain.ingest.whatsapp import run

    run(force=force)


@app.command()
def transcribe():
    """Transcribe videos/audios en data/raw/ con Whisper local."""
    from coach_brain.ingest.transcribe import run

    run()


@app.command()
def download_drive(folder_url: str):
    """Descarga una carpeta pública de Drive a data/raw/."""
    from coach_brain.ingest.drive import download

    download(folder_url)


@app.command()
def extract():
    """Extrae principios/situaciones/frases desde transcripts via Claude."""
    from coach_brain.kb.extract import run

    run()


@app.command()
def index():
    """Indexa lo extraído en Qdrant con embeddings bge-m3."""
    from coach_brain.kb.index import run

    run()


@app.command()
def app_run():
    """Lanza la UI de Streamlit."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).parent / "app" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script)])


if __name__ == "__main__":
    app()
