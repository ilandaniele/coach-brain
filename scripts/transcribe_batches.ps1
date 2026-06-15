# Transcribe los videos de a batches, un proceso fresco por batch.
# Cada batch libera la VRAM al terminar -> no traba la maquina como las 155 de una.
# Resumable: saltea los que ya tienen JSON. Corta solo cuando no queda nada nuevo.
#
# Uso:  .\.venv\Scripts\... no; correr con el python del venv:
#   powershell -NoProfile -File scripts\transcribe_batches.ps1 -BatchSize 8

param(
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$outDir = Join-Path $root "data\processed\transcripts"

# Cuenta todos los outputs de Whisper: clases (video_/audio_) + WhatsApp (wa_*).
function Get-Done {
    @("video_*.json","audio_*.json","wa_video_*.json","wa_audio_*.json") |
        ForEach-Object { Get-ChildItem -Path $outDir -Filter $_ -ErrorAction SilentlyContinue } |
        Measure-Object | Select-Object -ExpandProperty Count
}

$batch = 0
while ($true) {
    $batch++
    $before = Get-Done
    Write-Host "=== Batch $batch (hechos hasta ahora: $before) ===" -ForegroundColor Cyan

    & $py -m coach_brain.ingest.transcribe --limit $BatchSize

    $after = Get-Done
    Write-Host "=== Batch $batch termino. Nuevos: $($after - $before). Total: $after ===" -ForegroundColor Green

    if ($after -le $before) {
        Write-Host "No hubo progreso nuevo. Termine todos los videos." -ForegroundColor Yellow
        break
    }
}
