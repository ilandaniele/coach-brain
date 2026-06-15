# PASO FINAL: correr cuando la TRANSCRIPCION ya termino (GPU libre).
# Chequear con scripts\status.ps1 primero. Pipeline 100% LOCAL/GRATIS.
#
# Hace, en orden:
#   1) Junta las imagenes-chat (OCR) en un transcript   (whatsapp_images_ocr aggregate)
#   2) Extraccion LOCAL con Ollama/Qwen-14B (resumable)  (kb.extract, backend=ollama)
#   3) Re-indexa Qdrant desde cero (con --reset)         (kb.index --reset)
#   4) Levanta el MVP                                     (streamlit)
#
# OJO 1: Qdrant embebido toma lock de UN proceso. Este script CIERRA cualquier
#        Streamlit/python abierto antes de indexar.
# OJO 2: La extraccion usa la GPU (Ollama). NO corras esto con la transcripcion
#        (Whisper) activa, o el modelo se va a CPU y tarda ~9 min/chunk.
#        Por eso se corre cuando la transcripcion ya termino.
#
# Uso:  powershell -NoProfile -File scripts\finalize.ps1

$ErrorActionPreference = "Stop"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

Write-Host "=== PASO FINAL coach-brain (local/gratis) ===" -ForegroundColor Cyan

# 0a) Asegurar que el server de Ollama corra (lo necesita la extraccion local)
if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Arrancando servidor Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}

# 0b) Cerrar Streamlit / cualquier python que tenga el lock de Qdrant.
$pys = @(Get-Process python -ErrorAction SilentlyContinue)
if ($pys.Count -gt 0) {
    Write-Host "Cerrando $($pys.Count) proceso(s) python para liberar el lock de Qdrant..." -ForegroundColor Yellow
    $pys | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# 1) Agregar las imagenes-chat (OCR) al pipeline
Write-Host "`n[1/4] Agregando screenshots-chat (OCR)..." -ForegroundColor Cyan
& $py -u -m coach_brain.ingest.whatsapp_images_ocr aggregate 2>&1 | Tee-Object -FilePath logs\finalize.log

# 2) Extraccion LOCAL (Ollama). Lenta pero gratis; resumable. Agarra videos +
#    notas de voz + imagenes-chat + texto. Puede tardar HORAS/DIAS por el volumen.
Write-Host "`n[2/4] Extraccion local (Ollama/Qwen-14B, resumable)..." -ForegroundColor Cyan
& $py -u -m coach_brain.kb.extract 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 3) Re-index full con --reset (sino DUPLICA: ver gotcha #12)
Write-Host "`n[3/4] Indexando en Qdrant (--reset)..." -ForegroundColor Cyan
& $py -u -m coach_brain.kb.index --reset 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 4) MVP
Write-Host "`n[4/4] Levantando el MVP en http://localhost:8501 ..." -ForegroundColor Green
& $py -u -m streamlit run src\coach_brain\app\streamlit_app.py
