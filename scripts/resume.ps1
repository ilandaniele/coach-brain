# RETOMAR procesamiento tras reiniciar la PC. Pipeline 100% LOCAL/GRATIS.
# Todo es RESUMABLE: saltea lo que ya está hecho. Seguro correrlo las veces que quieras.
#
# Lanza 2 jobs que CONVIVEN bien (uno GPU, otro CPU):
#   1) Transcripción Whisper (GPU) de a batches  (155 videos -> 8399 notas de voz WhatsApp)
#   2) OCR local de imágenes de WhatsApp (CPU, RapidOCR)
#
# La EXTRACCIÓN (Ollama/Qwen-14B) NO se lanza acá: usa la GPU y pelearía con Whisper
# (se va a CPU = ~9 min/chunk). Corre en el paso final, con la GPU libre -> finalize.ps1
#
# Uso:  powershell -NoProfile -File scripts\resume.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

Write-Host "Retomando jobs locales (resumable) desde $root ..." -ForegroundColor Cyan

# 1) Transcripción (GPU) — de a batches, proceso fresco por batch
Start-Process powershell -WorkingDirectory $root -ArgumentList @(
    "-NoProfile","-NoExit","-Command",
    "& { .\scripts\transcribe_batches.ps1 -BatchSize 8 } 2>&1 | Tee-Object -FilePath logs\transcribe.log"
)

# 2) OCR de imágenes de WhatsApp (CPU, gratis) — reemplaza la vision con API
Start-Process powershell -WorkingDirectory $root -ArgumentList @(
    "-NoProfile","-NoExit","-Command",
    "`$env:PYTHONIOENCODING='utf-8'; & '$py' -u -m coach_brain.ingest.whatsapp_images_ocr run 2>&1 | Tee-Object -FilePath logs\ocr.log"
)

Write-Host ""
Write-Host "2 ventanas abiertas (transcripcion GPU / OCR CPU). Ambas gratis." -ForegroundColor Green
Write-Host "Progreso:   powershell -NoProfile -File scripts\status.ps1" -ForegroundColor Yellow
Write-Host "Cuando la TRANSCRIPCION termine (GPU libre), paso final (extrae+indexa+MVP):" -ForegroundColor Yellow
Write-Host "  powershell -NoProfile -File scripts\finalize.ps1" -ForegroundColor Yellow
