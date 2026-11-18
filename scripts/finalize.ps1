# PASO FINAL: correr cuando la TRANSCRIPCION ya termino (GPU libre).
# Chequear con scripts\status.ps1 primero. Pipeline 100% LOCAL/GRATIS.
#
# Hace, en orden:
#   1) Junta las imagenes-chat (OCR) en un transcript   (whatsapp_images_ocr aggregate)
#   2) Extraccion LOCAL con Ollama/Qwen-14B (resumable)  (kb.extract, backend=ollama)
#   3) Segunda pasada de extraccion                      (ver OJO 3)
#   4) Re-indexa Qdrant desde cero (con --reset)         (kb.index --reset)
#   5) Levanta el MVP                                     (streamlit)
#
# OJO 1: Qdrant embebido toma lock de UN proceso. Este script CIERRA cualquier
#        Streamlit abierto antes de indexar, pero NO toca un proceso del
#        pipeline: si hay una extraccion o una transcripcion corriendo, aborta.
#        Matarla perderia horas de trabajo.
# OJO 2: La extraccion usa la GPU (Ollama). NO corras esto con la transcripcion
#        (Whisper) activa, o el modelo se va a CPU y tarda ~9 min/chunk.
#        Por eso se corre cuando la transcripcion ya termino.
# OJO 3: `kb.extract` arma su lista de archivos UNA VEZ al arrancar
#        (`sorted(glob("*.json"))`), asi que un transcript que se escribe
#        durante la corrida no entra en ella. El paso 1 escribe justamente uno.
#        De ahi la segunda pasada: es barata, porque saltea todo lo que ya tiene
#        salida en disco.
# OJO 4: el paso 4 es DESTRUCTIVO: borra y reconstruye las 3 colecciones de
#        Qdrant, que son las que esta sirviendo la app. Pide confirmacion.
#        Para saltear la pregunta (desatendido): -Yes
#
# Uso:  powershell -NoProfile -File scripts\finalize.ps1
#       powershell -NoProfile -File scripts\finalize.ps1 -Yes

param([switch]$Yes)

$ErrorActionPreference = "Stop"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

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

# 0b) Liberar el lock de Qdrant, pero sin matar trabajo del pipeline.
#     Se mira la linea de comando de cada python: si alguno esta corriendo una
#     etapa del pipeline, este script no sigue. Una extraccion son dias.
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)
$pipeline = @($procs | Where-Object {
    $_.CommandLine -and ($_.CommandLine -match "coach_brain\.(kb\.extract|kb\.index|ingest\.)")
})
if ($pipeline.Count -gt 0) {
    Write-Host "`nABORTADO: hay $($pipeline.Count) proceso(s) del pipeline corriendo:" -ForegroundColor Red
    foreach ($p in $pipeline) { Write-Host "  PID $($p.ProcessId): $($p.CommandLine)" }
    Write-Host "Esperalos o paralos a mano. Este script no los mata: perderias el avance." -ForegroundColor Red
    exit 1
}
$otros = @($procs | Where-Object { $_.ProcessId -ne $PID })
if ($otros.Count -gt 0) {
    Write-Host "Cerrando $($otros.Count) proceso(s) python (Streamlit) para liberar el lock de Qdrant..." -ForegroundColor Yellow
    foreach ($p in $otros) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

# 1) Agregar las imagenes-chat (OCR) al pipeline
Write-Host "`n[1/5] Agregando screenshots-chat (OCR)..." -ForegroundColor Cyan
& $py -u -m coach_brain.ingest.whatsapp_images_ocr aggregate 2>&1 | Tee-Object -FilePath logs\finalize.log

# 2) Extraccion LOCAL (Ollama). Lenta pero gratis; resumable. Agarra videos +
#    notas de voz + imagenes-chat + texto. Puede tardar HORAS/DIAS por el volumen.
Write-Host "`n[2/5] Extraccion local (Ollama/Qwen-14B, resumable)..." -ForegroundColor Cyan
& $py -u -m coach_brain.kb.extract 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 3) Segunda pasada: barata, levanta lo que se escribio despues del glob (OJO 3).
Write-Host "`n[3/5] Segunda pasada de extraccion..." -ForegroundColor Cyan
& $py -u -m coach_brain.kb.extract 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 4) Re-index full con --reset (sino DUPLICA: ver gotcha #12)
Write-Host "`n[4/5] Indexando en Qdrant (--reset)..." -ForegroundColor Cyan
if (-not $Yes) {
    Write-Host "Esto BORRA las 3 colecciones de Qdrant y las reconstruye." -ForegroundColor Yellow
    Write-Host "Es lo que esta sirviendo la app. El backup esta en qdrant_storage.bak\." -ForegroundColor Yellow
    $r = Read-Host "Escribi SI para seguir"
    if ($r -ne "SI") { Write-Host "Cancelado. El indice quedo intacto." -ForegroundColor Yellow; exit 0 }
}
& $py -u -m coach_brain.kb.index --reset 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 5) MVP. El deploy a Fly es aparte y a mano: flyctl deploy (hornea el indice
#    dentro de la imagen, asi que va DESPUES de este paso).
Write-Host "`n[5/5] Levantando el MVP en http://localhost:8501 ..." -ForegroundColor Green
& $py -u -m streamlit run src\coach_brain\app\streamlit_app.py
