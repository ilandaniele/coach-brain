# PASO FINAL: correr cuando la TRANSCRIPCION ya termino (GPU libre).
# Chequear con scripts\status.ps1 primero. Pipeline 100% LOCAL/GRATIS.
#
# Hace, en orden:
#   1) Junta las imagenes-chat (OCR) en un transcript   (whatsapp_images_ocr aggregate)
#   2) Extraccion LOCAL con Ollama/Qwen-14B (resumable)  (kb.extract, backend=ollama)
#   3) Segunda pasada de extraccion                      (ver OJO 3)
#   4) Verifica huecos silenciosos del corpus             (scripts\verificar.py)
#   5) Informa el staging de la auditoria sin VAD        (--juzgar, no promueve)
#   6) Pasa el detector por el corpus VIVO               (scripts\escanear_corpus.py)
#   7) Re-indexa Qdrant desde cero (con --reset)         (kb.index --reset)
#   8) Regenera INVENTARIO.md                            (scripts\manifest.py)
#   9) Levanta el MVP                                     (streamlit)
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
# OJO 4: el paso 7 es DESTRUCTIVO: borra y reconstruye las 3 colecciones de
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
    # Los scripts de scripts/ tambien son trabajo del pipeline: la auditoria
    # sin VAD son horas de Whisper, y reparar_source_path ESCRIBE transcripts.
    # Si no estuvieran aca, el bloque de abajo los mataria como "Streamlit".
    $_.CommandLine -and ($_.CommandLine -match "coach_brain\.(kb\.extract|kb\.index|ingest\.)|retranscribir_sin_vad|reparar_source_path")
})
if ($pipeline.Count -gt 0) {
    Write-Host "`nABORTADO: hay $($pipeline.Count) proceso(s) del pipeline corriendo:" -ForegroundColor Red
    foreach ($p in $pipeline) { Write-Host "  PID $($p.ProcessId): $($p.CommandLine)" }
    Write-Host "Esperalos o paralos a mano. Este script no los mata: perderias el avance." -ForegroundColor Red
    exit 1
}
#     Y se matan SOLO los que son Streamlit. Antes se mataba todo python.exe
#     que no fuera este proceso, pero este proceso es powershell.exe, asi que
#     $PID nunca coincide con un python y el filtro no filtraba nada: se iba
#     abajo cualquier python de la maquina, de este proyecto o de otro. Si algo
#     que no es Streamlit tiene el lock, mejor que el paso 7 falle con el error
#     de Qdrant a la vista que perder trabajo ajeno en silencio.
$otros = @($procs | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine -match "streamlit"
})
if ($otros.Count -gt 0) {
    Write-Host "Cerrando $($otros.Count) proceso(s) de Streamlit para liberar el lock de Qdrant..." -ForegroundColor Yellow
    foreach ($p in $otros) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

# 1) Agregar las imagenes-chat (OCR) al pipeline
Write-Host "`n[1/9] Agregando screenshots-chat (OCR)..." -ForegroundColor Cyan
& $py -u -m coach_brain.ingest.whatsapp_images_ocr aggregate 2>&1 | Tee-Object -FilePath logs\finalize.log

# 2) Extraccion LOCAL (Ollama). Lenta pero gratis; resumable. Agarra videos +
#    notas de voz + imagenes-chat + texto. Puede tardar HORAS/DIAS por el volumen.
Write-Host "`n[2/9] Extraccion local (Ollama/Qwen-14B, resumable)..." -ForegroundColor Cyan
& $py -u -m coach_brain.kb.extract 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 3) Segunda pasada: barata, levanta lo que se escribio despues del glob (OJO 3).
Write-Host "`n[3/9] Segunda pasada de extraccion..." -ForegroundColor Cyan
& $py -u -m coach_brain.kb.extract 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 4) Verificacion de huecos silenciosos. Es de solo lectura y NO aborta: con el
#    corpus sano igual devuelve 1, porque hay notas de voz cortas que extraen 0
#    items legitimamente. Corre antes del paso destructivo para que el estado
#    quede escrito en el log junto al indice que se reconstruye.
Write-Host "`n[4/9] Verificando huecos del corpus..." -ForegroundColor Cyan
& $py -u scripts\verificar.py --sin-audio 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 5) Que hay en el staging de la auditoria sin VAD. Solo lectura: no promueve
#    ni borra nada. Si algo de ahi se va a promover, hay que hacerlo A MANO
#    ANTES del paso 7, porque el indice se reconstruye desde los transcripts.
Write-Host "`n[5/9] Auditoria sin VAD (informe, no promueve nada)..." -ForegroundColor Cyan
& $py -u scripts\retranscribir_sin_vad.py --juzgar 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 6) El mismo detector, sobre el corpus VIVO. Esta aca y no solo en la
#    auditoria porque el corpus con VAD tampoco esta limpio: el transcript
#    de wa_audio_00021896 son 1907 caracteres de "Gracias por ver el video"
#    repetido 25 veces. Lo que importa de ese informe es la columna de items:
#    el indice se arma desde extracted\, asi que un marcado con 0 items ya lo
#    filtro el extractor y no aporto ningun vector. Solo lectura, no aborta:
#    lo que se saque del corpus lo decide una persona que leyo el texto, y
#    tiene que decidirlo ANTES del paso 7, que reconstruye el indice.
Write-Host "`n[6/9] Escaneando el corpus vivo (informe, no saca nada)..." -ForegroundColor Cyan
& $py -u scripts\escanear_corpus.py 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 7) Re-index full con --reset (sino DUPLICA: ver gotcha #12)
Write-Host "`n[7/9] Indexando en Qdrant (--reset)..." -ForegroundColor Cyan
if (-not $Yes) {
    Write-Host "Esto BORRA las 3 colecciones de Qdrant y las reconstruye." -ForegroundColor Yellow
    Write-Host "Es lo que esta sirviendo la app. El backup esta en qdrant_storage.bak\." -ForegroundColor Yellow
    $r = Read-Host "Escribi SI para seguir"
    if ($r -ne "SI") { Write-Host "Cancelado. El indice quedo intacto." -ForegroundColor Yellow; exit 0 }
}
& $py -u -m coach_brain.kb.index --reset 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 8) Regenerar el inventario. Es de solo lectura sobre el material y reescribe
#    INVENTARIO.md, que es el registro de que esta cubierto y que no. Despues
#    del re-indexado, porque es lo que describe.
Write-Host "`n[8/9] Regenerando INVENTARIO.md..." -ForegroundColor Cyan
& $py -u scripts\manifest.py 2>&1 | Tee-Object -FilePath logs\finalize.log -Append

# 9) MVP. El deploy a Fly es aparte y a mano: flyctl deploy (hornea el indice
#    dentro de la imagen, asi que va DESPUES de este paso).
Write-Host "`n[9/9] Levantando el MVP en http://localhost:8501 ..." -ForegroundColor Green
& $py -u -m streamlit run src\coach_brain\app\streamlit_app.py
