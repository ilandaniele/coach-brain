# PROGRESO del pipeline. Solo lee, no toca nada. Corré las veces que quieras.
#
# Uso:  powershell -NoProfile -File scripts\status.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tx   = Join-Path $root "data\processed\transcripts"
$imgs = Join-Path $root "data\processed\whatsapp_images_ocr"
$extr = Join-Path $root "data\processed\extracted"

function Count($path, $filter) {
    if (-not (Test-Path $path)) { return 0 }
    (Get-ChildItem -Path $path -Filter $filter -ErrorAction SilentlyContinue | Measure-Object).Count
}

# --- Totales del material crudo (lo que hay que procesar) ---
$TOTAL_VIDEOS   = 155
$TOTAL_WA_AUDIO = 8399
$TOTAL_IMAGES   = 8953

# --- Hecho hasta ahora ---
$vid     = Count $tx "video_*.json"
$aud     = Count $tx "audio_*.json"
$waAud   = Count $tx "wa_audio_*.json"
$waVid   = Count $tx "wa_video_*.json"
$imgDone = Count $imgs "*.json"
$exDone  = Count $extr "*.json"

# Cuantas imagenes detectadas como chat (las utiles)
$chats = 0
if (Test-Path $imgs) {
    $chats = (Get-ChildItem -Path $imgs -Filter "*.json" -ErrorAction SilentlyContinue |
        Where-Object { (Get-Content $_.FullName -Raw) -match '"es_chat"\s*:\s*true' } |
        Measure-Object).Count
}

function Bar($done, $total) {
    if ($total -le 0) { return "" }
    $pct = [math]::Min(100, [math]::Round(($done / $total) * 100))
    $fill = [math]::Round($pct / 5)
    "[" + ("#" * $fill) + ("." * (20 - $fill)) + "] $pct%"
}

Write-Host ""
Write-Host "=== PROGRESO coach-brain ===" -ForegroundColor Cyan
Write-Host ""
Write-Host ("  Videos (clases)        {0,5} / {1,-5}  {2}" -f $vid, $TOTAL_VIDEOS, (Bar $vid $TOTAL_VIDEOS)) -ForegroundColor White
Write-Host ("  Notas de voz WhatsApp  {0,5} / {1,-5}  {2}" -f $waAud, $TOTAL_WA_AUDIO, (Bar $waAud $TOTAL_WA_AUDIO)) -ForegroundColor White
Write-Host ("  Imagenes WhatsApp      {0,5} / {1,-5}  {2}" -f $imgDone, $TOTAL_IMAGES, (Bar $imgDone $TOTAL_IMAGES)) -ForegroundColor White
Write-Host ("     -> detectadas como chat: {0}" -f $chats) -ForegroundColor DarkGray
if ($aud -gt 0)   { Write-Host ("  Audios sueltos         {0,5}" -f $aud) -ForegroundColor DarkGray }
if ($waVid -gt 0) { Write-Host ("  Videos WhatsApp        {0,5}" -f $waVid) -ForegroundColor DarkGray }
Write-Host ""
Write-Host ("  Extracciones LLM hechas: {0}" -f $exDone) -ForegroundColor White
Write-Host ""

# --- Esta corriendo algo? ---
$pys = @(Get-Process python -ErrorAction SilentlyContinue)
if ($pys.Count -gt 0) {
    Write-Host ("  {0} proceso(s) python corriendo." -f $pys.Count) -ForegroundColor Green
} else {
    Write-Host "  No hay procesos python corriendo." -ForegroundColor Yellow
}

$allDone = ($vid -ge $TOTAL_VIDEOS) -and ($waAud -ge $TOTAL_WA_AUDIO) -and ($imgDone -ge $TOTAL_IMAGES)
Write-Host ""
if ($allDone) {
    Write-Host "  TODO el material crudo esta procesado." -ForegroundColor Green
    Write-Host "  Paso final:  powershell -NoProfile -File scripts\finalize.ps1" -ForegroundColor Yellow
} else {
    Write-Host "  Todavia queda material por procesar. Si no hay jobs corriendo:" -ForegroundColor Yellow
    Write-Host "  powershell -NoProfile -File scripts\resume.ps1" -ForegroundColor Yellow
}
Write-Host ""
