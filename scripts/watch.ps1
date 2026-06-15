# Dashboard de progreso EN VIVO. Se autoactualiza cada N segundos.
# Solo lee del disco, no toca nada. Ctrl+C para salir.
#
# Uso:  powershell -NoProfile -File scripts\watch.ps1
#       powershell -NoProfile -File scripts\watch.ps1 -Every 10

param([int]$Every = 20)

$root = Split-Path -Parent $PSScriptRoot
$tx   = Join-Path $root "data\processed\transcripts"
$imgs = Join-Path $root "data\processed\whatsapp_images_ocr"
$extr = Join-Path $root "data\processed\extracted"

$TOTAL_VIDEOS = 155; $TOTAL_WA_AUDIO = 8399; $TOTAL_IMAGES = 8953

function Count($path, $filter) {
    if (-not (Test-Path $path)) { return 0 }
    (Get-ChildItem -Path $path -Filter $filter -ErrorAction SilentlyContinue | Measure-Object).Count
}
function Bar($done, $total) {
    if ($total -le 0) { return "" }
    $pct = [math]::Min(100, [math]::Round(($done / $total) * 100))
    $fill = [math]::Round($pct / 5)
    "[" + ("#" * $fill) + ("." * (20 - $fill)) + "] $pct%"
}

# Guarda la lectura anterior para mostrar el "ritmo" (cuanto avanza por ciclo)
$prev = @{ vid = $null; wa = $null; img = $null }

while ($true) {
    $vid = Count $tx "video_*.json"
    $wa  = Count $tx "wa_audio_*.json"
    $img = Count $imgs "*.json"
    $ex  = Count $extr "*.json"

    function Delta($cur, $old) { if ($null -eq $old) { "" } else { $d = $cur - $old; if ($d -gt 0) { "  (+$d)" } else { "" } } }

    Clear-Host
    Write-Host "=== coach-brain — PROGRESO EN VIVO ===" -ForegroundColor Cyan
    Write-Host ("    $(Get-Date -Format 'HH:mm:ss')  ·  refresca cada ${Every}s  ·  Ctrl+C para salir") -ForegroundColor DarkGray
    Write-Host ""
    Write-Host ("  Videos (clases)        {0,5} / {1,-5} {2}{3}" -f $vid, $TOTAL_VIDEOS, (Bar $vid $TOTAL_VIDEOS), (Delta $vid $prev.vid)) -ForegroundColor White
    Write-Host ("  Notas de voz WhatsApp  {0,5} / {1,-5} {2}{3}" -f $wa, $TOTAL_WA_AUDIO, (Bar $wa $TOTAL_WA_AUDIO), (Delta $wa $prev.wa)) -ForegroundColor White
    Write-Host ("  Imagenes (OCR)         {0,5} / {1,-5} {2}{3}" -f $img, $TOTAL_IMAGES, (Bar $img $TOTAL_IMAGES), (Delta $img $prev.img)) -ForegroundColor White
    Write-Host ("  Extracciones           {0,5}" -f $ex) -ForegroundColor White
    Write-Host ""

    $pys = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($pys.Count -gt 0) {
        Write-Host ("  {0} proceso(s) python corriendo." -f $pys.Count) -ForegroundColor Green
    } else {
        Write-Host "  No hay procesos python corriendo." -ForegroundColor Yellow
        if (($vid -ge $TOTAL_VIDEOS) -and ($wa -ge $TOTAL_WA_AUDIO) -and ($img -ge $TOTAL_IMAGES)) {
            Write-Host "  TODO procesado -> corre: scripts\finalize.ps1" -ForegroundColor Yellow
        } else {
            Write-Host "  Para retomar -> scripts\resume.ps1" -ForegroundColor Yellow
        }
    }

    $prev = @{ vid = $vid; wa = $wa; img = $img }
    Start-Sleep -Seconds $Every
}
