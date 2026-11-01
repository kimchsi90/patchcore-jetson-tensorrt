# Waits for the Jetson SD card to show up on the PC, then writes the JetPack 5.1.4 image with SHA256 read-back verification.
# Safety: only a non-system, non-boot USB disk of ~59.5 GB named "Generic*SD*" that already has a Jetson-style partition table
# (>= 8 partitions) is accepted. Anything else is ignored. A marker file prevents a second write.
param([int]$MaxHours = 14)
$ErrorActionPreference = 'Continue'
$sp = $PSScriptRoot   # write_sd.ps1 is expected next to this script
$zip = 'D:\jetson_test\downloads\jp514\JP514-xnx-sd-card-image_b11.zip'
$marker = 'D:\jetson_test\trt\sd_514_written.flag'
$log = 'D:\jetson_test\trt\sd_514_write.log'
function Log($m) { $l = "{0:HH:mm:ss} {1}" -f (Get-Date), $m; Add-Content -LiteralPath $log -Value $l; Write-Host $l }
if (Test-Path $marker) { Log 'marker exists, nothing to do'; exit 0 }
Log "waiting for the SD card (max $MaxHours h)"
$deadline = (Get-Date).AddHours($MaxHours)
while ((Get-Date) -lt $deadline) {
    $cands = @(Get-Disk -ErrorAction SilentlyContinue | Where-Object {
            $_.BusType -eq 'USB' -and -not $_.IsSystem -and -not $_.IsBoot -and $_.Size -gt 0 -and
            [math]::Abs($_.Size / 1GB - 59.5) -lt 1.0 -and $_.FriendlyName -like 'Generic*SD*' })
    if ($cands.Count -eq 1) {
        $n = $cands[0].Number
        $parts = @(Get-Partition -DiskNumber $n -ErrorAction SilentlyContinue).Count
        Log "candidate Disk $n, partitions=$parts, offline=$($cands[0].IsOffline)"
        if ($cands[0].IsOffline) { Set-Disk -Number $n -IsOffline $false -ErrorAction SilentlyContinue; Start-Sleep 3; $parts = @(Get-Partition -DiskNumber $n -ErrorAction SilentlyContinue).Count }
        if ($parts -ge 8) {
            Log 'Jetson-style partition table found, starting the write'
            Remove-Item 'D:\jetson_test\trt\sd_514_write_detail.log' -ErrorAction SilentlyContinue
            & "$sp\write_sd.ps1" -ZipPath $zip -LogPath 'D:\jetson_test\trt\sd_514_write_detail.log'
            # success is judged from the log, because the script does not set an exit code on success
            $ok = Select-String -Path 'D:\jetson_test\trt\sd_514_write_detail.log' -Pattern 'RESULT=SUCCESS' -Quiet
            if ($ok) { Set-Content $marker "written $(Get-Date -Format s)"; Log 'WRITE OK'; exit 0 }
            Log 'WRITE FAILED, see sd_514_write_detail.log'; exit 1
        }
        Log 'partition table does not look like the Jetson card, ignoring'
    }
    Start-Sleep 5
}
Log 'timeout waiting for the card'
exit 2
