param(
    [Parameter(Mandatory)][long]$Length,
    [Parameter(Mandatory)][string]$ExpectSha,
    [Parameter(Mandatory)][string]$LogPath,
    [double]$ExpectGB = 59.5
)
# Read-only: hashes the first $Length bytes of the single USB SD/MMC disk and compares with $ExpectSha.
$ErrorActionPreference = 'Stop'
function Log($m) { $l = "{0:HH:mm:ss} {1}" -f (Get-Date), $m; Add-Content -LiteralPath $LogPath -Value $l; Write-Host $l }
$fs = $null
try {
    $cands = @(Get-Disk | Where-Object {
            $_.BusType -eq 'USB' -and -not $_.IsSystem -and -not $_.IsBoot -and
            [math]::Abs($_.Size / 1GB - $ExpectGB) -lt 1.0 -and $_.FriendlyName -like 'Generic*SD*'
        })
    if ($cands.Count -ne 1) { throw "Expected exactly 1 target disk, found $($cands.Count)" }
    $n = $cands[0].Number
    Log "Verify Disk $n, $Length bytes"
    $blk = 4MB
    $fs = New-Object IO.FileStream("\\.\PhysicalDrive$n", [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite, 1)
    $sha = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
    $buf = New-Object byte[] $blk
    [long]$remain = $Length; [long]$done = 0; [long]$nextLog = 2GB
    while ($remain -gt 0) {
        $got = $fs.Read($buf, 0, $blk)
        if ($got -le 0) { throw "Read ended early at $done bytes" }
        if ([long]$got -gt $remain) { $use = [int]$remain } else { $use = $got }
        $sha.AppendData($buf, 0, $use)
        $remain -= $use; $done += $use
        if ($done -ge $nextLog) { Log ("verified {0:N2} GB" -f ($done / 1GB)); $nextLog += 2GB }
    }
    $h = [BitConverter]::ToString($sha.GetHashAndReset()).Replace('-', '')
    Log "Card SHA256   = $h"
    Log "Expected      = $ExpectSha"
    if ($h -ne $ExpectSha) { throw "VERIFY MISMATCH" }
    Log "VERIFY OK"
    $fs.Dispose(); $fs = $null
    Set-Disk -Number $n -IsOffline $true
    Log "Disk $n set offline (safe to remove). RESULT=SUCCESS"
}
catch {
    Log "ERROR: $($_.Exception.Message)"
    Log "RESULT=FAILED"
    exit 1
}
finally { if ($fs) { $fs.Dispose() } }
