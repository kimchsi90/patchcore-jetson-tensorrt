param(
    [Parameter(Mandatory)][string]$ZipPath,
    [Parameter(Mandatory)][string]$LogPath,
    [double]$ExpectGB = 59.5
)
# Writes the single .img inside a zip to the one USB SD/MMC disk of ~$ExpectGB GB.
# Safety: aborts unless exactly one non-system, non-boot USB disk matches.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Log($m) { $l = "{0:HH:mm:ss} {1}" -f (Get-Date), $m; Add-Content -LiteralPath $LogPath -Value $l; Write-Host $l }

$BLK = 4MB
$fs = $null; $zip = $null; $in = $null; $rd = $null
try {
    $cands = @(Get-Disk | Where-Object {
            $_.BusType -eq 'USB' -and -not $_.IsSystem -and -not $_.IsBoot -and
            [math]::Abs($_.Size / 1GB - $ExpectGB) -lt 1.0 -and $_.FriendlyName -like 'Generic*SD*'
        })
    if ($cands.Count -ne 1) { throw "Expected exactly 1 target disk, found $($cands.Count)" }
    $disk = $cands[0]; $n = $disk.Number
    Log "Target: Disk $n '$($disk.FriendlyName)' $([math]::Round($disk.Size/1GB,2)) GB, BusType=$($disk.BusType)"

    $zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    $entry = @($zip.Entries | Where-Object { $_.FullName -like '*.img' })
    if ($entry.Count -ne 1) { throw "Expected exactly 1 .img in zip, found $($entry.Count)" }
    $imgLen = $entry[0].Length
    if ($imgLen -ge $disk.Size) { throw "Image ($imgLen B) does not fit on disk ($($disk.Size) B)" }
    Log "Image: $($entry[0].FullName) $imgLen bytes"

    Log "Clear-Disk $n (removes all partitions)"
    Clear-Disk -Number $n -RemoveData -RemoveOEM -Confirm:$false

    $path = "\\.\PhysicalDrive$n"
    $fs = New-Object IO.FileStream($path, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite, 1, [IO.FileOptions]::WriteThrough)
    $in = $entry[0].Open()
    $sha = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)

    $buf = New-Object byte[] $BLK
    $first = $null; $total = 0L; $nextLog = 512MB
    while ($true) {
        $got = 0
        while ($got -lt $BLK) {
            $r = $in.Read($buf, $got, $BLK - $got)
            if ($r -le 0) { break }
            $got += $r
        }
        if ($got -le 0) { break }
        $sha.AppendData($buf, 0, $got)
        if ($got -lt $BLK) { [Array]::Clear($buf, $got, $BLK - $got) }   # pad last block with zeros
        if ($null -eq $first) {
            $first = $buf.Clone()      # hold the first block; write it last
            $fs.Seek($BLK, [IO.SeekOrigin]::Begin) | Out-Null
        } else {
            $fs.Write($buf, 0, $BLK)
        }
        $total += $got
        if ($total -ge $nextLog) { Log ("written {0:N2} / {1:N2} GB" -f ($total / 1GB), ($imgLen / 1GB)); $nextLog += 512MB }
    }
    if ($total -ne $imgLen) { throw "Short read: $total of $imgLen bytes" }
    $fs.Seek(0, [IO.SeekOrigin]::Begin) | Out-Null
    $fs.Write($first, 0, $BLK)
    $fs.Flush()
    $want = [BitConverter]::ToString($sha.GetHashAndReset()).Replace('-', '')
    Log "Write done. Source SHA256 = $want"

    # Read back exactly $imgLen bytes and compare.
    $fs.Seek(0, [IO.SeekOrigin]::Begin) | Out-Null
    $sha2 = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
    [long]$remain = $imgLen; [long]$done = 0; [long]$nextLog = 2GB
    while ($remain -gt 0) {
        # read whole 4 MiB blocks; the zero padding of the last block lies inside the written area
        $got = $fs.Read($buf, 0, $BLK)
        if ($got -le 0) { throw "Read-back ended early at $done bytes" }
        if ([long]$got -gt $remain) { $use = [int]$remain } else { $use = $got }
        $sha2.AppendData($buf, 0, $use)
        $remain -= $use; $done += $use
        if ($done -ge $nextLog) { Log ("verified {0:N2} GB" -f ($done / 1GB)); $nextLog += 2GB }
    }
    $got2 = [BitConverter]::ToString($sha2.GetHashAndReset()).Replace('-', '')
    Log "Card SHA256 = $got2"
    if ($got2 -ne $want) { throw "VERIFY MISMATCH: card hash differs from source" }
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
finally {
    if ($fs) { $fs.Dispose() }
    if ($in) { $in.Dispose() }
    if ($zip) { $zip.Dispose() }
}
