# P6 (PC side): download the official MVTec AD per-category archives and build the 256-input ONNX set for each category.
# Sources: official MVTec download page links (CC BY-NC-SA 4.0, non-commercial). Sequential, resumable (existing outputs are skipped).
# Usage: pwsh p6_prepare_all.ps1 [-Categories "cable,capsule"] [-Size 256]
param([string]$Categories = '', [int]$Size = 256)
$ErrorActionPreference = 'Continue'
$base = 'https://www.mydrive.ch/shares'
$urls = [ordered]@{
    cable      = "$base/163641/97fd0295152453e533cfb3db1d253a35/download/420937413-1629958794/cable.tar.xz"
    capsule    = "$base/163642/236d4304eaf9526b350b18807789ce86/download/420937454-1629958872/capsule.tar.xz"
    carpet     = "$base/163643/70e8bebbe9f2d2c55043311a4e67eb1d/download/420937484-1629959013/carpet.tar.xz"
    grid       = "$base/163644/8c7fc7cba43ef5275c87ca750a6d0004/download/420937487-1629959044/grid.tar.xz"
    hazelnut   = "$base/163645/8346f81c9e093d2ebbb3a73f98372217/download/420937545-1629959162/hazelnut.tar.xz"
    leather    = "$base/163646/7e0627b5718d2902bb85d24cff7a51ce/download/420937607-1629959262/leather.tar.xz"
    metal_nut  = "$base/163647/6d5a833cdee9035a45d5ce0921ecd24d/download/420937637-1629959294/metal_nut.tar.xz"
    pill       = "$base/163648/f8c4ca0d325888f2ba128a9ed44708f5/download/420938129-1629960351/pill.tar.xz"
    screw      = "$base/163649/cfd394b4e3a49013745fab9e2084988a/download/420938130-1629960389/screw.tar.xz"
    tile       = "$base/163650/55ad2038770deff6fe6771bd784af8cc/download/420938133-1629960456/tile.tar.xz"
    toothbrush = "$base/163651/f03d8421ec64b3e3bfa4a2ba2fbafb4f/download/420938134-1629960477/toothbrush.tar.xz"
    transistor = "$base/163652/619a88c3d5f3f57b46b00f44f04ad6e6/download/420938166-1629960554/transistor.tar.xz"
    wood       = "$base/163653/937e0d65bc9c4497d253cb7313868a60/download/420938383-1629960649/wood.tar.xz"
    zipper     = "$base/163654/7f6e6aefbf00a9e8a09cc0bbac236eb0/download/420938385-1629960680/zipper.tar.xz"
}
$dl = 'D:\jetson_test\downloads\mvtec'
$data = 'D:\jetson_test\patchcore\datasets\mvtec_ad'
$work = 'D:\jetson_test\patchcore\work'
$summary = 'D:\jetson_test\trt\p6_summary.csv'
New-Item -ItemType Directory -Force $dl, $data | Out-Null
if (-not (Test-Path $summary)) { 'category,status,detail' | Set-Content $summary }
$todo = if ($Categories) { $Categories -split ',' } else { @($urls.Keys) }

function Note([string]$c, [string]$status, [string]$detail) {
    "$c,$status,$detail" | Add-Content $summary
    Write-Host ("{0:HH:mm:ss} {1}: {2} {3}" -f (Get-Date), $c, $status, $detail)
}

foreach ($c in $todo) {
    if (-not $urls.Contains($c)) { Note $c 'SKIP' 'unknown category'; continue }
    $fused = "$work\${c}_$Size\fused_random384_c0.01_rmax_pf.onnx"
    if (Test-Path $fused) { Note $c 'OK' 'onnx already built'; continue }
    $f = "$dl\$c.tar.xz"
    if (-not (Test-Path "$data\$c\train\good")) {
        if (-not (Test-Path $f) -or (Get-Item $f).Length -lt 50MB) {
            curl.exe -L --fail -sS --retry 5 --retry-all-errors -o $f $urls[$c]
            if ($LASTEXITCODE -ne 0) { Note $c 'FAILED' "download curl exit $LASTEXITCODE"; continue }
        }
        tar -tf $f 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { Note $c 'FAILED' 'archive is not a valid tar'; continue }
        tar -xf $f -C $data
        if ($LASTEXITCODE -ne 0) { Note $c 'FAILED' "tar extract exit $LASTEXITCODE"; continue }
    }
    $ntr = @(Get-ChildItem "$data\$c\train\good" -Filter *.png -ErrorAction SilentlyContinue).Count
    $nte = @(Get-ChildItem "$data\$c\test" -Recurse -Filter *.png -ErrorAction SilentlyContinue).Count
    if ($ntr -lt 50 -or $nte -lt 20) { Note $c 'FAILED' "unexpected layout train=$ntr test=$nte"; continue }
    $log = "D:\jetson_test\trt\p1_${c}_$Size.log"
    pwsh -NoProfile -File D:\jetson_test\trt\p1_build_onnx.ps1 -Category $c -Size $Size *> $log
    if ((Get-Content $log -Tail 3 | Out-String) -match 'RESULT=SUCCESS' -and (Test-Path $fused)) { Note $c 'OK' "train=$ntr test=$nte" }
    else { Note $c 'FAILED' ("p1 build; last line: " + ((Get-Content $log -Tail 4 | Where-Object { $_ }) -join ' | ')) }
}
Write-Host 'BATCH DONE'
