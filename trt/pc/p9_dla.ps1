# DLA accuracy: build + dump on the board (board\dla_eval.sh, already uploaded to ~/trt_dla), pull the dumps, score on the PC.
# Usage: pwsh p9_dla.ps1 [-Categories bottle] [-Variants "fp16,int8mm,int8ent,int8mix"] [-Core 0]
# Output: out\dla_<cat>_<variant>[_core<k>].json, logs in out\dla_logs\, summary p9_dla_summary.csv (deltas against p6_ort_summary.csv).
param(
    [string]$Categories = 'bottle',
    [string]$Variants = 'fp16,int8mm,int8ent,int8mix',
    [int]$Core = 0,
    [string]$Board = '<user>@<board-ip>'
)
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$py = 'c:\dev_venvs\PatchcoreJetson\Scripts\python.exe'
Set-Location 'D:\jetson_test\trt'
$out = 'D:\jetson_test\trt\out'
$logs = "$out\dla_logs"; New-Item -ItemType Directory -Force $logs | Out-Null
$summary = 'D:\jetson_test\trt\p9_dla_summary.csv'
if (-not (Test-Path $summary)) { 'category,engine,image_auroc_max,pixel_auroc,aupro,ref_image_auroc_max,ref_pixel_auroc,ref_aupro,d_img,d_pixel,d_aupro' | Set-Content $summary }
$ref = @{}
Import-Csv 'D:\jetson_test\trt\p6_ort_summary.csv' | ForEach-Object { $ref[$_.category] = $_ }
$b = Get-Content "$out\bottle_256_ort_fp32.json" -Raw | ConvertFrom-Json
$ref['bottle'] = [pscustomobject]@{ image_auroc_max = $b.image_auroc_max; pixel_auroc = $b.pixel_auroc; aupro = $b.aupro }

function Run-Native([string]$exe, [string[]]$a, [int]$sec = 300) {
    $o = [IO.Path]::GetTempFileName()
    $q = $a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process $exe -ArgumentList $q -PassThru -NoNewWindow -RedirectStandardOutput $o -RedirectStandardError "$o.err"
    if (-not $p.WaitForExit($sec * 1000)) { taskkill /F /T /PID $p.Id 2>&1 | Out-Null; Write-Host "  timeout after $sec s: $exe $($a[-1])" }
    Remove-Item $o, "$o.err" -ErrorAction SilentlyContinue
}

foreach ($c in ($Categories -split ',')) {
    $w = "D:\jetson_test\patchcore\work\${c}_256"
    $remote = "/home/<user>/trt_dla/$c"
    foreach ($v in ($Variants -split ',')) {
        $tag = if ($Core -eq 0) { $v } else { "${v}_core$Core" }
        $dump = "$out\dla_${c}_$tag"
        if (Test-Path "$dump.json") { Write-Host "$c/$tag already scored"; continue }
        Write-Host ("{0:HH:mm:ss} {1}/{2}: build+dump" -f (Get-Date), $c, $v)
        Run-Native 'ssh' @('-o', 'BatchMode=yes', $Board, "cd ~/trt_dla && setsid nohup sh dla_eval.sh $c $v $Core > /dev/null 2>&1 < /dev/null &") 60
        $deadline = (Get-Date).AddMinutes(40); $done = $false
        while ((Get-Date) -lt $deadline) {
            Start-Sleep 30
            $r = ssh -o BatchMode=yes -o ConnectTimeout=10 $Board "ls $remote/$v.done 2>/dev/null; cat $remote/$v.status 2>/dev/null; cat /sys/class/thermal/thermal_zone1/temp" 2>&1
            if (($r -join ' ') -match "$v.done") { $done = $true; Write-Host ("  " + ($r -join ' ')); break }
        }
        if (-not $done) { Write-Host "$c/$v TIMEOUT"; continue }
        Run-Native 'scp' @('-o', 'BatchMode=yes', "${Board}:$remote/build_$v.log", "${Board}:$remote/${v}_dla$Core.json", $logs) 120
        Move-Item "$logs\build_$v.log" "$logs\${c}_${tag}_build.log" -Force -ErrorAction SilentlyContinue
        Move-Item "$logs\${v}_dla$Core.json" "$logs\${c}_${tag}_layers.json" -Force -ErrorAction SilentlyContinue
        Run-Native 'scp' @('-o', 'BatchMode=yes', "${Board}:$remote/out_${v}_m.npy", "${Board}:$remote/out_${v}_x.npy", $out) 240
        Move-Item "$out\out_${v}_m.npy" "${dump}_m.npy" -Force -ErrorAction SilentlyContinue
        Move-Item "$out\out_${v}_x.npy" "${dump}_x.npy" -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path "${dump}_m.npy")) { Write-Host "$c/$tag no dump (build/dump failed, see $logs)"; continue }
        & $py -I score_dump.py --dump $dump --labels "$w\test_labels.csv" --masks "$w\test_masks" > "$dump.json"
        if ($LASTEXITCODE -ne 0) { Write-Host "$c/$tag score failed"; Remove-Item "$dump.json"; continue }
        $j = Get-Content "$dump.json" -Raw | ConvertFrom-Json
        $rf = $ref[$c]
        $di = [math]::Round([double]$j.image_auroc_max - [double]$rf.image_auroc_max, 4)
        $dp = [math]::Round([double]$j.pixel_auroc - [double]$rf.pixel_auroc, 4); $da = [math]::Round([double]$j.aupro - [double]$rf.aupro, 4)
        "$c,dla_$tag,$($j.image_auroc_max),$($j.pixel_auroc),$($j.aupro),$($rf.image_auroc_max),$($rf.pixel_auroc),$($rf.aupro),$di,$dp,$da" | Add-Content $summary
        Write-Host ("{0:HH:mm:ss} {1}/{2}: img={3} pix={4} aupro={5} (d_img {6}, d_pix {7}, d_aupro {8})" -f (Get-Date), $c, $tag, $j.image_auroc_max, $j.pixel_auroc, $j.aupro, $di, $dp, $da)
        Remove-Item "${dump}_m.npy", "${dump}_x.npy" -ErrorAction SilentlyContinue
    }
}
Write-Host 'DLA DONE'
