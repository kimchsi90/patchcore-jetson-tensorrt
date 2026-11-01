# Per-category TensorRT accuracy: upload the category data, build + dump on the board (board\cat_eval.sh), score on the PC.
# Usage: pwsh p8_categories.ps1 [-Categories "screw,toothbrush"] [-Fp16For "screw,toothbrush,tile,wood"]
# Output: out\jp514_<cat>_256_trt_{mixed,fp16}.json and the summary p8_trt_summary.csv (deltas against the fp32 reference in p6_ort_summary.csv).
# Run only when the board is otherwise idle (the engine builds load the GPU and would disturb speed measurements).
param(
    [string]$Categories = 'screw,toothbrush,tile,wood,bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,transistor,zipper',
    [string]$Fp16For = 'screw,toothbrush,tile,wood',
    [string]$Board = '<user>@<board-ip>'
)
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$py = 'c:\dev_venvs\PatchcoreJetson\Scripts\python.exe'
Set-Location 'D:\jetson_test\trt'
$out = 'D:\jetson_test\trt\out'
$summary = 'D:\jetson_test\trt\p8_trt_summary.csv'
if (-not (Test-Path $summary)) { 'category,engine,image_auroc_max,pixel_auroc,aupro,ref_image_auroc_max,ref_pixel_auroc,ref_aupro,d_pixel,d_aupro' | Set-Content $summary }
$ref = @{}
Import-Csv 'D:\jetson_test\trt\p6_ort_summary.csv' | ForEach-Object { $ref[$_.category] = $_ }
$b = Get-Content "$out\bottle_256_ort_fp32.json" -Raw | ConvertFrom-Json
$ref['bottle'] = [pscustomobject]@{ image_auroc_max = $b.image_auroc_max; pixel_auroc = $b.pixel_auroc; aupro = $b.aupro }
$fp16Set = $Fp16For -split ','

# Windows OpenSSH sometimes never exits after a finished transfer ("close - IO is still pending"), which blocks the whole batch.
# Run scp/ssh with a time limit and kill the process tree on timeout. The caller checks the result (files / done flag) itself.
function Run-Native([string]$exe, [string[]]$a, [int]$sec = 300) {
    $o = [IO.Path]::GetTempFileName()
    $q = $a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process $exe -ArgumentList $q -PassThru -NoNewWindow -RedirectStandardOutput $o -RedirectStandardError "$o.err"
    if (-not $p.WaitForExit($sec * 1000)) { taskkill /F /T /PID $p.Id 2>&1 | Out-Null; Write-Host "  timeout after $sec s: $exe $($a[-1])" }
    Remove-Item $o, "$o.err" -ErrorAction SilentlyContinue
}

foreach ($c in ($Categories -split ',')) {
    $w = "D:\jetson_test\patchcore\work\${c}_256"
    $remote = "/home/<user>/trt_cat/$c"
    $pre = "$w\fused_random384_c0.01_rmax_pf_pre.onnx"
    if (-not (Test-Path $pre)) { Write-Host "$c MISSING onnx"; continue }
    $want = @('mixed'); if ($fp16Set -contains $c) { $want += 'fp16' }
    if (Test-Path "$out\jp514_${c}_256_trt_mixed.json") { Write-Host "$c already scored"; continue }
    Write-Host ("{0:HH:mm:ss} {1}: upload" -f (Get-Date), $c)
    Run-Native 'ssh' @('-o', 'BatchMode=yes', $Board, "mkdir -p $remote/test $remote/calib") 60
    Run-Native 'scp' @('-o', 'BatchMode=yes', $pre, "${Board}:$remote/pre.onnx") 240
    Run-Native 'scp' @('-o', 'BatchMode=yes', "$w\test_labels.csv", "${Board}:$remote/test_labels.csv") 60
    Run-Native 'scp' @('-o', 'BatchMode=yes', '-r', "$w\test\*", "${Board}:$remote/test/") 240
    Run-Native 'scp' @('-o', 'BatchMode=yes', '-r', "$w\calib\*", "${Board}:$remote/calib/") 240
    $mode = if ($fp16Set -contains $c) { 'fp16' } else { '' }
    Run-Native 'ssh' @('-o', 'BatchMode=yes', $Board, "cd ~/trt_run && setsid nohup sh /home/<user>/cat_eval.sh $c $mode > /dev/null 2>&1 < /dev/null &") 60
    $deadline = (Get-Date).AddMinutes(25); $done = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep 30
        $r = ssh -o BatchMode=yes -o ConnectTimeout=10 $Board "ls $remote/done.flag 2>/dev/null; tail -1 $remote/eval.log" 2>&1
        if (($r -join ' ') -match 'done.flag') { $done = $true; break }
    }
    if (-not $done) { Write-Host "$c TIMEOUT on the board"; continue }
    foreach ($e in $want) {
        $dump = "$out\jp514_${c}_256_trt_$e"
        Run-Native 'scp' @('-o', 'BatchMode=yes', "${Board}:$remote/out_${e}_m.npy", "${Board}:$remote/out_${e}_x.npy", $out) 240
        Move-Item "$out\out_${e}_m.npy" "${dump}_m.npy" -Force -ErrorAction SilentlyContinue
        Move-Item "$out\out_${e}_x.npy" "${dump}_x.npy" -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path "${dump}_m.npy")) { Write-Host "$c/$e no dump"; continue }
        & $py -I score_dump.py --dump $dump --labels "$w\test_labels.csv" --masks "$w\test_masks" > "$dump.json"
        if ($LASTEXITCODE -ne 0) { Write-Host "$c/$e score failed"; continue }
        $j = Get-Content "$dump.json" -Raw | ConvertFrom-Json
        $rf = $ref[$c]
        $dp = [math]::Round([double]$j.pixel_auroc - [double]$rf.pixel_auroc, 4); $da = [math]::Round([double]$j.aupro - [double]$rf.aupro, 4)
        "$c,$e,$($j.image_auroc_max),$($j.pixel_auroc),$($j.aupro),$($rf.image_auroc_max),$($rf.pixel_auroc),$($rf.aupro),$dp,$da" | Add-Content $summary
        Write-Host ("{0:HH:mm:ss} {1}/{2}: img={3} pix={4} aupro={5} (d_pix {6}, d_aupro {7})" -f (Get-Date), $c, $e, $j.image_auroc_max, $j.pixel_auroc, $j.aupro, $dp, $da)
        Remove-Item "${dump}_m.npy", "${dump}_x.npy" -ErrorAction SilentlyContinue
    }
}
Write-Host 'CATEGORIES DONE'
