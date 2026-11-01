# fp32 onnxruntime reference accuracy for every prepared category (CPU only, runs next to the GPU-free parts of the pipeline).
# Output: out\<cat>_256_ort_fp32.json (+ _m.npy/_x.npy dumps), summary in p6_ort_summary.csv. Existing results are skipped.
param([string]$Categories = 'bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper')
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$py = 'c:\dev_venvs\PatchcoreJetson\Scripts\python.exe'
Set-Location 'D:\jetson_test\trt'
$summary = 'D:\jetson_test\trt\p6_ort_summary.csv'
if (-not (Test-Path $summary)) { 'category,image_auroc_max,image_auroc_top10,pixel_auroc,aupro,peak_abs_m,peak_xproj_sq_norm' | Set-Content $summary }
foreach ($c in ($Categories -split ',')) {
    $w = "D:\jetson_test\patchcore\work\${c}_256"
    $json = "out\${c}_256_ort_fp32.json"
    if (Test-Path $json) { Write-Host "$c exists"; continue }
    $model = "$w\fused_random384_c0.01_rmax_pf_pre.onnx"
    if (-not (Test-Path $model)) { Write-Host "$c MISSING model"; continue }
    & $py -I dump_trt.py --backend ort --model $model --images "$w\test" --labels "$w\test_labels.csv" --out "out\${c}_256_ort_fp32" 2>&1 | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { Write-Host "$c FAILED dump"; continue }
    & $py -I score_dump.py --dump "out\${c}_256_ort_fp32" --labels "$w\test_labels.csv" --masks "$w\test_masks" > $json
    if ($LASTEXITCODE -ne 0) { Write-Host "$c FAILED score"; Remove-Item $json -ErrorAction SilentlyContinue; continue }
    $j = Get-Content $json -Raw | ConvertFrom-Json
    "$c,$($j.image_auroc_max),$($j.image_auroc_top10),$($j.pixel_auroc),$($j.aupro),$($j.peak_abs_m),$($j.peak_xproj_sq_norm)" | Add-Content $summary
    Write-Host ("{0:HH:mm:ss} {1}: img={2} pix={3} aupro={4}" -f (Get-Date), $c, $j.image_auroc_max, $j.pixel_auroc, $j.aupro)
}
Write-Host 'REFS DONE'
