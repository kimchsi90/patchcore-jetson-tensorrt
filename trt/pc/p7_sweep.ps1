# J5 config sweep (PC side): projection dim x coreset ratio x input size for selected categories.
# For each variant: build the ONNX (p1_build_onnx.ps1), add NHWC preprocessing, run the fp32 onnxruntime reference and score it.
# Output: out\sweep_<cat>_s<size>_d<dim>_c<ratio>.json and the summary p7_sweep_summary.csv. Existing results are skipped.
param([string]$Categories = 'bottle,screw')
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$py = 'c:\dev_venvs\PatchcoreJetson\Scripts\python.exe'
Set-Location 'D:\jetson_test\trt'
$summary = 'D:\jetson_test\trt\p7_sweep_summary.csv'
if (-not (Test-Path $summary)) { 'category,size,dim,ratio,bank_rows,image_auroc_max,image_auroc_top10,pixel_auroc,aupro,onnx_mb' | Set-Content $summary }
$variants = @(
    @(256, 384, 0.01), @(256, 384, 0.05), @(256, 384, 0.1),
    @(256, 768, 0.01), @(256, 768, 0.05), @(256, 768, 0.1),
    @(256, 1536, 0.01), @(256, 1536, 0.05), @(256, 1536, 0.1),
    @(224, 384, 0.01)
)
foreach ($c in ($Categories -split ',')) {
    foreach ($v in $variants) {
        $size = $v[0]; $dim = $v[1]; $ratio = $v[2]
        $key = "sweep_${c}_s${size}_d${dim}_c${ratio}"
        $json = "out\$key.json"
        if (Test-Path $json) { Write-Host "$key exists"; continue }
        $w = "D:\jetson_test\patchcore\work\${c}_$size"
        $tag = "random${dim}_c${ratio}"
        $log = "D:\jetson_test\trt\$key.build.log"
        pwsh -NoProfile -File D:\jetson_test\trt\p1_build_onnx.ps1 -Category $c -Size $size -Dim $dim -Ratio $ratio *> $log
        $pf = "$w\fused_${tag}_rmax_pf.onnx"
        if (-not (Test-Path $pf)) { Write-Host "$key FAILED build: $((Get-Content $log -Tail 3) -join ' | ')"; continue }
        $bank = (Select-String -Path $log -Pattern 'bank=\((\d+),' | Select-Object -First 1).Matches.Groups[1].Value
        $pre = "$w\fused_${tag}_rmax_pf_pre.onnx"
        if (-not (Test-Path $pre)) { & $py -I add_preproc.py $pf $pre 2>&1 | Out-Null }
        if (-not (Test-Path $pre)) { Write-Host "$key FAILED preproc"; continue }
        & $py -I dump_trt.py --backend ort --model $pre --images "$w\test" --labels "$w\test_labels.csv" --out "out\$key" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Host "$key FAILED dump"; continue }
        & $py -I score_dump.py --dump "out\$key" --labels "$w\test_labels.csv" --masks "$w\test_masks" > $json
        if ($LASTEXITCODE -ne 0) { Write-Host "$key FAILED score"; Remove-Item $json -ErrorAction SilentlyContinue; continue }
        $j = Get-Content $json -Raw | ConvertFrom-Json
        $mb = [math]::Round((Get-Item $pre).Length / 1MB, 1)
        "$c,$size,$dim,$ratio,$bank,$($j.image_auroc_max),$($j.image_auroc_top10),$($j.pixel_auroc),$($j.aupro),$mb" | Add-Content $summary
        Write-Host ("{0:HH:mm:ss} {1}: bank={2} img={3}/{4} pix={5} aupro={6} onnx={7}MB" -f (Get-Date), $key, $bank, $j.image_auroc_max, $j.image_auroc_top10, $j.pixel_auroc, $j.aupro, $mb)
        # the dump files are large for big banks; keep only the json
        Remove-Item "out\${key}_m.npy", "out\${key}_x.npy" -ErrorAction SilentlyContinue
    }
}
Write-Host 'SWEEP DONE'
