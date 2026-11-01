# P1: data prep and ONNX export for one MVTec AD category, no RKNN step.
# Same call order as patchcore\rknn\build_category_model.ps1, minus rewrite_max_tree / rewrite_raw_out / convert_algo.
# Usage: pwsh p1_build_onnx.ps1 -Category bottle -Size 256 [-Dim 384] [-Ratio 0.01]
param([string]$Category = 'bottle', [int]$Size = 256, [int]$Dim = 384, [double]$Ratio = 0.01)
$ErrorActionPreference = 'Stop'
$py = 'c:\dev_venvs\PatchcoreJetson\Scripts\python.exe'
$root = 'D:\jetson_test\patchcore'
$env:PYTHONUTF8 = '1'
$env:DATASET_ROOT = "$root\datasets\mvtec_ad"
$env:NPU_SIZE = "$Size"; $env:NPU_DIR_SUFFIX = "_$Size"
$bb = "$root\work\backbones\wide_resnet50_2_$Size.onnx"
$env:NPU_BACKBONE_ONNX = $bb
$w = "$root\work\${Category}_$Size"
$tag = "random${Dim}_c${Ratio}"

function Step([string]$name, [string[]]$pyArgs, [string]$expect) {
    Write-Host ("{0:HH:mm:ss} {1}" -f (Get-Date), $name)
    & $py @pyArgs 2>&1 | Select-Object -Last 6 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) { throw "step failed ($LASTEXITCODE): $name" }
    if ($expect -and -not (Test-Path -LiteralPath $expect)) { throw "step produced no output: $name -> $expect" }
}

if (-not (Test-Path "$env:DATASET_ROOT\$Category\train\good")) { throw "dataset missing: $env:DATASET_ROOT\$Category" }
Push-Location $root
try {
    New-Item -ItemType Directory -Force (Split-Path $bb) | Out-Null
    Step 'prep_data (test + 100 calib images)' @('-I', 'export\prep_data.py', $Category, "work\${Category}_$Size", "$Size") "$w\calib.txt"
    if (-not (Test-Path $bb)) { Step 'export_backbone' @('-I', '-W', 'ignore', 'export\export_backbone.py', 'wide_resnet50_2', "$Size", $bb) $bb }
    $env:HEAD_SEED = '1'
    Step 'make_knn_head (CUDA coreset)' @('-W', 'ignore', 'export\make_knn_head.py', "$Dim", "$Ratio", 'random', $Category) $null
    Remove-Item Env:\HEAD_SEED
    Step 'make_fused' @('-I', '-W', 'ignore', 'export\make_fused.py', $tag, $Category) "$w\fused_$tag.onnx"
    Step 'rewrite_knn_reduce' @('-I', '-W', 'ignore', 'export\rewrite_knn_reduce.py', "$w\fused_$tag.onnx", "$w\fused_${tag}_rmax.onnx") "$w\fused_${tag}_rmax.onnx"
    if ($Dim -eq 1536) {
        # No projection exists at 1536 channels, so there is nothing to move in front of the upsample: reuse the rmax graph.
        Copy-Item "$w\fused_${tag}_rmax.onnx" "$w\fused_${tag}_rmax_pf.onnx" -Force
        Write-Host '    dim 1536: rewrite_proj_first skipped, rmax graph copied'
    }
    else {
        Step 'rewrite_proj_first' @('-I', '-W', 'ignore', 'export\rewrite_proj_first.py', "$w\fused_${tag}_rmax.onnx", "$w\fused_${tag}_rmax_pf.onnx") "$w\fused_${tag}_rmax_pf.onnx"
    }
    Get-ChildItem $w -Filter *.onnx | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } } | Format-Table -AutoSize | Out-String | Write-Host
    Write-Host 'RESULT=SUCCESS'
}
catch { Write-Host "ERROR: $($_.Exception.Message)"; Write-Host 'RESULT=FAILED'; exit 1 }
finally { Pop-Location }
