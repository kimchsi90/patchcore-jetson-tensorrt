# Resize-method check on the existing (Pillow-trained) banks for the categories the first study did not cover.
# For each category: test frames resized from the ORIGINALS with INTER_AREA and INTER_LINEAR, fp32 onnxruntime on the PC.
# Output: out\resize_study\results.csv (kind = mismatch). Reference (pil_aa) numbers are in p6_ort_summary.csv.
param([string]$Categories = 'cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,tile,transistor,zipper')
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$py = 'c:\dev_venvs\PatchcoreJetson\Scripts\python.exe'
Set-Location 'D:\jetson_test\trt\pc'
foreach ($c in ($Categories -split ',')) {
    $done = Select-String -Path 'D:\jetson_test\trt\out\resize_study\results.csv' -Pattern "^$c,mismatch,linear," -Quiet -ErrorAction SilentlyContinue
    if ($done) { Write-Host "$c already done"; continue }
    Write-Host ("{0:HH:mm:ss} {1}: start" -f (Get-Date), $c)
    & $py -W ignore resize_study.py mismatch --cat $c --methods area,linear 2>&1 | Select-Object -Last 3
    Write-Host ("{0:HH:mm:ss} {1}: done" -f (Get-Date), $c)
}
Write-Host 'RESIZE REST DONE'
