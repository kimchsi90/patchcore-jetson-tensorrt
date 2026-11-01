# Accuracy of the pipelined mode: run the 83 frames at 900x900 through bench_e2e.py --pipelined --mode acc on the board, copy the npz, score on the PC.
# Run only when the board is idle. Usage: pwsh e2e_pipe_acc.ps1
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$py = 'c:\dev_venvs\PatchcoreJetson\Scripts\python.exe'
$B = '<user>@<board-ip>'
$dir = 'D:\jetson_test\trt\out\e2e_pipe_acc'
New-Item -ItemType Directory -Force $dir | Out-Null
function Run-Native([string]$exe, [string[]]$a, [int]$sec = 300) {
    $o = [IO.Path]::GetTempFileName()
    $q = $a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process $exe -ArgumentList $q -PassThru -NoNewWindow -RedirectStandardOutput $o -RedirectStandardError "$o.err"
    if (-not $p.WaitForExit($sec * 1000)) { taskkill /F /T /PID $p.Id 2>&1 | Out-Null; Write-Host "  timeout after $sec s: $exe $($a[-1])" }
    Remove-Item $o, "$o.err" -ErrorAction SilentlyContinue
}
# name | pipe | interp | workers (board paths) | number of workers used
$cfg = @(
    @('pipe_P4_dla0_area',   'P4', 'area',   'dla0=engines/int8mix_dla0.engine', 1),
    @('pipe_P4_dla0_linear', 'P4', 'linear', 'dla0=engines/int8mix_dla0.engine', 1),
    @('pipe_P1_gpu_area',    'P1', 'area',   'gpu=engines/p1_gpu_post.engine', 1),
    @('pipe_P1_gpu_gpu2_area', 'P1', 'area', 'gpu=engines/p1_gpu_post.engine,gpu2=engines/p1_gpu_post.engine', 2),
    @('pipe_P4_dla0_dla1_pt2_area', 'P4', 'area', 'dla0=engines/int8mix_dla0.engine,dla1=engines/int8mix_dla0.engine', 2)
)
$items = @()
foreach ($c in $cfg) {
    $n = $c[0]
    $extra = if ($n -like '*pt2*') { '--post-threads 2' } else { '' }
    Run-Native 'ssh' @('-o', 'BatchMode=yes', $B, "cd ~/e2e && mkdir -p acc && python3 bench_e2e.py --mode acc --pipelined --acc-workers $($c[4]) --name $n --pipe $($c[1]) --interp $($c[2]) --workers $($c[3]) $extra --out acc/$n.npz > acc/$n.log 2>&1; tail -2 acc/$n.log") 240
    Run-Native 'scp' @('-o', 'BatchMode=yes', "${B}:e2e/acc/$n.npz", "$dir\$n.npz") 120
    if (-not (Test-Path "$dir\$n.npz")) { Write-Host "$n missing npz"; continue }
    $items += "$n=$dir\$n.npz"
}
Set-Location 'D:\jetson_test\trt\pc'
& $py -I e2e_score.py @items --out 'D:\jetson_test\trt\out\e2e_pipe_accuracy.json'
