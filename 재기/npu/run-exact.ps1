param(
    [ValidateSet('kernel','boundary','model','unit','tiled-test','planned-test','fusion-test')][string]$Mode = 'boundary',
    [ValidateSet('dot','tiled','planned','fused')][string]$Backend = 'dot',
    [string]$Tokens = '1,2,3',
    [int]$Repeats = 2,
    [string]$Output = '빌드/npu/results/exact/model.jsonl'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot/../..").Path
Set-Location -LiteralPath $root
$vswhere = 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -property installationPath
& "$vs/Common7/Tools/Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
. "$root/빌드/npu/dev/mlir-aie/iron_env.ps1"
# Native compiler diagnostics use the Windows locale. UTF-8 mode breaks their decoding.
$env:PYTHONUTF8 = '0'
$python = "$root/빌드/npu/dev/mlir-aie/ironenv/Scripts/python.exe"
switch ($Mode) {
    kernel { & $python "$PSScriptRoot/exact_kernel.py" }
    boundary { & $python "$PSScriptRoot/test_npu_contract.py" }
    unit { & $python "$PSScriptRoot/test_exact_linear.py" }
    model { & $python "$PSScriptRoot/trace_model.py" "$root/빌드/npu/models/Qwen3-0.6B-Q8_0.gguf" --tokens $Tokens --repeats $Repeats --output $Output --backend $Backend }
    tiled-test { & $python "$PSScriptRoot/test_tiled.py" }
    planned-test { & $python "$PSScriptRoot/test_planned.py" }
    fusion-test { & $python "$PSScriptRoot/test_fused.py" }
}
if ($LASTEXITCODE -ne 0) { throw "Exact NPU test failed: $LASTEXITCODE" }
