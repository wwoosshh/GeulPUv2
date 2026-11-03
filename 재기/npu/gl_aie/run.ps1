param([ValidateSet('build','bench','test','model')][string]$Mode='bench', [ValidateSet('gl','c','c_unrolled','aie_cpp')][string]$Backend='gl')
$ErrorActionPreference='Stop'
$root=(Resolve-Path "$PSScriptRoot/../../..").Path
Set-Location -LiteralPath $root
$vs = & 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe' -latest -products '*' -property installationPath
& "$vs/Common7/Tools/Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
. "$root/빌드/npu/dev/mlir-aie/iron_env.ps1"
$env:PYTHONUTF8='0'
$python="$root/빌드/npu/dev/mlir-aie/ironenv/Scripts/python.exe"
switch ($Mode) {
    build { & $python "$PSScriptRoot/build.py" }
    test { & $python "$PSScriptRoot/test_lower.py" }
    bench { & $python "$PSScriptRoot/bench.py" }
    model { & $python "$PSScriptRoot/model.py" --backend $Backend }
}
if ($LASTEXITCODE -ne 0) { throw "GL/AIE task failed: $LASTEXITCODE" }
