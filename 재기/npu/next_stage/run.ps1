param([ValidateSet('build','post','device','device_vector','overlap','model','generation','generation_profile','host')][string]$Mode='post')
$ErrorActionPreference='Stop'
$root=(Resolve-Path "$PSScriptRoot/../../..").Path
Set-Location -LiteralPath $root
$vs = & 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe' -latest -products '*' -property installationPath
& "$vs/Common7/Tools/Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if ($Mode -eq 'build') {
    New-Item -ItemType Directory -Force '빌드/npu/next-stage' | Out-Null
    & cl /nologo /LD /O2 /fp:strict /MD "$PSScriptRoot/post.c" '/Fo:빌드/npu/next-stage/post.obj' /link '/OUT:빌드/npu/next-stage/post.dll' '/IMPLIB:빌드/npu/next-stage/post.lib'
} else {
    . "$root/빌드/npu/dev/mlir-aie/iron_env.ps1"
    $env:PYTHONUTF8='0'
    $python="$root/빌드/npu/dev/mlir-aie/ironenv/Scripts/python.exe"
    & $python "$PSScriptRoot/test_$Mode.py"
    if ($LASTEXITCODE -ne 0) { throw "Test failed: $LASTEXITCODE" }
    if ($Mode -eq 'host') {
        & $python -X utf8 "$root/geul-lang/ref/geulc.py" "$PSScriptRoot/host_smoke.gl" -o "$root/빌드/npu/next-stage/host.exe"
        if ($LASTEXITCODE -ne 0) { throw 'GL host compilation failed' }
        & $python "$PSScriptRoot/run_host.py"
    }
}
if ($LASTEXITCODE -ne 0) { throw "Next stage failed: $LASTEXITCODE" }
