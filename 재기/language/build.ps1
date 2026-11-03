$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot/../..").Path
Set-Location -LiteralPath $root
$python = "$root/빌드/npu/dev/mlir-aie/ironenv/Scripts/python.exe"
New-Item -ItemType Directory -Force "$root/빌드/language" | Out-Null
& $python -X utf8 "$root/geul-lang/ref/geulc.py" "$PSScriptRoot/bench.gl" -o "$root/빌드/language/geul.exe"
if ($LASTEXITCODE -ne 0) { throw 'GL compilation failed' }
$vs = & 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe' -latest -products '*' -property installationPath
& "$vs/Common7/Tools/Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
& cl /nologo /O2 /W4 "/Fe:$root/빌드/language/native.exe" "/Fo:$root/빌드/language/native.obj" "$PSScriptRoot/bench.c"
if ($LASTEXITCODE -ne 0) { throw 'C compilation failed' }
(Get-Item (Get-Command cl).Source).VersionInfo | Format-List FileVersion,ProductVersion | Out-File "$root/빌드/language/compiler-version.txt"
