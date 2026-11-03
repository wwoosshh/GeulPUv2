param([ValidateSet('build_row1','measure_npu','phase_bench','bench_joined','paired_joined','bench_multi','bench_multi_model','bench_parallel_post','bench_reuse_overlap','bench_long_overlap','bench_batch_capacity','check_adaptive_reuse','study_partial_range','test_packed_partials','test_packed_partials_v2','test_packed_partials_v3','diagnose_transfer','test_retained_device','test_output_view','check_adaptive_output_view','check_model_integration','compare_systems')][string]$Mode='measure_npu')
$ErrorActionPreference='Stop'
$root=(Resolve-Path "$PSScriptRoot/../../..").Path
$vs = & 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe' -latest -products '*' -property installationPath
& "$vs/Common7/Tools/Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
. "$root/빌드/npu/dev/mlir-aie/iron_env.ps1"
& "$root/빌드/npu/dev/mlir-aie/ironenv/Scripts/python.exe" "$PSScriptRoot/$Mode.py"
exit $LASTEXITCODE
