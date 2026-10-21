# GeulPU 앱 꾸리기(36단계) — 소스에서 릴리스 폴더와 zip 을 만든다. 릴리스 CI(.github/workflows/release.yml)가 이것을 그대로 돌리고,
# 이 PC 에서도 같은 것을 만들 수 있다(올리지 않는다).
#
#   앱\꾸리기.ps1 -GeulRoot <글 저장소(고정 커밋)> [-GeulC <글 컴파일러 exe>] [-Out <폴더>] [-NoZip] [-AllowHashChange]
#
# 1. 글 컴파일러: -GeulC 가 없으면 글 저장소에서 두 번 빌드한다(파이썬 참조 구현 → 글로 쓴 컴파일러 → 같은 소스로 한 번 더 — 고정점).
# 2. 뒷단 둘(PTX · SPIR-V)은 글 저장소에서, 나머지는 이 저장소에서 빌드한다:
#      GeulPU.exe(앱/관리자.gl) · bin\server-cuda · server-vulkan · device-vulkan · device-cuda · diag-cuda · diag-vulkan · kernel-gen ·
#      ptx-backend · spirv-backend, kernels\ 의 진단 커널(PTX · SPIR-V — 협력 모양 셋)
# 3. web\ · data\(받을 모델 목록 · 교차검증 기준) · LICENSE · README.txt 를 넣고 version.json(판 · 커밋 · 빌드 때 · 파일마다 SHA-256)을 쓴다.
# 4. 교차검증 기준(expected.json)에 있는 빌드 해시와 견준다 — 다르면 멈춘다(-AllowHashChange 면 경고만). 릴리스의 서버 · 생성기 · 뒷단이
#    기준 PC 에서 R1 32 문제를 비트까지 맞춘 빌드와 같은 바이트라는 것을 릴리스마다 확인한다.
# 5. GeulPU-<판>-windows-x64.zip 과 SHA256SUMS.txt.
param([string]$GeulRoot = '', [string]$GeulC = '', [string]$Out = '', [switch]$NoZip, [switch]$AllowHashChange)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path      # …\GeulPU\앱
$Root = Split-Path -Parent $Here                               # …\GeulPU
if (-not $Out) { $Out = Join-Path $Root '빌드\앱' }
if (-not $GeulRoot) { Write-Host '-GeulRoot <글 저장소> 가 필요합니다 (고정 커밋: 시험\글저장소커밋.txt)' -ForegroundColor Red; exit 1 }
$GeulRoot = (Resolve-Path -LiteralPath $GeulRoot).Path
New-Item -ItemType Directory -Path $Out -Force | Out-Null
$Out = (Resolve-Path -LiteralPath $Out).Path
$Logs = Join-Path $Out 'logs'
New-Item -ItemType Directory -Path $Logs -Force | Out-Null

function Say([string]$m) { Write-Host ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $m) }
function Fail([string]$m) { Write-Host ('멈춤: ' + $m) -ForegroundColor Red; exit 1 }
function Sha([string]$p) { return (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() }
function Quote([string]$s) { if ($s -match '[\s"]') { return '"' + ($s -replace '"', '\"') + '"' }; return $s }
function Run([string]$exe, [string[]]$argv, [string]$cwd, [string]$name) {
    $o = Join-Path $Logs ($name + '.log'); $e = Join-Path $Logs ($name + '-err.log')
    $p = Start-Process -FilePath $exe -ArgumentList ($argv | ForEach-Object { Quote $_ }) -WorkingDirectory $cwd -NoNewWindow -PassThru `
        -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $p.Handle
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) { Get-Content -LiteralPath $e -ErrorAction SilentlyContinue | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }; Fail ($name + ' 실패 (끝 코드 ' + $p.ExitCode + ') — ' + $o) }
}

# ---- 판 · 커밋 ----
$panGl = [IO.File]::ReadAllText((Join-Path $Here '판.gl'), [Text.Encoding]::UTF8)
if ($panGl -notmatch '앱판 = "([0-9]+\.[0-9]+\.[0-9]+[^"]*)"') { Fail '앱/판.gl 에서 판을 읽지 못했습니다' }
$Ver = $Matches[1]
$Commit = ''
$Dirty = $false
try {
    $Commit = (& git -c safe.directory=* -C $Root rev-parse HEAD 2>$null) -join ''
    $Dirty = @(& git -c safe.directory=* -C $Root status --porcelain -- 엔진 커널 앱 교차검증/expected.json 교차검증/requests.json 2>$null).Count -gt 0
} catch { }
$Pin = ([IO.File]::ReadAllText((Join-Path $Root '시험\글저장소커밋.txt'))).Trim()
$GeulHead = ''
try { $GeulHead = (& git -c safe.directory=* -C $GeulRoot rev-parse HEAD 2>$null) -join '' } catch { }
Say ("GeulPU {0} · 커밋 {1}{2} · 글 저장소 {3} (고정 커밋 {4})" -f $Ver, $Commit, $(if ($Dirty) { ' (고친 파일 있음)' } else { '' }), $GeulHead, $Pin)
if ($GeulHead -and $GeulHead -ne $Pin) { Write-Host '경고: 글 저장소가 고정 커밋이 아닙니다 — 빌드 해시가 기준과 다를 수 있습니다' -ForegroundColor Yellow }

[Environment]::SetEnvironmentVariable('GEUL_ROOT', $GeulRoot, 'Process')
[Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'Process')
$Tmp = Join-Path $Out 'tmp'
New-Item -ItemType Directory -Path $Tmp -Force | Out-Null

# ---- 1. 글 컴파일러(고정점) ----
if ($GeulC) { $Comp = (Resolve-Path -LiteralPath $GeulC).Path; Say ('글 컴파일러: ' + $Comp) }
else {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { Fail '-GeulC 가 없고 파이썬도 없습니다 (글 컴파일러를 빌드할 수 없음)' }
    Say '글 컴파일러를 빌드합니다 (파이썬 참조 구현 → 글로 쓴 컴파일러 → 고정점)...'
    $s1 = Join-Path $Tmp 'geulc_s1.exe'
    Run $py.Source @('ref/geulc.py', 'self/컴파일러.gl', '-o', $s1) $GeulRoot 'compiler-s1'
    $Comp = Join-Path $Tmp 'geulc.exe'
    Run $s1 @('self/컴파일러.gl', '-o', $Comp) $GeulRoot 'compiler-s2'
}

# ---- 2. 빌드 ----
$Dist = Join-Path $Out ('GeulPU-' + $Ver)
if (Test-Path -LiteralPath $Dist) { Remove-Item -LiteralPath $Dist -Recurse -Force }
foreach ($d in @('bin', 'kernels', 'web', 'data', 'data\crosscheck')) { New-Item -ItemType Directory -Path (Join-Path $Dist $d) -Force | Out-Null }
$Bin = Join-Path $Dist 'bin'
Say 'PTX · SPIR-V 백엔드...'
Run $Comp @('백엔드/PTX/PTX.gl', '-o', (Join-Path $Bin 'ptx-backend.exe')) $Root 'ptx-backend'           # 이 저장소의 백엔드 (앞단은 GEUL_ROOT 의 self/)
Run $Comp @('백엔드/SPIRV/SPIRV.gl', '-o', (Join-Path $Bin 'spirv-backend.exe')) $Root 'spirv-backend'
$progs = @(
    @('앱/관리자.gl', (Join-Path $Dist 'GeulPU.exe'), 'app'),
    @('엔진/서버.gl', (Join-Path $Bin 'server-cuda.exe'), 'server-cuda'),
    @('엔진/서버V.gl', (Join-Path $Bin 'server-vulkan.exe'), 'server-vulkan'),
    @('엔진/장치V.gl', (Join-Path $Bin 'device-vulkan.exe'), 'device-vulkan'),
    @('엔진/장치C.gl', (Join-Path $Bin 'device-cuda.exe'), 'device-cuda'),
    @('엔진/진단.gl', (Join-Path $Bin 'diag-cuda.exe'), 'diag-cuda'),
    @('엔진/진단V.gl', (Join-Path $Bin 'diag-vulkan.exe'), 'diag-vulkan'),
    @('커널/커널생성.gl', (Join-Path $Bin 'kernel-gen.exe'), 'kernel-gen')
)
foreach ($p in $progs) { Say ('빌드: ' + $p[0]); Run $Comp @($p[0], '-o', $p[1]) $Root $p[2] }
Say '진단 커널(PTX · SPIR-V)...'
$K = Join-Path $Dist 'kernels'
Run (Join-Path $Bin 'ptx-backend.exe') @('커널/진단.gl', '-o', (Join-Path $K 'diag.ptx')) $Root 'kernel-diag-ptx'
Run (Join-Path $Bin 'spirv-backend.exe') @('커널/진단V.gl', '-o', (Join-Path $K 'diag-vulkan.spvs')) $Root 'kernel-diag-vulkan'
foreach ($s in @('16x16x32', '16x8x32', '16x16x16')) {
    Run (Join-Path $Bin 'spirv-backend.exe') @('커널/진단협력.gl', '-o', (Join-Path $K ('diag-vulkan-coop-' + $s + '.spvs')), '--협력', $s) $Root ('kernel-diag-coop-' + $s)
}

# ---- 3. 화면 · 자료 ----
Copy-Item -Path (Join-Path $Here 'web\*') -Destination (Join-Path $Dist 'web') -Recurse
Copy-Item -LiteralPath (Join-Path $Here 'data\models.json') -Destination (Join-Path $Dist 'data')
Copy-Item -LiteralPath (Join-Path $Root '교차검증\expected.json') -Destination (Join-Path $Dist 'data\crosscheck')
Copy-Item -LiteralPath (Join-Path $Root '교차검증\requests.json') -Destination (Join-Path $Dist 'data\crosscheck')
Copy-Item -LiteralPath (Join-Path $Root 'LICENSE') -Destination $Dist
Copy-Item -LiteralPath (Join-Path $Here '읽어보기.txt') -Destination (Join-Path $Dist 'README.txt')

# ---- 4. 기준 해시와 견주기 ----
$exp = [IO.File]::ReadAllText((Join-Path $Root '교차검증\expected.json'), [Text.Encoding]::UTF8) | ConvertFrom-Json
$checks = @(
    @('generator', (Join-Path $Bin 'kernel-gen.exe'), $exp.build.generator),
    @('ptx_backend', (Join-Path $Bin 'ptx-backend.exe'), $exp.build.ptx_backend),
    @('server_r1 (server-cuda)', (Join-Path $Bin 'server-cuda.exe'), $exp.build.server_r1),
    @('spirv_backend', (Join-Path $Bin 'spirv-backend.exe'), $exp.build_vulkan.spirv_backend),
    @('server_r1_vulkan (server-vulkan)', (Join-Path $Bin 'server-vulkan.exe'), $exp.build_vulkan.server_r1_vulkan)
)
$bad = 0
foreach ($c in $checks) {
    $h = Sha $c[1]
    if (-not $c[2]) { Write-Host ('  [정보] {0}: 기준 없음 {1}' -f $c[0], $h.Substring(0, 12)) }
    elseif ($h -eq $c[2]) { Write-Host ('  [같음] {0}: {1}' -f $c[0], $h.Substring(0, 12)) -ForegroundColor Green }
    else { Write-Host ('  [다름] {0}: {1} (기준 {2})' -f $c[0], $h.Substring(0, 12), $c[2].Substring(0, 12)) -ForegroundColor Red; $bad++ }
}
if ($bad -gt 0 -and -not $AllowHashChange) { Fail ('빌드 해시 {0} 개가 교차검증 기준과 다릅니다 — 기준을 새로 적었는지 보세요(-AllowHashChange 로 넘김)' -f $bad) }

# ---- 5. version.json · zip ----
$files = [ordered]@{}
Get-ChildItem -LiteralPath $Dist -Recurse -File | Sort-Object FullName | ForEach-Object {
    $rel = $_.FullName.Substring($Dist.Length + 1).Replace('\', '/')
    $files[$rel] = Sha $_.FullName
}
$verObj = [ordered]@{ name = 'GeulPU'; version = $Ver; commit = $Commit; dirty = $Dirty; built = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ');
                      geul_commit = $GeulHead; platform = 'windows-x64'; files = $files }
[IO.File]::WriteAllText((Join-Path $Dist 'version.json'), ($verObj | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))
Remove-Item -LiteralPath $Tmp -Recurse -Force -ErrorAction SilentlyContinue
if ($NoZip) { Say ('꾸림: ' + $Dist); exit 0 }
$zip = Join-Path $Out ('GeulPU-' + $Ver + '-windows-x64.zip')
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($Dist, $zip, [IO.Compression.CompressionLevel]::Optimal, $true)
$zsha = Sha $zip
[IO.File]::WriteAllText((Join-Path $Out 'SHA256SUMS.txt'), ('{0}  {1}' -f $zsha, (Split-Path -Leaf $zip)) + "`n", (New-Object System.Text.UTF8Encoding($false)))
Say ('zip: {0} ({1:N1} MB) · SHA-256 {2}' -f $zip, ((Get-Item -LiteralPath $zip).Length / 1MB), $zsha)
