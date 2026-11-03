param(
    [string]$Model = 'qwen3:0.6b',
    [ValidateSet('performance','balanced','powersaver','turbo')][string]$PowerMode = 'performance',
    [int]$Context = 32768,
    [int]$Repeat = 5,
    [int]$Port = 44560
)
$ErrorActionPreference = 'Stop'
$npuRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$npuExe = Join-Path $npuRoot '빌드/npu/flm-1.0.10/flm.exe'
$npuStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$npuOut = Join-Path $npuRoot "빌드/npu/results/$npuStamp-$PowerMode"
$npuPreviousModels = $env:FLM_MODEL_PATH
$env:FLM_MODEL_PATH = Join-Path $npuRoot '빌드/npu/models'
$npuServer = $null
try {
    if (!(Test-Path -LiteralPath $npuExe)) { throw 'Download and extract the pinned FLM release first; see README.md.' }
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already in use." }
    New-Item -ItemType Directory -Path $npuOut | Out-Null
    $npuValidation = & $npuExe validate --json
    $npuValidation | Set-Content -Encoding utf8 (Join-Path $npuOut 'validation.json')
    if ($LASTEXITCODE -ne 0 -or !(($npuValidation | ConvertFrom-Json).ready)) { throw 'NPU validation failed.' }
    $npuInstalled = (& $npuExe list --filter installed --json | ConvertFrom-Json).models
    if (!($npuInstalled | Where-Object { $_.model -eq $Model })) { throw "Model is not installed: $Model. Run flm pull explicitly first." }
    $npuMeta = @{
        runtime='FastFlowLM 1.0.10'; backend='flm NPU'; pmode=$PowerMode; context=$Context
        repository_commit=(git -C $npuRoot rev-parse HEAD)
        harness_hash=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'bench.mjs') -Algorithm SHA256).Hash
        request_options_hash=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'flm-greedy.json') -Algorithm SHA256).Hash
        cpu=(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
        computer=(Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory)
        npu_driver=(Get-CimInstance Win32_PnPSignedDriver | Where-Object DeviceName -match 'NPU Compute' | Select-Object DeviceName,DriverVersion)
        battery=(Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining)
        power_scheme=(powercfg /getactivescheme)
        runtime_hash=(Get-FileHash -LiteralPath $npuExe -Algorithm SHA256).Hash
        model_files=(Get-ChildItem -LiteralPath $env:FLM_MODEL_PATH -Recurse -File | Get-FileHash -Algorithm SHA256 | Select-Object Path,Hash)
        kernels=(Get-ChildItem -LiteralPath (Split-Path $npuExe) -Filter '*npu*.dll' | Get-FileHash -Algorithm SHA256 | Select-Object Path,Hash)
    }
    $npuMetaPath = Join-Path $npuOut 'metadata.json'
    $npuMeta | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $npuMetaPath
    $npuServer = Start-Process -FilePath $npuExe -ArgumentList @('serve',$Model,'--host','127.0.0.1','--port',$Port,'--ctx-len',$Context,'--pmode',$PowerMode,'--cors','0') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $npuOut 'server.log') -RedirectStandardError (Join-Path $npuOut 'server-error.log')
    $npuReady = $false
    for ($npuAttempt = 0; $npuAttempt -lt 60; $npuAttempt++) {
        if ($npuServer.HasExited) { throw 'Server exited; inspect server-error.log.' }
        try { Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2 | Out-Null; $npuReady=$true; break } catch { Start-Sleep -Seconds 1 }
    }
    if (!$npuReady) { throw 'Server readiness timeout.' }
    & node (Join-Path $PSScriptRoot 'bench.mjs') --url "http://127.0.0.1:$Port" --model $Model --out $npuOut --metadata $npuMetaPath --repeat $Repeat --request-options (Join-Path $PSScriptRoot 'flm-greedy.json')
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed; inspect $npuOut" }
    Write-Output "Results: $npuOut"
} finally {
    if ($npuServer -and !$npuServer.HasExited) { Stop-Process -Id $npuServer.Id }
    $env:FLM_MODEL_PATH = $npuPreviousModels
}
