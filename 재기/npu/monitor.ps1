param(
    [Parameter(Mandatory=$true)][int]$TargetProcessId,
    [Parameter(Mandatory=$true)][string]$OutputFile,
    [int]$Seconds = 180
)
# Process memory is host memory, not NPU SRAM or total system energy.
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputFile) { throw 'Choose a new telemetry output file.' }
$npuWatch = [System.Diagnostics.Stopwatch]::StartNew()
while ($npuWatch.Elapsed.TotalSeconds -lt $Seconds) {
    $npuProcess = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
    if (-not $npuProcess) { break }
    [ordered]@{
        at = [DateTimeOffset]::UtcNow.ToString('o')
        elapsed_seconds = $npuWatch.Elapsed.TotalSeconds
        process_id = $TargetProcessId
        working_set_bytes = $npuProcess.WorkingSet64
        private_bytes = $npuProcess.PrivateMemorySize64
        cpu_seconds = $npuProcess.TotalProcessorTime.TotalSeconds
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $OutputFile -Encoding utf8
    Start-Sleep -Milliseconds 1000
}
