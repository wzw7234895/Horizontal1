param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArguments
)
$ErrorActionPreference = 'Stop'
$pythonExe = 'D:\Anaconda\envs\crane-person\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Run setup_env.ps1 first or change $pythonExe to the environment Python.'
}
$env:PYTHONUTF8 = '1'
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot '.cache\ultralytics'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & $pythonExe -m person_tracking @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: exit $LASTEXITCODE" }
}
finally {
    Pop-Location
}
