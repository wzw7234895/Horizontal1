param(
    [string]$EnvironmentName = 'crane-person',
    [switch]$CpuOnly
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$cacheDir = Join-Path $PSScriptRoot '.cache'
$env:CONDA_PKGS_DIRS = Join-Path $cacheDir 'conda'
$env:TEMP = Join-Path $cacheDir 'temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
$condaExe = 'D:\Anaconda\Scripts\conda.exe'
if (-not (Test-Path -LiteralPath $condaExe)) { throw 'Set $condaExe to your conda executable.' }
$envList = (& $condaExe env list --json | ConvertFrom-Json).envs
$environmentPath = Join-Path 'D:\Anaconda\envs' $EnvironmentName
if ($envList -notcontains $environmentPath) {
    & $condaExe create -n $EnvironmentName python=3.11 pip -y
    if ($LASTEXITCODE -ne 0) {
        # This workstation already has a tested video environment. Cloning avoids
        # failures caused by missing packages in a channel's current index.
        & $condaExe create -n $EnvironmentName --clone camera-review --offline -y
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the conda environment.' }
        # The cloned conda OpenCV has no pip RECORD; unlink it through conda
        # before installing our tested pip wheel in this new environment.
        & $condaExe remove -n $EnvironmentName --force opencv --offline -y
        if ($LASTEXITCODE -ne 0) { throw 'Could not replace the cloned OpenCV package.' }
    }
}
$pythonExe = Join-Path $environmentPath 'python.exe'
$torchIndex = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu124' }
$torchWheel = Join-Path $cacheDir 'torch-2.5.1+cu124-cp311-cp311-win_amd64.whl'
$torchPackage = if ((-not $CpuOnly) -and (Test-Path -LiteralPath $torchWheel)) { $torchWheel } else { 'torch==2.5.1' }
& $pythonExe -m pip install --no-cache-dir $torchPackage torchvision==0.20.1 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $pythonExe -m pip install --no-cache-dir -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Push-Location (Split-Path $PSScriptRoot -Parent)
try { & $pythonExe -m person_tracking doctor }
finally { Pop-Location }
