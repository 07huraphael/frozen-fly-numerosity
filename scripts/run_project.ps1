param(
    [ValidateSet('smoke', 'pilot', 'prepare')][string]$Mode = 'pilot',
    [switch]$IncludeC
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try {
    $env:OPENBLAS_NUM_THREADS = '2'
    $env:OMP_NUM_THREADS = '2'
    $env:MPLCONFIGDIR = Join-Path $taskRoot '.mpl-cache'
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        uv --cache-dir .uv-cache venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
    .venv\Scripts\python.exe -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Scientific invariant checks failed.' }
    if ($Mode -ne 'smoke') {
        python scripts\fetch_data.py
        if ($LASTEXITCODE -ne 0) { throw 'Data download/checksum verification failed.' }
        if (-not (Test-Path -LiteralPath 'data\processed\flywire_visual.npz')) {
            .venv\Scripts\python.exe -m flynum prepare
            if ($LASTEXITCODE -ne 0) { throw 'Connectome preparation failed.' }
        }
    }
    if ($Mode -ne 'prepare') {
        $taskStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $taskConfig = if ($Mode -eq 'smoke') { 'configs\smoke.json' } else { 'configs\pilot.json' }
        $taskCommand = if ($Mode -eq 'smoke') { 'demo' } else { 'run' }
        $taskArgs = @('-m', 'flynum', $taskCommand, '--config', $taskConfig, '--out', "results\$Mode-$taskStamp")
        if ($IncludeC) { $taskArgs += '--include-c' }
        & '.venv\Scripts\python.exe' @taskArgs
        if ($LASTEXITCODE -ne 0) { throw 'Experiment execution failed.' }
    }
} finally {
    Pop-Location
}
