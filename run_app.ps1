param([int]$Port = 8502)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython -m streamlit run app/streamlit_app.py --server.port $Port --server.headless true --server.address 127.0.0.1
} else {
    $workspaceRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)
    $preparedDependencies = @(
        (Join-Path $workspaceRoot 'work\optimind_deps'),
        (Join-Path $workspaceRoot 'work\deps')
    ) | Where-Object { Test-Path -LiteralPath $_ }
    if ($preparedDependencies.Count -gt 0) {
        $env:PYTHONPATH = ($preparedDependencies -join ';') + ';' + $env:PYTHONPATH
    }
    & python -m streamlit run app/streamlit_app.py --server.port $Port --server.headless true --server.address 127.0.0.1
}
