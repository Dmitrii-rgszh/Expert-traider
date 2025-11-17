param(
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8000
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

$pythonPath = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Error "Python venv not found at $pythonPath. Create .venv first."
    exit 1
}

Write-Host "Starting backend (uvicorn)..." -ForegroundColor Cyan
Start-Process -FilePath $pythonPath `
    -ArgumentList "-m uvicorn backend.app.main:app --host $BackendHost --port $BackendPort" `
    -WorkingDirectory $repoRoot `
    -WindowStyle Minimized

Start-Sleep -Seconds 5

Write-Host "Starting Flet desktop client..." -ForegroundColor Cyan
Start-Process -FilePath $pythonPath `
    -ArgumentList "desktop_client/flet_main.py" `
    -WorkingDirectory $repoRoot `
    -WindowStyle Normal

Write-Host "Super-trader backend and Flet client started." -ForegroundColor Green
