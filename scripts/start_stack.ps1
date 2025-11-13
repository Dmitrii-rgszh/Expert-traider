param(
    [switch]$SkipDb
)

# Поднимает cloudflared, backend (uvicorn), фронтенд (Vite) и Telegram-бота.
# По умолчанию дополнительно стартует Postgres из docker-compose. Чтобы пропустить — используйте -SkipDb.

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backendDir = $root
$frontendDir = Join-Path $root "frontend"
$pythonPath = Join-Path $root ".venv\Scripts\python.exe"
$dockerComposeFile = Join-Path $root "docker-compose.yml"
$backendEnvPath = Join-Path $root "backend/.env"
$rootEnvPath = Join-Path $root ".env"

function Test-CommandAvailable {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Import-EnvFile {
    param([string]$FilePath)
    if (-not (Test-Path $FilePath)) {
        return
    }
    Write-Host "Загружаю переменные окружения из $FilePath" -ForegroundColor Yellow
    Get-Content $FilePath | ForEach-Object {
        $line = $_.Trim()
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
            return
        }
        $parts = $line -split "=", 2
        if ($parts.Length -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
        }
    }
}

function Stop-StackProcesses {
    $patterns = @(
        'cloudflared tunnel run auto-trader',
        'uvicorn backend.app.main:app',
        'bot/main.py',
        'vite 127.0.0.1 5173',
        'vite 0.0.0.0 5173',
        'vite 5173',
        'vite\\bin\\vite.js'
    )
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        if (-not $cmd) { return $false }
        foreach ($pattern in $patterns) {
            if ($cmd -like "*$pattern*") {
                return $true
            }
        }
        return $false
    }
    if ($procs) {
        Write-Host "Останавливаю ранее запущенные процессы стека..." -ForegroundColor Yellow
        foreach ($proc in $procs) {
            try {
                Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
                Write-Host "  остановлен PID $($proc.ProcessId)" -ForegroundColor Yellow
            }
            catch {
                Write-Host "  не удалось остановить PID $($proc.ProcessId): $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}

Import-EnvFile -FilePath $rootEnvPath
Import-EnvFile -FilePath $backendEnvPath
Stop-StackProcesses

$webAppUrl = $env:WEBAPP_URL
if (-not $webAppUrl) {
    $webAppUrl = "https://app.glazok.site"
    Write-Host "WEBAPP_URL не задана. Использую $webAppUrl" -ForegroundColor Yellow
}

if (-not $SkipDb) {
    if (-not (Test-CommandAvailable "docker")) {
        throw "Команда 'docker' недоступна в PATH. Установите Docker Desktop или добавьте docker в PATH."
    }
    Write-Host "Запускаю docker-compose (db)..." -ForegroundColor Yellow
    try {
        docker compose -f $dockerComposeFile up -d db | Out-Null
        Write-Host "Postgres поднят через docker-compose." -ForegroundColor Green
    }
    catch {
        Write-Host "Не удалось запустить docker-compose для db: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Убедитесь, что порт 5432 свободен и Docker Desktop запущен." -ForegroundColor Red
        throw
    }
}
else {
    Write-Host "Флаг -SkipDb: пропускаю запуск docker-compose." -ForegroundColor Yellow
}

if (-not (Test-CommandAvailable "cloudflared")) {
    throw "Команда 'cloudflared' недоступна в PATH. Установите Cloudflared или добавьте его в PATH."
}
if (-not (Test-CommandAvailable "npm")) {
    throw "Команда 'npm' недоступна в PATH. Установите Node.js с npm."
}
if (-not (Test-Path $pythonPath)) {
    throw "Не найден интерпретатор виртуального окружения по пути $pythonPath"
}

Write-Host "Запускаю cloudflared туннель..." -ForegroundColor Yellow
$cloudflaredJob = Start-Process pwsh -ArgumentList @(
    "-NoLogo",
    "-Command",
    "Set-Location `"$root`"; cloudflared tunnel run auto-trader"
) -WorkingDirectory $root -WindowStyle Minimized -PassThru

Write-Host "Запускаю backend (uvicorn)..." -ForegroundColor Yellow
$backendJob = Start-Process pwsh -ArgumentList @(
    "-NoLogo",
    "-Command",
    "Set-Location `"$backendDir`"; `"$pythonPath`" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
) -WorkingDirectory $backendDir -WindowStyle Minimized -PassThru

Write-Host "Запускаю фронтенд (Vite dev server)..." -ForegroundColor Yellow
$frontendJob = Start-Process pwsh -ArgumentList @(
    "-NoLogo",
    "-Command",
    "Set-Location `"$frontendDir`"; `$env:VITE_HOST='127.0.0.1'; `$env:VITE_PORT='5173'; npm run dev"
) -WorkingDirectory $frontendDir -WindowStyle Minimized -PassThru

Write-Host "Запускаю Telegram-бота..." -ForegroundColor Yellow
$botJob = Start-Process pwsh -ArgumentList @(
    "-NoLogo",
    "-Command",
    "Set-Location `"$root`"; `"$pythonPath`" bot/main.py"
) -WorkingDirectory $root -WindowStyle Minimized -PassThru

$tunneledApiUrl = "{0}/api" -f ($webAppUrl.TrimEnd('/'))

Write-Host "Стек запущен. PID процессов:" -ForegroundColor Green
Write-Host "  cloudflared  PID: $($cloudflaredJob.Id)"
Write-Host "  backend      PID: $($backendJob.Id)"
Write-Host "  frontend     PID: $($frontendJob.Id)"
Write-Host "  bot          PID: $($botJob.Id)"
Write-Host ""
Write-Host "URL мини-приложения: $webAppUrl" -ForegroundColor Green
Write-Host "API через туннель: $tunneledApiUrl" -ForegroundColor Green
Write-Host "Остановите сервисы через Stop-Process -Id <PID> или закройте окно PowerShell." -ForegroundColor Green
Write-Host "Кнопка в Telegram Mini App должна указывать на $webAppUrl" -ForegroundColor Green
