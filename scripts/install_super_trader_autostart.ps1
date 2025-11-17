param(
    [string]$TaskName = "AutoTraderSuperTrader"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$startScript = Resolve-Path (Join-Path $scriptDir "start_super_trader.ps1")

if (-not (Test-Path $startScript)) {
    Write-Error "start_super_trader.ps1 not found at $startScript"
    exit 1
}

$psExe = (Get-Command powershell.exe).Source
$action = "/Create /TN `"$TaskName`" /SC ONLOGON /RL HIGHEST /TR `"$psExe -ExecutionPolicy Bypass -File `"$startScript`"`""

Write-Host "Registering scheduled task '$TaskName' for autostart..." -ForegroundColor Cyan
Start-Process -FilePath schtasks.exe -ArgumentList $action -Verb RunAs -Wait

Write-Host "Scheduled task '$TaskName' created. Super-trader will start at user logon." -ForegroundColor Green

