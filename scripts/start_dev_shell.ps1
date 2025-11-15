param(
    [switch]$NewWindow
)

$repoRoot = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$repoRoot = Split-Path -Path $repoRoot -Parent
$venvActivate = Join-Path $repoRoot '.venv\Scripts\Activate.ps1'
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$pipExe = Join-Path $repoRoot '.venv\Scripts\pip.exe'

if (!(Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found at $venvActivate"
    exit 1
}

$initScript = @"
param(
    [string]`$repoRoot,
    [string]`$venvActivate,
    [string]`$pythonExe,
    [string]`$pipExe
)
Set-Location `$repoRoot
. `$venvActivate
Remove-Item Alias:python -ErrorAction SilentlyContinue
Remove-Item Alias:pip -ErrorAction SilentlyContinue
Set-Alias -Name python -Value `$pythonExe
Set-Alias -Name pip -Value `$pipExe
Write-Host "Dev shell ready. python -> `$pythonExe"
Write-Host "Repo root: `$repoRoot"
"@

if ($NewWindow) {
    $tempFile = New-TemporaryFile
    Set-Content -Path $tempFile -Value $initScript -Encoding UTF8
    Start-Process pwsh -ArgumentList "-NoExit", "-File", $tempFile, $repoRoot, $venvActivate, $pythonExe, $pipExe
} else {
    $scriptBlock = [ScriptBlock]::Create($initScript)
    & $scriptBlock $repoRoot $venvActivate $pythonExe $pipExe
}
