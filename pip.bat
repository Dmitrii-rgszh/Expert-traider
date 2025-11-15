@echo off
setlocal
set "VENV_PIP=%~dp0\.venv\Scripts\pip.exe"
if not exist "%VENV_PIP%" (
    echo [pip.bat] Virtual environment pip not found at %VENV_PIP%
    exit /b 1
)
"%VENV_PIP%" %*
