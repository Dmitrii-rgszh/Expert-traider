@echo off
setlocal
set "VENV_PY=%~dp0\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [python.bat] Virtual environment interpreter not found at %VENV_PY%
    exit /b 1
)
"%VENV_PY%" %*
