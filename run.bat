@echo off
REM Flipkart Review API - Windows Batch Launcher
setlocal

cd /d "%~dp0"

REM Check for Python in PATH or py launcher
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=py -3"
    goto :run
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=python"
    goto :run
)

echo [-] Error: Python is not installed or not in PATH.
echo     Please install Python 3.9+ from https://www.python.org/
echo     Ensure you check the box "Add Python to PATH" during installation.
pause
exit /b 1

:run
%PYTHON_CMD% run.py %*
if %ERRORLEVEL% neq 0 (
    pause
)
