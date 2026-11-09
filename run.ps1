<#
.SYNOPSIS
    Flipkart Review API - Windows PowerShell Launcher
.DESCRIPTION
    Automatically sets up a virtual environment and runs the API or crawler on Windows.
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$pythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
    $pythonArgs = @("-3", "run.py") + $args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
    $pythonArgs = @("run.py") + $args
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
    $pythonArgs = @("run.py") + $args
} else {
    Write-Host "[-] Error: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "    Please install Python 3.9+ from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

& $pythonCmd $pythonArgs
