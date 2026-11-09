#!/usr/bin/env bash
# Flipkart Review API - macOS / Linux Launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find python interpreter
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[-] Error: Python is not installed or not in PATH."
    echo "    Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

# Run the cross-platform bootstrap script
exec "$PYTHON_CMD" run.py "$@"
