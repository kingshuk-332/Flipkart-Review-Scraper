#!/usr/bin/env python3
"""
Cross-platform auto-bootstrapping runner for Flipkart Review API.
Works seamlessly on Windows, macOS, and Linux without requiring manual venv setup.

Usage:
    python run.py                 # Auto-setup venv & start API server
    python run.py serve           # Start API server
    python run.py crawl <args>    # Run Scrapy spider directly
    python run.py test            # Run unit tests
    python run.py setup           # Only setup/update virtual environment
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
STAMP_FILE = VENV_DIR / ".requirements.hash"


def get_venv_python() -> Path:
    """Return the platform-specific path to the python interpreter in .venv."""
    if os.name == "nt":  # Windows
        exe = VENV_DIR / "Scripts" / "python.exe"
    else:  # Linux / macOS
        exe = VENV_DIR / "bin" / "python"
    return exe


def is_running_in_project_venv() -> bool:
    """Check if the currently running Python is the project's .venv Python."""
    current_prefix = Path(sys.prefix).resolve()
    target_prefix = VENV_DIR.resolve()
    return current_prefix == target_prefix


def get_file_hash(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_virtualenv():
    """Ensure that the local virtual environment exists and is up to date."""
    venv_python = get_venv_python()

    # 1. Create virtual environment if missing
    if not venv_python.exists():
        print(f"[*] Creating virtual environment in {VENV_DIR}...")
        try:
            import venv
            venv.create(VENV_DIR, with_pip=True)
        except Exception:
            # Fallback to invoking python -m venv via subprocess
            subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("[+] Virtual environment created successfully.")

    # 2. Check and install dependencies if requirements.txt changed or uninstalled
    req_hash = get_file_hash(REQUIREMENTS_FILE)
    stamp_hash = STAMP_FILE.read_text().strip() if STAMP_FILE.exists() else ""

    if req_hash != stamp_hash or not stamp_hash:
        print("[*] Installing / updating dependencies from requirements.txt...")
        # Upgrade pip first silently, then install requirements
        subprocess.check_call(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call([
            str(venv_python),
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_FILE),
        ])
        if req_hash:
            STAMP_FILE.write_text(req_hash)
        print("[+] Dependencies are up to date.")


def run_in_venv():
    """Re-executes this script inside the project's virtual environment."""
    ensure_virtualenv()
    venv_python = get_venv_python()
    args = [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:]

    # On POSIX, replace current process; on Windows, use subprocess.call
    if os.name != "nt":
        os.execv(str(venv_python), args)
    else:
        sys.exit(subprocess.call(args))


def run_server():
    """Run the API server using gunicorn on POSIX or waitress/flask on Windows."""
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"\n[+] Starting Flipkart Review API on http://{host}:{port}")

    debug_mode = (
        os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
        or "--debug" in sys.argv
    )

    if debug_mode:
        print(f"[*] Running via Flask server in debug mode on http://{host}:{port}...")
        from app import app
        app.run(host=host, port=port, debug=True)
        return

    if os.name == "nt":  # Windows
        try:
            from waitress import serve
            from app import app

            print(f"[+] Production WSGI server (waitress) running on http://{host}:{port}")
            serve(app, host=host, port=port)
        except ImportError:
            print("[*] Running via Flask server...")
            from app import app
            app.run(host=host, port=port, debug=False)
    else:  # Linux / macOS
        try:
            import gunicorn  # noqa: F401
            cmd = ["gunicorn", "-b", f"{host}:{port}", "app:app"]
            print(f"[+] Production WSGI server (gunicorn) running: {' '.join(cmd)}")
            subprocess.run(cmd)
        except ImportError:
            from app import app

            debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
            app.run(host=host, port=port, debug=debug_mode)


def run_crawl(args):
    """Run Scrapy crawler with passed arguments."""
    cmd = [sys.executable, "-m", "scrapy", "crawl"] + args
    print(f"[+] Running Scrapy crawl: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


def run_tests():
    """Run unit tests."""
    print("[+] Running test suite...")
    cmd = [sys.executable, "-m", "unittest", "discover", "tests"]
    sys.exit(subprocess.call(cmd))


def main():
    if not is_running_in_project_venv():
        run_in_venv()
        return

    # If we reach here, we are guaranteed to be running inside the venv!
    if len(sys.argv) <= 1 or sys.argv[1] == "serve":
        run_server()
    elif sys.argv[1] == "crawl":
        run_crawl(sys.argv[2:])
    elif sys.argv[1] == "test":
        run_tests()
    elif sys.argv[1] in ("setup", "install"):
        print("[+] Environment and dependencies are verified and ready.")
        sys.exit(0)
    elif sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        sys.exit(0)
    else:
        # Default fallback: run custom scrapy or python arguments
        cmd = [sys.executable] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
