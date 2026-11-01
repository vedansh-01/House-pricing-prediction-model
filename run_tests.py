"""
run_tests.py — Helper to invoke pytest from within Python, working around
the PowerShell bracket issue with the [pending] project folder name.

Usage:
    python run_tests.py [optional: test file name]
"""
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / "venv" / "Scripts" / "python.exe"

def main():
    test_target = sys.argv[1] if len(sys.argv) > 1 else "tests"
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "pytest", test_target, "-v", "--tb=short"],
        cwd=str(PROJECT_DIR),
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
