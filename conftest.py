"""
conftest.py — Adds project root to sys.path so src.* imports work without
needing to install the package, regardless of where pytest is called from.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
