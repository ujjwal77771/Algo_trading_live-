# tests/conftest.py
"""
Pytest configuration.
Adds the project root to sys.path so 'src.*' imports resolve
without requiring a package install.
"""
import sys
from pathlib import Path

# Insert project root (parent of tests/) into sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
