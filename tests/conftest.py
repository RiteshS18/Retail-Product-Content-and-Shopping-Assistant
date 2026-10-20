"""
conftest.py – pytest configuration for the retail assistant test suite.
Sets up the project root on sys.path so all imports work without installation.
"""
import sys
from pathlib import Path

# Ensure project src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
