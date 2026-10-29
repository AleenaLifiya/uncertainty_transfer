"""Portable paths for the copied experiment; independent of the original workspace."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def root_path(relative=''):
    return str(ROOT / relative)
