import sys
from pathlib import Path

def get_root() -> Path:
    """Return the root path of the project."""
    fillter_path = [p for p in sys.path if "ML-linear-regression/modules" in p]
    return  Path(fillter_path[0].replace("/modules", ""))
