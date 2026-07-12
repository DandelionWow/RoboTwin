import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = PROJECT_ROOT / "rpy_direction_validation"

if not (PROJECT_ROOT / "third_party" / "robotwin").exists():
    raise RuntimeError(f"Unexpected PROJECT_ROOT for rpy_grid_rpy_rotation shim: {PROJECT_ROOT}")

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from grid_rpy_rotation_core import rpy_grid_rpy_rotation


__all__ = ["rpy_grid_rpy_rotation"]
