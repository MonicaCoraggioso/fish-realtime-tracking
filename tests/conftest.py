"""Make the repository modules importable in tests without installing them."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("4_dataset_building", "3_pose_estimation/notebooks"):
    sys.path.insert(0, str(ROOT / sub))


def load_module(relpath, name):
    """Import a single file by path (2_acquisition/ and 3_pose_estimation/ both have a `utilities` package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
