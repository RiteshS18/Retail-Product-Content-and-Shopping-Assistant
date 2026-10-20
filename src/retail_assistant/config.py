from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r") as f:
        return yaml.safe_load(f)


def abs_path(rel: str) -> Path:
    return ROOT / rel
