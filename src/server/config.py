"""Configuration settings for the server."""

from pathlib import Path
from typing import Any, Dict
import yaml

# Adjust path: src/server/config.py -> src/server -> src -> root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "attribution.yaml"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

def load_config() -> Dict[str, Any]:
    """Load attribution configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}
