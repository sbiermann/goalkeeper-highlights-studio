from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("default.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if path:
        with path.open("r", encoding="utf-8") as handle:
            config = deep_merge(config, yaml.safe_load(handle) or {})
    return config
