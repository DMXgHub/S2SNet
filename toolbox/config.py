"""Configuration loading and path normalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATH_KEYS = {
    "root",
    "dense_text_feature_path",
    "longclip_repo",
    "longclip_checkpoint",
    "dinov3_local_path",
    "dinov3_weight_path",
}


def _resolve_path(value: str, base_dir: Path) -> str:
    value = os.path.expandvars(os.path.expanduser(value.strip()))
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    # Repository-relative paths are more predictable than process-CWD paths.
    candidate = (PROJECT_ROOT / path).resolve()
    if candidate.exists() or not (base_dir / path).exists():
        return str(candidate)
    return str((base_dir / path).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key in _PATH_KEYS:
        value = cfg.get(key)
        if isinstance(value, str):
            cfg[key] = _resolve_path(value, config_path.parent)
    cfg["_config_path"] = str(config_path)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    output = {key: value for key, value in cfg.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
