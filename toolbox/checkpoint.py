"""Checkpoint I/O with compatibility for the original S2S implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


_STATE_KEYS = ("state_dict", "model_state_dict", "model", "net", "network", "weights")
_PREFIXES = ("module.", "_orig_mod.", "model.", "network.")


@dataclass(frozen=True)
class CheckpointLoadResult:
    path: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    epoch: int | None = None
    best_miou: float | None = None

    @property
    def exact(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys


def _is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()
    )


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    if isinstance(checkpoint, nn.Module):
        return dict(checkpoint.state_dict()), {}
    if _is_tensor_mapping(checkpoint):
        return dict(checkpoint), checkpoint
    if isinstance(checkpoint, Mapping):
        for key in _STATE_KEYS:
            candidate = checkpoint.get(key)
            if isinstance(candidate, nn.Module):
                return dict(candidate.state_dict()), checkpoint
            if _is_tensor_mapping(candidate):
                return dict(candidate), checkpoint
    raise TypeError(
        "Unsupported checkpoint format. Expected a raw state_dict, a checkpoint "
        "containing state_dict/model_state_dict/model, or a serialized nn.Module."
    )


def _strip_repeated_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def normalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = _strip_repeated_prefixes(key)
        if clean_key.startswith("encoder."):
            clean_key = "backbone." + clean_key[len("encoder."):]
        elif clean_key.startswith("seg_decoder."):
            clean_key = "decoder." + clean_key[len("seg_decoder."):]
        normalized[clean_key] = value
    return normalized


def load_s2s_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointLoadResult:
    """Load S2S weights from raw or nested checkpoint formats.

    Compatibility includes raw ``model.pth`` state dictionaries, DataParallel
    ``module.`` prefixes, ``torch.compile`` prefixes, and common nested
    checkpoint dictionaries. Parameter compatibility is determined by the
    child-module state keys, which are kept stable.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict, metadata = extract_state_dict(checkpoint)
    state_dict = normalize_state_dict(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    epoch = metadata.get("epoch") if isinstance(metadata, Mapping) else None
    best_miou = metadata.get("best_miou") if isinstance(metadata, Mapping) else None
    return CheckpointLoadResult(
        path=str(path),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        epoch=int(epoch) if epoch is not None else None,
        best_miou=float(best_miou) if best_miou is not None else None,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    epoch: int | None = None,
    best_miou: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    payload: dict[str, Any] = {
        "model_name": "S2S",
        "state_dict": unwrapped.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if best_miou is not None:
        payload["best_miou"] = float(best_miou)
    if config is not None:
        payload["config"] = {k: v for k, v in config.items() if not str(k).startswith("_")}
    torch.save(payload, target)


def save_raw_state_dict(path: str | Path, model: nn.Module) -> None:
    """Save a raw ``state_dict`` compatible with existing ``model.pth`` files."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    torch.save(unwrapped.state_dict(), target)
