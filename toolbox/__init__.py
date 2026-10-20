"""Utility entry points used by the original S2S training and evaluation code."""

from .log import get_logger
from .optim.AdamW import AdamW
from .optim.Ranger import Ranger
from .utils import ClassWeight, class_to_RGB, save_ckpt, setup_seed

def get_dataset(cfg):
    """Create the SUS train, validation, and test datasets."""
    if cfg.get("dataset") != "SUS":
        raise ValueError("This repository currently supports the SUS dataset")
    from .datasets.SUS import SUS

    return SUS(cfg, mode="train"), SUS(cfg, mode="val"), SUS(cfg, mode="test")


def get_model(cfg):
    """Build S2S using the original ``get_model(cfg)`` calling convention."""
    model_name = str(cfg.get("model_name", "S2S"))
    if model_name != "S2S":
        raise ValueError(f"Unsupported model_name={model_name!r}; expected 'S2S'")

    from MyModel.S2S import S2S

    return S2S(
        num_classes=int(cfg.get("n_classes", 6)),
        text_encoder=cfg.get("text_encoder", "longclip"),
        longclip_repo=cfg.get("longclip_repo") or None,
        longclip_checkpoint=cfg.get("longclip_checkpoint") or None,
        clip_model=cfg.get("clip_model", "ViT-B/32"),
        beta_cal_beta=float(cfg.get("beta_cal_beta", 0.2)),
        beta_cal_weight=float(cfg.get("beta_cal_weight", 0.005)),
        proto_contrast_temperature=float(cfg.get("proto_contrast_temperature", 0.05)),
        proto_contrast_weight=cfg.get("proto_contrast_weight"),
        prototype_kl_weight=float(cfg.get("prototype_kl_weight", 1e-3)),
        prompt_balance_weight=float(cfg.get("prompt_balance_weight", 0.01)),
        prompt_confidence_weight=float(cfg.get("prompt_confidence_weight", 0.001)),
        prompt_route_history_maxlen=cfg.get("prompt_route_history_maxlen", 1),
        dinov3_weight_path=cfg.get("dinov3_weight_path") or None,
        dinov3_local_path=cfg.get("dinov3_local_path") or None,
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        initialize_text_encoder=bool(cfg.get("initialize_text_encoder", True)),
        text_embed_dim=int(cfg.get("dense_text_feature_dim", 512)),
        device=cfg.get("device"),
    )


def get_metrics(cfg):
    """Create the metric helpers used by the original code."""
    if cfg.get("dataset") != "SUS":
        raise ValueError("This repository currently supports the SUS dataset")
    from toolbox.metrics_SUS import averageMeter, runningScore

    return averageMeter(), runningScore(cfg["n_classes"])

__all__ = [
    "AdamW",
    "ClassWeight",
    "Ranger",
    "class_to_RGB",
    "get_dataset",
    "get_logger",
    "get_metrics",
    "get_model",
    "save_ckpt",
    "setup_seed",
]
