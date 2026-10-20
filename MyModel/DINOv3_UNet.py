
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DINOV3_REPO = _PROJECT_ROOT / "dinov3-main"


class DINOv3_UNet(nn.Module):

    def __init__(
        self,
        dinov3_weight_path: Optional[str] = None,
        dinov3_local_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        repo_path = Path(
            os.path.expandvars(os.path.expanduser(dinov3_local_path))
            if dinov3_local_path
            else _DEFAULT_DINOV3_REPO
        ).resolve()
        if not repo_path.is_dir():
            raise FileNotFoundError(
                f"DINOv3 source directory was not found: {repo_path}. "
                "Set 'dinov3_local_path' in the configuration."
            )

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from dinov3.hub.backbones import dinov3_convnext_small

        self.dino = dinov3_convnext_small(pretrained=False)

        if dinov3_weight_path:
            weight_path = Path(
                os.path.expandvars(os.path.expanduser(dinov3_weight_path))
            ).resolve()
            if not weight_path.is_file():
                raise FileNotFoundError(f"DINOv3 checkpoint was not found: {weight_path}")
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            incompatible = self.dino.load_state_dict(checkpoint, strict=False)
            print(
                "Loaded DINOv3 weights "
                f"(missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)})."
            )

        if freeze_backbone:
            for parameter in self.dino.parameters():
                parameter.requires_grad = False
            for name, parameter in self.dino.named_parameters():
                if "ada" in name.lower():
                    parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        intermediate_outputs = self.dino.get_intermediate_layers(
            x,
            n=[0, 1, 2, 3],
            reshape=True,
            return_class_token=True,
            norm=True,
        )
        return [output[0] for output in intermediate_outputs]
