"""SUS RGB-T semantic-segmentation dataset loader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

from toolbox.datasets.augmentations import (
    ColorJitter,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomScale,
)


class SUS(data.Dataset):
    """Load paired RGB/thermal images and semantic labels.

    Expected minimum layout::

        root/
          train.txt, val.txt, test.txt
          seperated_images/<id>_rgb.jpg
          seperated_images/<id>_th.jpg
          labels/<id>.png

    The historical ``boundary``, ``binary``, ``SDF`` and ``weight`` folders are
    optional. Zero-valued placeholders are returned when they are absent, so
    standard semantic-segmentation training does not require auxiliary labels.
    """

    MODES = {"train", "trainval", "val", "test", "test_day", "test_night"}

    def __init__(self, cfg: dict[str, Any], mode: str = "train", do_aug: bool = True) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported SUS split: {mode}")
        self.mode = mode
        self.root = Path(cfg["root"]).expanduser()
        self.n_classes = int(cfg.get("n_classes", 6))
        self.do_aug = bool(do_aug and mode in {"train", "trainval"})
        self.image_dir = self.root / cfg.get("image_dir", "seperated_images")
        self.label_dir = self.root / cfg.get("label_dir", "labels")
        self.rgb_suffix = cfg.get("rgb_suffix", "_rgb.jpg")
        self.thermal_suffix = cfg.get("thermal_suffix", "_th.jpg")

        if not self.root.is_dir():
            raise FileNotFoundError(f"SUS dataset root was not found: {self.root}")
        split_file = self.root / f"{mode}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"SUS split file was not found: {split_file}")
        self.infos = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.ir_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        scale_range = tuple(float(value) for value in str(cfg.get("scales_range", "0.5 2.0")).split())
        crop_size = tuple(int(value) for value in str(cfg.get("crop_size", "480 640")).split())
        self.aug = Compose([
            ColorJitter(
                brightness=float(cfg.get("brightness", 0.5)),
                contrast=float(cfg.get("contrast", 0.5)),
                saturation=float(cfg.get("saturation", 0.5)),
            ),
            RandomHorizontalFlip(float(cfg.get("p", 0.5))),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True),
        ])

        class_weight_mode = cfg.get("class_weight", "enet")
        if class_weight_mode == "enet":
            self.class_weight = np.array([2.0013, 4.2586, 27.5196, 22.8228, 11.2535, 31.3595], dtype=np.float32)
        elif class_weight_mode == "median_freq_balancing":
            self.class_weight = np.array([0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433], dtype=np.float32)
        elif class_weight_mode in {"none", "no", None}:
            self.class_weight = np.ones(self.n_classes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported class_weight mode: {class_weight_mode}")

        mapping_path = self.root / cfg.get("domain_mapping_file", "domain_mapping.json")
        if mapping_path.is_file():
            self.domain_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        else:
            self.domain_mapping = {}

        dense_feature_path = str(cfg.get("dense_text_feature_path", "")).strip()
        if dense_feature_path and Path(dense_feature_path).is_file():
            self.dense_text_features = torch.load(dense_feature_path, map_location="cpu", weights_only=False)
        else:
            self.dense_text_features = {}
        self.dense_text_feature_dim = int(cfg.get("dense_text_feature_dim", 512))

    def __len__(self) -> int:
        return len(self.infos)

    @staticmethod
    def _open_optional(path: Path, size: tuple[int, int], mode: str = "L", fill: int = 0) -> Image.Image:
        if path.is_file():
            return Image.open(path).convert(mode)
        return Image.new(mode, size, color=fill)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.infos[index]
        rgb_filename = image_id + self.rgb_suffix
        rgb_path = self.image_dir / rgb_filename
        thermal_path = self.image_dir / (image_id + self.thermal_suffix)
        label_path = self.label_dir / f"{image_id}.png"
        if not rgb_path.is_file() or not thermal_path.is_file() or not label_path.is_file():
            missing = [str(path) for path in (rgb_path, thermal_path, label_path) if not path.is_file()]
            raise FileNotFoundError("Missing SUS sample files: " + ", ".join(missing))

        image = Image.open(rgb_path).convert("RGB")
        thermal = Image.open(thermal_path).convert("RGB")
        label = Image.open(label_path).convert("L")
        size = image.size

        sample = {
            "image": image,
            "depth": thermal,
            "label": label,
            "binary": self._open_optional(self.root / "binary" / f"{image_id}.png", size),
            "boundary": self._open_optional(self.root / "boundary" / f"{image_id}.png", size),
            "sdf": self._open_optional(self.root / "SDF" / f"{image_id}.png", size, fill=128),
            "weight": self._open_optional(self.root / "weight" / f"{image_id}.png", size),
        }
        if self.do_aug:
            sample = self.aug(sample)

        dense_feature = self.dense_text_features.get(rgb_filename)
        if dense_feature is None:
            dense_feature = self.dense_text_features.get(image_id)
        if dense_feature is None:
            dense_feature = torch.zeros(self.dense_text_feature_dim, dtype=torch.float32)
        elif not torch.is_tensor(dense_feature):
            dense_feature = torch.as_tensor(dense_feature, dtype=torch.float32)
        else:
            dense_feature = dense_feature.float()

        output = {
            "image": self.im_to_tensor(sample["image"]),
            "depth": self.ir_to_tensor(sample["depth"]),
            "label": torch.from_numpy(np.asarray(sample["label"], dtype=np.int64).copy()).long(),
            "boundary": torch.from_numpy((np.asarray(sample["boundary"], dtype=np.float32) / 255.0).copy()).long(),
            "binary": torch.from_numpy((np.asarray(sample["binary"], dtype=np.float32) / 255.0).copy()).long(),
            "sdf": torch.from_numpy(((np.asarray(sample["sdf"], dtype=np.float32) / 127.5) - 1.0).copy()),
            "weight": torch.from_numpy((np.asarray(sample["weight"], dtype=np.float32) / 255.0).copy()),
            "dense_text_feature": dense_feature,
            "domain_gt": torch.tensor(self.domain_mapping.get(rgb_filename, 0), dtype=torch.long),
            "label_path": Path(image_id).name + ".png",
            "sample_id": image_id,
        }
        return output

    @property
    def cmap(self) -> list[tuple[int, int, int]]:
        return [
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
        ]
