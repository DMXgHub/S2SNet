"""Semantic-segmentation metrics for the SUS label set."""

from __future__ import annotations

import numpy as np


CLASS_NAMES = ("unlabeled", "road", "sidewalk", "person", "car", "bicycle")


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int | None = None) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target = np.asarray(target).reshape(-1)
        prediction = np.asarray(prediction).reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        if self.ignore_index is not None:
            valid &= target != self.ignore_index
        encoded = self.num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
        self.confusion += np.bincount(
            encoded, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self, exclude_from_mean: tuple[int, ...] = (0,)) -> dict[str, object]:
        hist = self.confusion.astype(np.float64)
        tp = np.diag(hist)
        support = hist.sum(axis=1)
        predicted = hist.sum(axis=0)
        union = support + predicted - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        precision = np.divide(tp, predicted, out=np.full_like(tp, np.nan), where=predicted > 0)
        recall = np.divide(tp, support, out=np.full_like(tp, np.nan), where=support > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.full_like(tp, np.nan),
            where=(precision + recall) > 0,
        )
        keep = np.ones(self.num_classes, dtype=bool)
        for index in exclude_from_mean:
            if 0 <= index < self.num_classes:
                keep[index] = False
        total = hist.sum()
        pixel_acc = float(tp.sum() / total) if total else float("nan")
        return {
            "pixel_accuracy": pixel_acc,
            "miou": float(np.nanmean(iou[keep])),
            "mf1": float(np.nanmean(f1[keep])),
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": self.confusion.copy(),
        }
