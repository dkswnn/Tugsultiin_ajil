from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.linalg import norm


class CosineGallery:
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        threshold: float | None = 0.8,
        margin_threshold: float | None = None,
    ) -> None:
        g = np.asarray(embeddings, dtype="float32")
        g = g / (norm(g, axis=1, keepdims=True) + 1e-8)
        self.gallery = g
        self.labels = np.asarray(labels)
        self.threshold = float(threshold) if threshold is not None else None
        self.margin_threshold = float(margin_threshold) if margin_threshold is not None else None

    def predict(self, embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if embs.ndim == 1:
            embs = embs[None, :]
        q = embs.astype("float32")
        q = q / (norm(q, axis=1, keepdims=True) + 1e-8)
        sims = q @ self.gallery.T  # (N, M)
        idx = np.argmax(sims, axis=1)
        scores = sims[np.arange(sims.shape[0]), idx]
        labels = self.labels[idx].astype(object)

        if self.threshold is not None:
            labels = np.array([lab if sc >= self.threshold else "unknown" for lab, sc in zip(labels, scores)], dtype=object)
        else:
            labels = labels.astype(object)

        if self.margin_threshold is not None and sims.shape[1] >= 2:
            part = np.partition(sims, -2, axis=1)
            top1 = part[:, -1]
            top2 = part[:, -2]
            margin = (top1 - top2).astype("float32")
            labels = np.array([lab if m >= self.margin_threshold else "unknown" for lab, m in zip(labels, margin)], dtype=object)
        return labels, scores.astype("float32")
