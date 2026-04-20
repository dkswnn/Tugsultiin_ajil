from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.linalg import norm


class KNNClassifier:
    """k-NN classifier over embeddings using cosine similarity."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
    ) -> None:
        g = np.asarray(embeddings, dtype="float32")
        g = g / (norm(g, axis=1, keepdims=True) + 1e-8)
        self.gallery = g
        self.labels = np.asarray(labels)
        self.k = int(max(1, k))
        self.threshold = float(threshold) if threshold is not None else None
        self.margin_threshold = float(margin_threshold) if margin_threshold is not None else None

    def predict(self, embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if embs.ndim == 1:
            embs = embs[None, :]
        q = embs.astype("float32")
        q = q / (norm(q, axis=1, keepdims=True) + 1e-8)

        sims = q @ self.gallery.T
        idx_sorted = np.argsort(-sims, axis=1)
        topk_idx = idx_sorted[:, : self.k]
        topk_sims = np.take_along_axis(sims, topk_idx, axis=1)
        topk_labels = self.labels[topk_idx]

        out_labels = []
        out_scores = []
        for i in range(q.shape[0]):
            sims_i = topk_sims[i]
            labs_i = topk_labels[i]
            class_scores = {}
            for lab, sc in zip(labs_i, sims_i):
                class_scores[lab] = class_scores.get(lab, 0.0) + float(sc)
            items = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
            best_lab, best_sum = items[0]
            second_sum = items[1][1] if len(items) > 1 else -1e9
            best_mask = (labs_i == best_lab)
            best_top_sim = float(np.max(sims_i[best_mask])) if np.any(best_mask) else float(np.max(sims_i))

            lab = best_lab
            if self.threshold is not None and best_top_sim < self.threshold:
                lab = "unknown"
            if lab != "unknown" and self.margin_threshold is not None:
                if (best_sum - second_sum) < self.margin_threshold:
                    lab = "unknown"

            out_labels.append(lab)
            out_scores.append(best_top_sim)

        return np.asarray(out_labels, dtype=object), np.asarray(out_scores, dtype="float32")
