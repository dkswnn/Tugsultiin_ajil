from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm


class AdaptiveGallery:
    """Adaptive, class-conditional recognizer combining cosine and Mahalanobis."""

    def __init__(
        self,
        per_image_embeddings: np.ndarray,
        per_image_labels: np.ndarray,
        mean_embeddings: np.ndarray,
        mean_labels: np.ndarray,
        weight: float = 0.6,
        threshold: Optional[float] = 0.8,
        margin_threshold: Optional[float] = None,
    ) -> None:
        self.threshold = float(threshold) if threshold is not None else None
        self.margin_threshold = float(margin_threshold) if margin_threshold is not None else None
        self.weight = float(weight)

        ea = np.asarray(per_image_embeddings, dtype="float32")
        ea = ea / (norm(ea, axis=1, keepdims=True) + 1e-8)
        la = np.asarray(per_image_labels)

        em = np.asarray(mean_embeddings, dtype="float32")
        em = em / (norm(em, axis=1, keepdims=True) + 1e-8)
        lm = np.asarray(mean_labels)

        self.class_to_centroids: Dict[str, List[np.ndarray]] = {}
        for vec, lab in zip(em, lm):
            s = str(lab)
            self.class_to_centroids.setdefault(s, []).append(vec.astype("float32"))

        self.class_mean: Dict[str, np.ndarray] = {}
        self.class_cov: Dict[str, np.ndarray] = {}
        lab_strings = [str(l) for l in la]
        uniq = sorted(set(lab_strings))
        for cls in uniq:
            idx = [i for i, lbl in enumerate(lab_strings) if lbl == cls]
            if not idx:
                continue
            x = ea[idx]
            mu = x.mean(axis=0)
            mu = mu / (norm(mu) + 1e-8)
            c = np.cov(x.T)
            c = c + (1e-2) * np.eye(c.shape[0], dtype=c.dtype)
            self.class_mean[cls] = mu.astype("float32")
            self.class_cov[cls] = c.astype("float32")

    def _mahalanobis(self, x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
        d = x - mu
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)
        return float(d @ inv @ d)

    def predict(self, embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if embs.ndim == 1:
            embs = embs[None, :]
        q = embs.astype("float32")
        q = q / (norm(q, axis=1, keepdims=True) + 1e-8)

        classes = list(self.class_to_centroids.keys())
        out_labels = []
        out_scores = []
        for i in range(q.shape[0]):
            qi = q[i]
            cos_scores = []
            for cls in classes:
                cents = self.class_to_centroids.get(cls, [])
                if not cents:
                    cos_scores.append(-1.0)
                    continue
                sims = [float(qi @ c) for c in cents]
                cos_scores.append(max(sims))

            maha_dists = []
            for cls in classes:
                mu = self.class_mean.get(cls)
                cov = self.class_cov.get(cls)
                if mu is None or cov is None:
                    maha_dists.append(1e9)
                    continue
                md = self._mahalanobis(qi, mu, cov)
                maha_dists.append(md)
            arr = np.asarray(maha_dists, dtype=float)
            mn, mx = float(np.min(arr)), float(np.max(arr))
            if mx - mn < 1e-8:
                maha_sim = [0.0 for _ in classes]
            else:
                maha_sim = [float(1.0 - (v - mn) / (mx - mn)) for v in arr]

            combined = [self.weight * c + (1.0 - self.weight) * m for c, m in zip(cos_scores, maha_sim)]
            order = np.argsort(combined)[::-1]
            top = int(order[0])
            best_cls = classes[top]
            best_score = float(combined[top])

            if self.margin_threshold is not None and len(order) > 1:
                second = float(combined[int(order[1])])
                if (best_score - second) < self.margin_threshold:
                    out_labels.append("unknown")
                    out_scores.append(best_score)
                    continue

            if self.threshold is not None and best_score < self.threshold:
                out_labels.append("unknown")
                out_scores.append(best_score)
            else:
                out_labels.append(best_cls)
                out_scores.append(best_score)

        return np.asarray(out_labels, dtype=object), np.asarray(out_scores, dtype="float32")
