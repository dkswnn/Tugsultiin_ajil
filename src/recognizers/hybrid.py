from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.linalg import norm


class HybridKnnCosine:
    def __init__(
        self,
        per_image_embeddings: np.ndarray,
        per_image_labels: np.ndarray,
        mean_embeddings: np.ndarray,
        mean_labels: np.ndarray,
        k: int = 5,
        alpha: float = 0.5,
        threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        mahalanobis_weight: Optional[float] = None,
        precomputed_mahal: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> None:
        g = np.asarray(per_image_embeddings, dtype="float32")
        g = g / (norm(g, axis=1, keepdims=True) + 1e-8)
        self.gallery = g
        self.labels = np.asarray(per_image_labels)

        m = np.asarray(mean_embeddings, dtype="float32")
        m = m / (norm(m, axis=1, keepdims=True) + 1e-8)
        self.mean_gallery = m
        self.mean_labels = np.asarray(mean_labels)

        self.k = int(max(1, k))
        self.alpha = float(alpha)
        self.threshold = float(threshold) if threshold is not None else None
        self.margin_threshold = float(margin_threshold) if margin_threshold is not None else None
        self.mahalanobis_weight = float(mahalanobis_weight) if mahalanobis_weight is not None else 0.0
        self.class_mean: Dict[str, np.ndarray] = {}
        self.class_inv: Dict[str, np.ndarray] = {}

        if self.mahalanobis_weight > 0.0:
            if precomputed_mahal is not None:
                cm = precomputed_mahal.get("class_mean", {})
                ci = precomputed_mahal.get("class_inv", {})
                self.class_mean = {k: v.astype("float32") for k, v in cm.items()}
                self.class_inv = {k: v.astype("float32") for k, v in ci.items()}
            else:
                lab_strings = [str(l) for l in self.labels]
                uniq = sorted(set(lab_strings))
                ea = self.gallery
                for cls in uniq:
                    idx = [i for i, lbl in enumerate(lab_strings) if lbl == cls]
                    if not idx:
                        continue
                    x = ea[idx]
                    mu = x.mean(axis=0)
                    mu = mu / (norm(mu) + 1e-8)
                    c = np.cov(x.T)
                    c = c + (1e-2) * np.eye(c.shape[0], dtype=c.dtype)
                    inv = np.linalg.pinv(c)
                    self.class_mean[cls] = mu.astype("float32")
                    self.class_inv[cls] = inv.astype("float32")

    def predict(self, embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if embs.ndim == 1:
            embs = embs[None, :]
        q = embs.astype("float32")
        q = q / (norm(q, axis=1, keepdims=True) + 1e-8)

        sims_all = q @ self.gallery.T
        idx_sorted = np.argsort(-sims_all, axis=1)
        topk_idx = idx_sorted[:, : self.k]
        topk_sims = np.take_along_axis(sims_all, topk_idx, axis=1)
        topk_labels = self.labels[topk_idx]

        sims_mean = q @ self.mean_gallery.T

        out_labels = []
        out_scores = []
        for i in range(q.shape[0]):
            sims_i = topk_sims[i]
            labs_i = topk_labels[i]

            class_knn = {}
            for lab, sc in zip(labs_i, sims_i):
                class_knn[lab] = class_knn.get(lab, 0.0) + float(sc)
            for lab in list(class_knn.keys()):
                class_knn[lab] = class_knn[lab] / float(self.k)

            class_mean = {}
            tmp_means: dict = {}
            for j, lab in enumerate(self.mean_labels):
                sc = float(sims_mean[i, j])
                if lab not in tmp_means or sc > tmp_means[lab]:
                    tmp_means[lab] = sc
            class_mean.update(tmp_means)

            maha_sim: Dict[object, float] = {}
            if self.mahalanobis_weight > 0.0:
                dists = []
                keys = []
                for cls in set(list(class_mean.keys()) + list(class_knn.keys())):
                    mu = self.class_mean.get(str(cls))
                    inv = self.class_inv.get(str(cls))
                    if mu is None or inv is None:
                        dists.append(1e9)
                        keys.append(cls)
                        continue
                    d = q[i] - mu
                    md = float(d @ inv @ d)
                    dists.append(md)
                    keys.append(cls)
                arr = np.asarray(dists, dtype=float)
                mn, mx = float(np.min(arr)), float(np.max(arr))
                if mx - mn < 1e-8:
                    sims = [0.0 for _ in keys]
                else:
                    sims = [float(1.0 - (v - mn) / (mx - mn)) for v in arr]
                for kkey, sval in zip(keys, sims):
                    maha_sim[kkey] = sval

            combined = {}
            all_classes = set(list(class_mean.keys()) + list(class_knn.keys()))
            for lab in all_classes:
                knn_part = class_knn.get(lab, 0.0)
                mean_part = class_mean.get(lab, 0.0)
                maha_part = maha_sim.get(lab, 0.0) if self.mahalanobis_weight > 0.0 else 0.0
                combined[lab] = self.alpha * knn_part + (1.0 - self.alpha) * mean_part + self.mahalanobis_weight * maha_part

            items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            best_lab, best_comb = items[0]
            second_comb = items[1][1] if len(items) > 1 else -1e9

            best_mask = (labs_i == best_lab)
            best_top_sim = float(np.max(sims_i[best_mask])) if np.any(best_mask) else float(np.max(sims_i))

            lab = best_lab
            if self.threshold is not None and best_top_sim < self.threshold:
                lab = "unknown"
            if lab != "unknown" and self.margin_threshold is not None:
                if (best_comb - second_comb) < self.margin_threshold:
                    lab = "unknown"

            out_labels.append(lab)
            out_scores.append(best_top_sim)

        return np.asarray(out_labels, dtype=object), np.asarray(out_scores, dtype="float32")
