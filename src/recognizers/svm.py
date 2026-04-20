from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import norm


class SVMClassifier:
    """SVM classifier over embeddings with unknown rejection by probability and margin."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        c: float = 8.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        model_path: Path | None = None,
    ) -> None:
        self.threshold = float(threshold) if threshold is not None else None
        self.margin_threshold = float(margin_threshold) if margin_threshold is not None else None
        self.model = None

        x = np.asarray(embeddings, dtype="float32")
        x = x / (norm(x, axis=1, keepdims=True) + 1e-8)
        y = np.asarray(labels).astype(str)

        try:
            import joblib
            from sklearn.svm import SVC
        except Exception as exc:
            raise RuntimeError("SVM requires scikit-learn and joblib to be installed.") from exc

        loaded = False
        if model_path is not None and model_path.exists():
            try:
                loaded_model = joblib.load(model_path)
                if hasattr(loaded_model, "predict_proba"):
                    self.model = loaded_model
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            self.model = SVC(
                C=float(c),
                kernel=str(kernel),
                gamma=gamma,
                probability=True,
                class_weight="balanced",
            )
            self.model.fit(x, y)
            if model_path is not None:
                try:
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(self.model, model_path)
                except Exception:
                    pass

    def predict(self, embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if embs.ndim == 1:
            embs = embs[None, :]
        q = embs.astype("float32")
        q = q / (norm(q, axis=1, keepdims=True) + 1e-8)

        probs = self.model.predict_proba(q)
        cls = self.model.classes_
        idx = np.argmax(probs, axis=1)
        scores = probs[np.arange(probs.shape[0]), idx].astype("float32")
        labels = cls[idx].astype(object)

        if self.threshold is not None:
            labels = np.array([lab if sc >= self.threshold else "unknown" for lab, sc in zip(labels, scores)], dtype=object)
        else:
            labels = labels.astype(object)

        if self.margin_threshold is not None and probs.shape[1] >= 2:
            part = np.partition(probs, -2, axis=1)
            top1 = part[:, -1]
            top2 = part[:, -2]
            margin = (top1 - top2).astype("float32")
            labels = np.array([lab if m >= self.margin_threshold else "unknown" for lab, m in zip(labels, margin)], dtype=object)

        return labels, scores
