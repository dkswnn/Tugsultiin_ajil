from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class ArcFaceHeadRecognizer:
    def __init__(self, model_path: Path, threshold: float | None = None, margin_threshold: float | None = None) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        ckpt = torch.load(str(model_path), map_location="cpu")
        class_names = ckpt.get("class_names", [])
        if not class_names:
            raise ValueError(f"Invalid ArcFace checkpoint (missing class_names): {model_path}")

        in_dim = int(ckpt["in_dim"])
        hidden_dim = int(ckpt["hidden_dim"])
        dropout = float(ckpt.get("dropout", 0.2))
        scale = float(ckpt.get("arcface_scale", 30.0))

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                self.weight = nn.Parameter(torch.empty(len(class_names), hidden_dim))
                nn.init.xavier_uniform_(self.weight)

            def forward(self, x):
                z = self.backbone(x)
                logits = F.linear(F.normalize(z), F.normalize(self.weight)) * scale
                return logits

        self._torch = torch
        self.model = Head().eval()
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.class_names = [str(c) for c in class_names]
        self.threshold = float(threshold) if threshold is not None else float(ckpt.get("unknown_threshold", 0.55))
        self.margin_threshold = float(margin_threshold) if margin_threshold is not None else float(ckpt.get("margin_threshold", 0.10))

    def predict(self, embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if embs.ndim == 1:
            embs = embs[None, :]

        torch = self._torch
        x = torch.from_numpy(embs.astype("float32"))

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype("float32")

        idx = np.argmax(probs, axis=1)
        top1 = probs[np.arange(probs.shape[0]), idx]
        sorted_probs = np.sort(probs, axis=1)
        top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
        margins = top1 - top2

        labels = np.array([self.class_names[i] for i in idx], dtype=object)
        out = []
        for lab, p1, mg in zip(labels, top1, margins):
            if p1 < self.threshold or mg < self.margin_threshold:
                out.append("unknown")
            else:
                out.append(str(lab))
        return np.asarray(out, dtype=object), top1.astype("float32")
