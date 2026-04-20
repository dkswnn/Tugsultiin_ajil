from __future__ import annotations

from typing import List
import numpy as np


def _device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Embedder:
    def __init__(self, device: str | None = None, model_name: str = "facenet", fusion_alpha: float = 0.35) -> None:
        import torch  # type: ignore
        from facenet_pytorch import InceptionResnetV1  # type: ignore

        self._torch = torch
        self.device = device or _device()
        self.model_name = str(model_name).lower()
        if self.model_name not in ("facenet", "arcface", "facenet_dual"):
            self.model_name = "facenet"
        self.fusion_alpha = float(min(0.9, max(0.0, fusion_alpha)))
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def embed_tensors(self, faces: List) -> np.ndarray:
        if not faces:
            return np.zeros((0, 512), dtype="float32")
        torch = self._torch

        def _to_facenet_input(batch):
            if float(batch.min()) >= 0.0 and float(batch.max()) <= 1.0:
                return (batch - 0.5) / 0.5
            return batch

        with torch.no_grad():
            batch = torch.stack([f for f in faces], dim=0).to(self.device)
            base_in = _to_facenet_input(batch)
            base_embs = self.model(base_in).cpu().numpy().astype("float32")

            if self.model_name == "facenet_dual":
                # Second branch: mild contrast perturbation, then embed and fuse.
                # This keeps one model but forms an embedding ensemble for robustness.
                b2 = batch.clone()
                if float(b2.min()) >= 0.0 and float(b2.max()) <= 1.0:
                    b2 = torch.clamp((b2 - 0.5) * 1.25 + 0.5, 0.0, 1.0)
                b2_in = _to_facenet_input(b2)
                b2_embs = self.model(b2_in).cpu().numpy().astype("float32")
                a = self.fusion_alpha
                embs = ((1.0 - a) * base_embs + a * b2_embs).astype("float32")
            else:
                embs = base_embs
        # L2 normalize rows
        n = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        return (embs / n).astype("float32")
