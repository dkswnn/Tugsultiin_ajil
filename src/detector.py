from __future__ import annotations
from typing import List, Tuple
import numpy as np

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is required by facenet_pytorch runtime
    torch = None  # type: ignore


def _device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Detector:
    def __init__(
        self,
        image_size: int = 160,
        margin: int = 14,
        keep_all: bool = True,
        device: str | None = None,
        backend: str = "mtcnn",
        retina_confidence: float = 0.8,
    ) -> None:
        self.image_size = image_size
        self.margin = int(max(0, margin))
        self.keep_all = bool(keep_all)
        self.backend = str(backend or "mtcnn").lower().strip()
        self.retina_confidence = float(retina_confidence)

        self.mtcnn = None
        self._retina = None

        if self.backend == "retinaface":
            from retinaface import RetinaFace  # type: ignore

            self._retina = RetinaFace
        else:
            from facenet_pytorch import MTCNN  # type: ignore

            self.mtcnn = MTCNN(image_size=image_size, margin=margin, keep_all=keep_all, device=(device or _device()))

    def detect(self, img) -> Tuple[List, List[float]]:
        """Detect and align faces from a PIL.Image; returns (faces, probs).
        If no faces, returns ([], []).
        """
        faces, probs, _ = self.detect_with_boxes(img)
        return faces, probs

    def detect_with_boxes(self, img) -> Tuple[List, List[float], List[Tuple[float, float, float, float]]]:
        """Detect faces and return (faces, probs, boxes).

        - faces: aligned face tensors (3xSxS)
        - probs: detection confidences
        - boxes: list of (x1, y1, x2, y2) on the original image
        """
        if self.backend == "retinaface":
            if torch is None or Image is None:
                return [], [], []

            arr = np.asarray(img.convert("RGB"), dtype="uint8")
            h, w = arr.shape[:2]

            try:
                import cv2  # type: ignore
            except Exception:
                return [], [], []

            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            try:
                res = self._retina.detect_faces(bgr) if self._retina is not None else None
            except Exception:
                res = None

            if not isinstance(res, dict) or len(res) == 0:
                return [], [], []

            faces_list: List = []
            probs_list: List[float] = []
            boxes_list: List[Tuple[float, float, float, float]] = []

            items = []
            for _, info in res.items():
                try:
                    score = float(info.get("score", 0.0))
                    fa = info.get("facial_area", None)
                    if fa is None or len(fa) < 4:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in fa[:4]]
                    if score < self.retina_confidence:
                        continue
                    items.append((score, x1, y1, x2, y2))
                except Exception:
                    continue

            if not items:
                return [], [], []

            items.sort(key=lambda x: x[0], reverse=True)
            if not self.keep_all:
                items = items[:1]

            for score, x1, y1, x2, y2 in items:
                x1 = max(0, x1 - self.margin)
                y1 = max(0, y1 - self.margin)
                x2 = min(w, x2 + self.margin)
                y2 = min(h, y2 + self.margin)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = arr[y1:y2, x1:x2, :]
                if crop.size == 0:
                    continue
                pil_crop = Image.fromarray(crop).resize((self.image_size, self.image_size))
                crop_np = np.asarray(pil_crop, dtype="float32") / 255.0
                t = torch.from_numpy(crop_np).permute(2, 0, 1).contiguous()

                faces_list.append(t)
                probs_list.append(float(score))
                boxes_list.append((float(x1), float(y1), float(x2), float(y2)))

            return faces_list, probs_list, boxes_list

        # MTCNN branch
        boxes, probs_det = self.mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return [], [], []
        boxes_list: List[Tuple[float, float, float, float]] = []
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b]
            boxes_list.append((x1, y1, x2, y2))

        with torch.no_grad():  # type: ignore[attr-defined]
            faces = self.mtcnn.extract(img, boxes, save_path=None)

        faces_list = list(faces) if faces is not None else []
        probs_list = [float(p) for p in probs_det] if probs_det is not None else []
        n = min(len(faces_list), len(probs_list), len(boxes_list))
        return faces_list[:n], probs_list[:n], boxes_list[:n]
