from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple, List
import csv
import numpy as np


def list_person_dirs(students_dir: Path) -> List[Path]:
    return [p for p in sorted(Path(students_dir).glob("*")) if p.is_dir()]


def iter_images(folder: Path) -> Iterator[Path]:
    """Yield image paths (case-insensitive) including common phone formats.

    Handles uppercase extensions (e.g. .JPG) and HEIC/HEIF if present.
    """
    if not Path(folder).exists():
        return
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix.lower() in allowed:
            yield p


def save_embeddings_npz(path: Path, embeddings: np.ndarray, labels: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, labels=labels)


def load_embeddings_npz(path: Path):
    path = Path(path)
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    return data.get("embeddings"), data.get("labels")


def list_captures(captures_dir: Path) -> List[Path]:
    imgs = list(iter_images(captures_dir))
    return sorted(imgs)


def write_session_csv(csv_path: Path, rows: List[Tuple[str, str, int, str, float]]):
    """Write session results.

    Columns: timestamp, filename, face_index, label, score
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "filename", "face_index", "label", "score"])
        w.writerows(rows)


def annotate_and_save(img, boxes: List[Tuple[float, float, float, float]], labels: List[str], scores: List[float], out_path: Path) -> None:
    """Draw simple boxes and labels on a PIL.Image and save to out_path.

    - boxes: (x1, y1, x2, y2) in original image coordinates
    - labels/scores aligned to boxes
    """
    from PIL import ImageDraw

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        # box
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
        # label text
        text = f"{lab} {sc:.2f}"
        # simple text background
        tw, th = draw.textlength(text), 12
        draw.rectangle([(x1, max(0, y1 - th - 4)), (x1 + tw + 6, y1)], fill=(0, 255, 0))
        draw.text((x1 + 3, max(0, y1 - th - 2)), text, fill=(0, 0, 0))

    img.save(out_path)
