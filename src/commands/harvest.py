from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List


def cmd_harvest(
    cfg: Dict,
    paths: Dict[str, Path],
    min_score: float = 0.80,
    include_unknown: bool = False,
    to_students: bool = False,
    out_dir: Path | None = None,
    max_per_label: int = 100,
    periodic_only: bool = False,
    outside_period_only: bool = False,
    recent_hours: float | None = None,
    min_crop_size: int = 32,
    min_focus_var: float = 18.0,
    unknown_folder: str = "unrecognized",
    crop_expand: float = 0.20,
    review_only: bool = False,
) -> None:
    from PIL import Image
    import numpy as np

    from src.detector import Detector
    from src.embedding import Embedder
    from src.io_utils import load_embeddings_npz
    from src.recognizers import (
        CosineGallery,
        KNNClassifier,
        HybridKnnCosine,
        AdaptiveGallery,
        ArcFaceHeadRecognizer,
        SVMClassifier,
    )

    E = L = None
    if not review_only:
        E, L = load_embeddings_npz(paths["embeddings_file"])
        if E is None or L is None or len(E) == 0:
            print(f"No gallery embeddings found at {paths['embeddings_file']}. Run 'python main.py build' first.")
            return

    captures_root = paths["captures_dir"]
    if not captures_root.exists():
        print(f"Captures dir not found: {captures_root}")
        return

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
    skip_parts = {"annotated", ".periodic_queue", "outputs", "_all_flat", "_quarantine"}
    def _is_periodic_raw(p: Path) -> bool:
        try:
            rel = p.relative_to(captures_root)
        except Exception:
            return False
        # Periodic raw frames are typically under .../<lesson>/captures/<file>
        return any(part.lower() == "captures" for part in rel.parts[:-1])

    def _is_outside_period(p: Path) -> bool:
        try:
            rel = p.relative_to(captures_root)
        except Exception:
            return False
        return any(part.lower() == "outside_period" for part in rel.parts)

    if periodic_only and outside_period_only:
        print("Choose only one filter: --periodic-only or --outside-period-only")
        return

    cutoff_dt = None
    if recent_hours is not None and float(recent_hours) > 0:
        cutoff_dt = datetime.now() - timedelta(hours=float(recent_hours))

    images = []
    for p in captures_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in allowed:
            continue
        if not skip_parts.isdisjoint({part.lower() for part in p.parts}):
            continue
        if periodic_only and not _is_periodic_raw(p):
            continue
        if outside_period_only and not _is_outside_period(p):
            continue
        if cutoff_dt is not None:
            try:
                if datetime.fromtimestamp(p.stat().st_mtime) < cutoff_dt:
                    continue
            except Exception:
                continue
        images.append(p)

    images = sorted(images)
    if not images:
        print(f"No images found under {captures_root}")
        return

    det_cfg = cfg.get("detector", {})
    facenet_cfg = cfg.get("facenet", {})
    det = Detector(
        keep_all=True,
        image_size=int(facenet_cfg.get("image_size", 160) or 160),
        margin=int(facenet_cfg.get("margin", 14) or 14),
        backend=str(det_cfg.get("backend", "retinaface") or "retinaface"),
        retina_confidence=float(det_cfg.get("retina_confidence", 0.8) or 0.8),
    )

    rec_cfg = cfg.get("recognition", {})
    method = str(rec_cfg.get("method", "cosine")).lower()
    thr = float(rec_cfg.get("decision_threshold", 0.8))
    margin_thr = rec_cfg.get("margin_threshold", None)
    margin_thr = float(margin_thr) if margin_thr is not None else None
    knn_k = int(rec_cfg.get("knn_k", 5))
    hybrid_alpha = float(rec_cfg.get("hybrid_alpha", 0.5))
    svm_c = float(rec_cfg.get("svm_c", 8.0) or 8.0)
    svm_kernel = str(rec_cfg.get("svm_kernel", "rbf") or "rbf")
    svm_gamma = rec_cfg.get("svm_gamma", "scale")

    rec = None
    if not review_only:
        emb_cfg = cfg.get("embedding", {})
        emb = Embedder(
            model_name=str(emb_cfg.get("model", "facenet")),
            fusion_alpha=float(emb_cfg.get("fusion_alpha", 0.35) or 0.35),
        )
        if method == "arcface_head":
            model_path = paths.get("arcface_model_file")
            if model_path is None or not model_path.exists():
                print("ArcFace head model not found. Run 'python main.py train' first.")
                return
            rec = ArcFaceHeadRecognizer(model_path, threshold=thr, margin_threshold=margin_thr)
        elif method == "knn":
            Eall, Lall = load_embeddings_npz(paths["embeddings_all_file"])
            if Eall is None or Lall is None or len(Eall) == 0:
                print(f"No per-image embeddings found at {paths['embeddings_all_file']}. Run 'python main.py build' first.")
                return
            rec = KNNClassifier(Eall, Lall, k=knn_k, threshold=thr, margin_threshold=margin_thr)
        elif method == "hybrid":
            Eall, Lall = load_embeddings_npz(paths["embeddings_all_file"])
            if Eall is None or Lall is None or len(Eall) == 0:
                print(f"No per-image embeddings found at {paths['embeddings_all_file']}. Run 'python main.py build' first.")
                return
            maha_w = float(rec_cfg.get("mahalanobis_weight", 0.0) or 0.0)
            rec = HybridKnnCosine(Eall, Lall, E, L, k=knn_k, alpha=hybrid_alpha, threshold=thr, margin_threshold=margin_thr, mahalanobis_weight=maha_w)
        elif method == "adaptive":
            Eall, Lall = load_embeddings_npz(paths["embeddings_all_file"])
            if Eall is None or Lall is None or len(Eall) == 0:
                print(f"No per-image embeddings found at {paths['embeddings_all_file']}. Run 'python main.py build' first.")
                return
            weight = hybrid_alpha if hybrid_alpha is not None else 0.6
            rec = AdaptiveGallery(Eall, Lall, E, L, weight=weight, threshold=thr, margin_threshold=margin_thr)
        elif method == "svm":
            Eall, Lall = load_embeddings_npz(paths["embeddings_all_file"])
            if Eall is None or Lall is None or len(Eall) == 0:
                print(f"No per-image embeddings found at {paths['embeddings_all_file']}. Run 'python main.py build' first.")
                return
            rec = SVMClassifier(
                Eall,
                Lall,
                threshold=thr,
                margin_threshold=margin_thr,
                c=svm_c,
                kernel=svm_kernel,
                gamma=svm_gamma,
                model_path=paths.get("svm_model_file"),
            )
        else:
            rec = CosineGallery(E, L, threshold=thr, margin_threshold=margin_thr)

    students_dir = paths["students_dir"]
    review_root = out_dir or (students_dir.parent / "harvest_review")
    review_root.mkdir(parents=True, exist_ok=True)

    min_detect_prob = float(rec_cfg.get("min_detect_prob", 0.85))
    min_box_size = int(rec_cfg.get("min_box_size", 24) or 24)

    def _focus_measure_from_crop(crop_img) -> float:
        arr = np.asarray(crop_img, dtype="float32")
        if arr.ndim == 3:
            g = arr.mean(axis=2)
        else:
            g = arr
        if g.shape[0] < 4 or g.shape[1] < 4:
            return 0.0
        gx = np.diff(g, axis=1)
        gy = np.diff(g, axis=0)
        return float(np.var(gx) + np.var(gy))

    label_counts: Dict[str, int] = {}
    saved = 0
    scanned_faces = 0

    print(f"Harvesting from {len(images)} image(s) under {captures_root}")
    print(f"Recognizer method: {'review-only' if review_only else method}")
    print(f"Destination mode: {'students_dir' if to_students else 'harvest_review'}")
    if review_only:
        print("Mode: detector-only fast review")
    if periodic_only:
        print("Filter: periodic raw captures only")
    if outside_period_only:
        print("Filter: outside_period captures only")
    if cutoff_dt is not None:
        print(f"Filter: recent only (last {recent_hours}h)")
    print(f"Min crop size: {min_crop_size}px")
    print(f"Min focus var: {min_focus_var}")
    print(f"Crop expand: {max(0.0, float(crop_expand)):.2f}")

    for i, img_path in enumerate(images, start=1):
        if i == 1 or i % 20 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        faces, probs, boxes = det.detect_with_boxes(img)
        if not faces:
            continue

        keep = []
        for j, (p, b) in enumerate(zip(probs, boxes)):
            x1, y1, x2, y2 = b
            if float(p) < min_detect_prob:
                continue
            if int(x2 - x1) < min_box_size or int(y2 - y1) < min_box_size:
                continue
            keep.append(j)

        if not keep:
            continue

        face_tensors = [faces[j] for j in keep]
        kept_boxes = [boxes[j] for j in keep]

        if review_only:
            labels = ["unknown"] * len(face_tensors)
            scores = [0.0] * len(face_tensors)
        else:
            Efaces = emb.embed_tensors(face_tensors)
            if Efaces.size == 0:
                continue
            labels, scores = rec.predict(Efaces)

        for fi, (lab, sc, b) in enumerate(zip(labels, scores, kept_boxes), start=1):
            scanned_faces += 1
            label = str(lab)
            score = float(sc)

            if not review_only:
                if label == "unknown":
                    if not include_unknown:
                        continue
                else:
                    if score < float(min_score):
                        continue

            label_counts[label] = label_counts.get(label, 0)
            if label_counts[label] >= int(max_per_label):
                continue

            x1, y1, x2, y2 = [int(v) for v in b]
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            ex = int(round(bw * max(0.0, float(crop_expand))))
            ey = int(round(bh * max(0.0, float(crop_expand))))
            x1 -= ex
            y1 -= ey
            x2 += ex
            y2 += ey
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if int(x2 - x1) < int(min_crop_size) or int(y2 - y1) < int(min_crop_size):
                continue

            crop = img.crop((x1, y1, x2, y2))
            if float(min_focus_var) > 0.0:
                focus_val = _focus_measure_from_crop(crop)
                if focus_val < float(min_focus_var):
                    continue
            ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            fname = f"IRL_{img_path.stem}_f{fi}_s{score:.2f}_{ts}.jpg"

            if to_students and label != "unknown" and not review_only:
                dst_dir = students_dir / label
                if not dst_dir.exists():
                    # Keep safety first: unknown labels or missing class folders go to review.
                    dst_dir = review_root / label
            else:
                out_label = unknown_folder if label == "unknown" else label
                dst_dir = review_root / out_label

            dst_dir.mkdir(parents=True, exist_ok=True)
            crop.save(dst_dir / fname)
            label_counts[label] += 1
            saved += 1

    print("\nHarvest complete")
    print(f"  faces_scanned: {scanned_faces}")
    print(f"  crops_saved  : {saved}")
    print(f"  output_root  : {review_root if not to_students else students_dir}")
    if label_counts:
        print("  per_label:")
        for k in sorted(label_counts.keys()):
            print(f"    {k}: {label_counts[k]}")
