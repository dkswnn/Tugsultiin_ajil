from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import time


def cmd_process(
    cfg: Dict,
    paths: Dict[str, Path],
    annotate: bool = False,
    out_dir: Path | None = None,
    camera_once: bool = False,
        camera_capture_dir: Path | None = None,
) -> None:
    from PIL import Image
    from src.io_utils import load_embeddings_npz, list_captures, write_session_csv, annotate_and_save
    from src.detector import Detector
    from src.embedding import Embedder
    from src.recognizers import CosineGallery, KNNClassifier, HybridKnnCosine, AdaptiveGallery, ArcFaceHeadRecognizer, SVMClassifier
    import numpy as _np
    import numpy as np

    E, L = load_embeddings_npz(paths["embeddings_file"])
    if E is None or L is None or len(E) == 0:
        print(f"No gallery embeddings found at {paths['embeddings_file']}. Run 'python main.py build' first.")
        return

    images = list_captures(paths["captures_dir"])
    if camera_once:
        try:
            import cv2  # type: ignore
        except Exception:
            print("OpenCV is required for --camera capture. Install opencv-python.")
            return
        cam_cfg = cfg.get("camera", {})
        cap_cfg = cfg.get("capture", {})
        source = cam_cfg.get("source", 0)
        width = int(cam_cfg.get("width", 0) or 0)
        height = int(cam_cfg.get("height", 0) or 0)
        fps = int(cam_cfg.get("fps", 0) or 0)
        burst = int(cap_cfg.get("burst", 1) or 1)
        burst = max(1, burst)
        burst_gap_ms = int(cap_cfg.get("burst_gap_ms", 250) or 250)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open camera source: {source}")
            return
        try:
            if width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, fps)
            cap_dir = camera_capture_dir if camera_capture_dir is not None else paths["captures_dir"]
            cap_dir.mkdir(parents=True, exist_ok=True)
            captured: List[Path] = []
            for i in range(burst):
                ok = False
                frame = None
                for _ in range(6):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break
                if not ok or frame is None:
                    continue
                stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                suffix = f"_{i+1}" if burst > 1 else ""
                cap_name = f"CAM_{stamp}{suffix}.jpg"
                cap_path = cap_dir / cap_name
                cv2.imwrite(str(cap_path), frame)
                captured.append(cap_path)
                if i < burst - 1 and burst_gap_ms > 0:
                    time.sleep(burst_gap_ms / 1000.0)
            if not captured:
                print("Failed to capture frame from camera.")
                return
            images = captured
            print(f"Captured {len(captured)} image(s) from camera burst")
        finally:
            cap.release()
    elif not images:
        print(f"No images found in {paths['captures_dir']}")
        return

    print(f"Found {len(images)} image(s) to process in {paths['captures_dir']}")

    det_cfg = cfg.get("detector", {})
    facenet_cfg = cfg.get("facenet", {})
    det = Detector(
        keep_all=True,
        image_size=int(facenet_cfg.get("image_size", 160) or 160),
        margin=int(facenet_cfg.get("margin", 14) or 14),
        backend=str(det_cfg.get("backend", "retinaface") or "retinaface"),
        retina_confidence=float(det_cfg.get("retina_confidence", 0.8) or 0.8),
    )
    print(f"Detector backend: {str(det_cfg.get('backend', 'retinaface') or 'retinaface')}")
    emb_cfg = cfg.get("embedding", {})
    emb = Embedder(
        model_name=str(emb_cfg.get("model", "facenet")),
        fusion_alpha=float(emb_cfg.get("fusion_alpha", 0.35) or 0.35),
    )
    rec_cfg = cfg.get("recognition", {})
    thr = float(rec_cfg.get("decision_threshold", 0.8))
    margin_thr = rec_cfg.get("margin_threshold", None)
    margin_thr = float(margin_thr) if margin_thr is not None else None
    min_prob = float(rec_cfg.get("min_detect_prob", 0.9))
    min_box = int((rec_cfg.get("min_box_size", 0) or 0))
    cosine_gate = rec_cfg.get("cosine_gate", None)
    cosine_gate = float(cosine_gate) if cosine_gate is not None else None
    method = str(cfg.get("recognition", {}).get("method", "cosine")).lower()
    knn_k = int(rec_cfg.get("knn_k", 5))
    hybrid_alpha = float(rec_cfg.get("hybrid_alpha", 0.5))
    svm_c = float(rec_cfg.get("svm_c", 8.0) or 8.0)
    svm_kernel = str(rec_cfg.get("svm_kernel", "rbf") or "rbf")
    svm_gamma = rec_cfg.get("svm_gamma", "scale")
    flip_avg = bool(rec_cfg.get("flip_average", False))
    # De-duplication controls
    dedupe_per_image = bool(rec_cfg.get("dedupe_per_image", True))
    dedupe_session = bool(rec_cfg.get("dedupe_session", True))
    # Per-class gating controls
    use_pclass_gate = bool(rec_cfg.get("use_per_class_gate", False))
    pclass_gate_pct = float(rec_cfg.get("per_class_gate_percentile", 0.1))
    pclass_min_count = int(rec_cfg.get("per_class_gate_min_count", 8))
    pclass_gate_cap = float(rec_cfg.get("per_class_gate_cap", 0.85))
    # Confusion-pair margin gate (extra rejection only for look-alike class pairs)
    use_confusion_pair_margin = bool(rec_cfg.get("use_confusion_pair_margin", False))
    confusion_pair_topk = int(rec_cfg.get("confusion_pair_topk", 2) or 2)
    confusion_margin_extra = float(rec_cfg.get("confusion_margin_extra", 0.03) or 0.03)
    confusion_pair_min_cos = float(rec_cfg.get("confusion_pair_min_cos", 0.70) or 0.70)
    # Cohort-normalized open-set gate (score compared against impostor class distribution)
    use_cohort_norm = bool(rec_cfg.get("use_cohort_norm", False))
    cohort_z_threshold = float(rec_cfg.get("cohort_z_threshold", 0.9))
    # Quality-gate controls
    quality_gate_enabled = bool(rec_cfg.get("quality_gate_enabled", True))
    min_focus_var = float(rec_cfg.get("min_focus_var", 18.0))
    min_rel_face_area = float(rec_cfg.get("min_rel_face_area", 0.012))
    # Optional temporal voting controls
    temporal_vote_enabled = bool(rec_cfg.get("temporal_vote_enabled", False))
    temporal_vote_window = int(rec_cfg.get("temporal_vote_window", 2))
    temporal_vote_min_count = int(rec_cfg.get("temporal_vote_min_count", 2))
    max_unknown_per_image_cfg = rec_cfg.get("max_unknown_per_image", None)
    max_unknown_per_image = int(max_unknown_per_image_cfg) if max_unknown_per_image_cfg is not None else None

    if method == "arcface_head":
        model_path = paths.get("arcface_model_file")
        if model_path is None or not model_path.exists():
            print("ArcFace head model not found. Run 'python main.py train' first.")
            return
        rec = ArcFaceHeadRecognizer(model_path, threshold=thr, margin_threshold=margin_thr)
    elif method == "knn":
        from src.io_utils import load_embeddings_npz as _load
        Eall, Lall = _load(paths["embeddings_all_file"])
        if Eall is None or Lall is None or len(Eall) == 0:
            print(f"No per-image embeddings found at {paths['embeddings_all_file']}. Run 'python main.py build' first.")
            return
        rec = KNNClassifier(Eall, Lall, k=knn_k, threshold=thr, margin_threshold=margin_thr)
    elif method == "hybrid":
        from src.io_utils import load_embeddings_npz as _load
        Eall, Lall = _load(paths["embeddings_all_file"])
        if Eall is None or Lall is None or len(Eall) == 0:
            print(f"No per-image embeddings found at {paths['embeddings_all_file']}. Run 'python main.py build' first.")
            return
        # Optional Mahalanobis augmentation directly inside Hybrid
        maha_w = float(rec_cfg.get("mahalanobis_weight", 0.0) or 0.0)
        rec = HybridKnnCosine(Eall, Lall, E, L, k=knn_k, alpha=hybrid_alpha, threshold=thr, margin_threshold=margin_thr, mahalanobis_weight=maha_w)
    elif method == "adaptive":
        from src.io_utils import load_embeddings_npz as _load
        Eall, Lall = _load(paths["embeddings_all_file"])
        if Eall is None or Lall is None or len(Eall) == 0:
            print(f"No per-image embeddings found at {paths['embeddings_all_file']}. Run 'python main.py build' first.")
            return
        # weight controls cosine vs Mahalanobis balance; reuse alpha if provided
        weight = hybrid_alpha if hybrid_alpha is not None else 0.6
        rec = AdaptiveGallery(Eall, Lall, E, L, weight=weight, threshold=thr, margin_threshold=margin_thr)
    elif method == "svm":
        from src.io_utils import load_embeddings_npz as _load
        Eall, Lall = _load(paths["embeddings_all_file"])
        if Eall is None or Lall is None or len(Eall) == 0:
            print(f"No per-image embeddings found at {paths['embeddings_all_file']}. Run 'python main.py build' first.")
            return
        model_path = paths.get("svm_model_file")
        rec = SVMClassifier(
            Eall,
            Lall,
            threshold=thr,
            margin_threshold=margin_thr,
            c=svm_c,
            kernel=svm_kernel,
            gamma=svm_gamma,
            model_path=model_path,
        )
    else:
        rec = CosineGallery(E, L, threshold=thr, margin_threshold=margin_thr)

    label_to_mean = {str(label): _np.asarray(vec, dtype="float32") for vec, label in zip(E, L)}
    label_to_mean_norm = {
        k: (v / (float(_np.linalg.norm(v)) + 1e-8)).astype("float32")
        for k, v in label_to_mean.items()
    }
    cohort_labels = list(label_to_mean_norm.keys())
    cohort_means = _np.stack([label_to_mean_norm[k] for k in cohort_labels], axis=0).astype("float32") if cohort_labels else _np.zeros((0, E.shape[1]), dtype="float32")
    confusion_pairs: set[frozenset[str]] = set()
    label_to_idx = {lab: i for i, lab in enumerate(cohort_labels)}
    if use_confusion_pair_margin and cohort_means.shape[0] > 1:
        sims = cohort_means @ cohort_means.T
        ncls = sims.shape[0]
        for i in range(ncls):
            order = _np.argsort(-sims[i])
            kept = 0
            for j in order:
                if int(j) == i:
                    continue
                if float(sims[i, j]) < confusion_pair_min_cos:
                    continue
                confusion_pairs.add(frozenset((cohort_labels[i], cohort_labels[int(j)])))
                kept += 1
                if kept >= confusion_pair_topk:
                    break
    # Optional: compute per-class cosine gates from enrollment distribution
    per_class_mu: dict[str, _np.ndarray] = {}
    per_class_gate: dict[str, float] = {}
    if use_pclass_gate:
        from src.io_utils import load_embeddings_npz as _load
        Eall2, Lall2 = _load(paths["embeddings_all_file"])
        if Eall2 is not None and Lall2 is not None and len(Eall2) > 0:
            Ea = _np.asarray(Eall2, dtype="float32")
            Ea = Ea / (_np.linalg.norm(Ea, axis=1, keepdims=True) + 1e-8)
            labs = [str(l) for l in Lall2]
            for cls in sorted(set(labs)):
                idx = [i for i, Lc in enumerate(labs) if Lc == cls]
                if not idx or len(idx) < pclass_min_count:
                    continue
                X = Ea[idx]
                mu = X.mean(axis=0)
                mu = mu / (_np.linalg.norm(mu) + 1e-8)
                cos = X @ mu.astype("float32")
                gate_c = float(_np.percentile(cos, pclass_gate_pct * 100.0))
                gate_c = float(min(gate_c, pclass_gate_cap))
                per_class_mu[cls] = mu.astype("float32")
                per_class_gate[cls] = gate_c

    rows: List[Tuple[str, str, int, str, float]] = []

    def _focus_measure_from_crop(crop_img) -> float:
        arr = _np.asarray(crop_img, dtype="float32")
        if arr.ndim == 3:
            g = arr.mean(axis=2)
        else:
            g = arr
        if g.shape[0] < 4 or g.shape[1] < 4:
            return 0.0
        gx = _np.diff(g, axis=1)
        gy = _np.diff(g, axis=0)
        return float(_np.var(gx) + _np.var(gy))

    total_images = len(images)
    t0_all = time.time()
    for img_index, img_path in enumerate(images, start=1):
        t0_img = time.time()
        print(f"[{img_index}/{total_images}] Processing {img_path.name} ...")
        img = Image.open(img_path)
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        img = img.convert("RGB")
        faces, probs, boxes = det.detect_with_boxes(img)
        if faces and min_prob is not None:
            keep = [i for i, p in enumerate(probs) if p is not None and float(p) >= min_prob]
            faces = [faces[i] for i in keep]
            probs = [probs[i] for i in keep]
            if boxes:
                boxes = [boxes[i] for i in keep]
        if faces and min_box and boxes:
            def _box_ok(b):
                x1, y1, x2, y2 = b
                return (x2 - x1) >= min_box and (y2 - y1) >= min_box
            keep2 = [i for i, b in enumerate(boxes) if _box_ok(b)]
            faces = [faces[i] for i in keep2]
            probs = [probs[i] for i in keep2]
            boxes = [boxes[i] for i in keep2]

        if quality_gate_enabled and faces and boxes:
            w_img, h_img = img.size
            img_area = float(max(1, w_img * h_img))
            keep3: List[int] = []
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in b]
                x1 = max(0, min(x1, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                x2 = max(0, min(x2, w_img))
                y2 = max(0, min(y2, h_img))
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                rel_area = float((bw * bh) / img_area)
                if bw <= 1 or bh <= 1 or rel_area < min_rel_face_area:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                focus_val = _focus_measure_from_crop(crop)
                if focus_val < min_focus_var:
                    continue
                keep3.append(i)
            faces = [faces[i] for i in keep3]
            probs = [probs[i] for i in keep3]
            boxes = [boxes[i] for i in keep3]

        if not faces:
            rows.append((datetime.now().isoformat(timespec="seconds"), img_path.name, -1, "no_face", 0.0))
            if annotate:
                out_base = out_dir or (paths["attendance_dir"].parent / "outputs" / "annotated")
                out_path = out_base / img_path.name
                annotate_and_save(img.copy(), [], [], [], out_path)
            dt_img = time.time() - t0_img
            print(f"[{img_index}/{total_images}] done in {dt_img:.1f}s (no_face)")
            continue

        # Embed faces (optionally flip-average each)
        face_vecs: List[_np.ndarray] = []
        if flip_avg:
            try:
                import torch  # type: ignore
                for f in faces:
                    flipped = torch.flip(f, dims=[2])
                    Ef_pair = emb.embed_tensors([f, flipped])
                    if Ef_pair.size == 0:
                        continue
                    v = Ef_pair.mean(axis=0)
                    v = v / (np.linalg.norm(v) + 1e-8)
                    face_vecs.append(v.astype("float32"))
            except Exception:
                Ef_all = emb.embed_tensors(faces)
                for v in Ef_all:
                    face_vecs.append(v.astype("float32"))
        else:
            Ef_all = emb.embed_tensors(faces)
            for v in Ef_all:
                face_vecs.append(v.astype("float32"))
        if not face_vecs:
            rows.append((datetime.now().isoformat(timespec="seconds"), img_path.name, -1, "no_face", 0.0))
            if annotate:
                out_base = out_dir or (paths["attendance_dir"].parent / "outputs" / "annotated")
                out_path = out_base / img_path.name
                annotate_and_save(img.copy(), boxes or [], [], [], out_path)
            dt_img = time.time() - t0_img
            print(f"[{img_index}/{total_images}] done in {dt_img:.1f}s (no_face)")
            continue
        Efaces = _np.stack(face_vecs, axis=0).astype("float32")

        labels, scores = rec.predict(Efaces)
        if cosine_gate is not None or use_pclass_gate or use_cohort_norm or use_confusion_pair_margin:
            for i in range(len(labels)):
                lab = str(labels[i])
                if lab == "unknown":
                    continue
                # Prefer per-class mu when available, else fall back to gallery mean
                cm = per_class_mu.get(lab) if use_pclass_gate and lab in per_class_mu else label_to_mean_norm.get(lab)
                if cm is None:
                    continue
                cos = float(_np.dot(Efaces[i].astype("float32"), cm.astype("float32")))
                # Determine applicable threshold
                thr_c = cosine_gate if cosine_gate is not None else -1.0
                if use_pclass_gate and lab in per_class_gate:
                    # Use the stricter between global and per-class, but cap was already applied above
                    thr_c = max(thr_c, per_class_gate[lab])
                if cos < thr_c:
                    labels[i] = "unknown"
                    continue
                if (use_cohort_norm or use_confusion_pair_margin) and cohort_means.shape[0] > 1 and lab in label_to_mean_norm:
                    all_cos = cohort_means @ Efaces[i].astype("float32")
                    pred_idx = label_to_idx.get(lab, -1)
                    if pred_idx >= 0:
                        # second-closest class under centroid-cosine
                        second_idx = int(_np.argsort(-all_cos)[1]) if all_cos.shape[0] > 1 else -1
                        second_lab = cohort_labels[second_idx] if second_idx >= 0 else ""
                        second_cos = float(all_cos[second_idx]) if second_idx >= 0 else -1.0
                        mean_margin = float(cos - second_cos)

                        if use_confusion_pair_margin and second_idx >= 0:
                            pair_key = frozenset((lab, second_lab))
                            if pair_key in confusion_pairs and mean_margin < confusion_margin_extra:
                                labels[i] = "unknown"
                                continue

                        imp = _np.delete(all_cos, pred_idx)
                        if imp.size > 0:
                            imp_mu = float(_np.mean(imp))
                            imp_sd = float(_np.std(imp) + 1e-8)
                            z = (cos - imp_mu) / imp_sd
                            if float(z) < cohort_z_threshold:
                                labels[i] = "unknown"

        # Optional per-image de-duplication for known labels: keep the highest score per label
        keep_indices = list(range(len(labels)))
        if dedupe_per_image and len(labels) > 1:
            best_per_label: dict[str, tuple[float, int]] = {}
            for i, (lab, sc) in enumerate(zip(labels, scores)):
                lab_s = str(lab)
                if lab_s == "unknown":
                    continue
                prev = best_per_label.get(lab_s)
                if prev is None or float(sc) > prev[0]:
                    best_per_label[lab_s] = (float(sc), i)
            keep_set = set(best_per_label[i][1] for i in best_per_label)
            # Always keep unknowns (different people) and known best indices
            keep_indices = [i for i in range(len(labels)) if str(labels[i]) == "unknown" or i in keep_set]

        # Cap unknown rows per image to reduce noisy/background detections.
        if max_unknown_per_image is not None and max_unknown_per_image >= 0:
            unknown_idx = [i for i in keep_indices if str(labels[i]) == "unknown"]
            if len(unknown_idx) > max_unknown_per_image:
                unknown_idx_sorted = sorted(unknown_idx, key=lambda i: float(scores[i]), reverse=True)
                unknown_keep = set(unknown_idx_sorted[:max_unknown_per_image])
                keep_indices = [
                    i for i in keep_indices
                    if str(labels[i]) != "unknown" or i in unknown_keep
                ]

        # Prepare rows and annotation based on kept indices
        ts = datetime.now().isoformat(timespec="seconds")
        for i in keep_indices:
            rows.append((ts, img_path.name, i, str(labels[i]), float(scores[i])))

        if annotate:
            out_base = out_dir or (paths["attendance_dir"].parent / "outputs" / "annotated")
            out_path = out_base / img_path.name
            if boxes:
                sel_boxes = [boxes[i] for i in keep_indices]
            else:
                sel_boxes = []
            sel_labels = [str(labels[i]) for i in keep_indices]
            sel_scores = [float(scores[i]) for i in keep_indices]
            annotate_and_save(img.copy(), sel_boxes, sel_labels, sel_scores, out_path)

        known_count = sum(1 for i in keep_indices if str(labels[i]) not in ("unknown", "no_face"))
        unknown_count = sum(1 for i in keep_indices if str(labels[i]) == "unknown")
        dt_img = time.time() - t0_img
        print(f"[{img_index}/{total_images}] done in {dt_img:.1f}s (known={known_count}, unknown={unknown_count})")

    session_dir = paths["attendance_dir"] / "sessions"
    exp_tag_raw = str(rec_cfg.get("experiment_tag", "") or "").strip()
    exp_tag = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in exp_tag_raw)
    ts_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_name = f"{exp_tag}_{ts_name}.csv" if exp_tag else f"{ts_name}.csv"
    csv_path = session_dir / session_name
    # Optional de-duplication across session: keep one entry per known label with highest score
    if temporal_vote_enabled and camera_once and len(images) > 1 and temporal_vote_window > 0 and temporal_vote_min_count > 1 and rows:
        known_idx = [i for i, r in enumerate(rows) if r[3] not in ("unknown", "no_face")]
        known_labels = [str(rows[i][3]) for i in known_idx]
        if len(known_labels) >= temporal_vote_min_count:
            to_unknown: set[int] = set()
            for pos, ridx in enumerate(known_idx):
                lo = max(0, pos - temporal_vote_window)
                hi = min(len(known_idx), pos + temporal_vote_window + 1)
                window_labels = known_labels[lo:hi]
                cur = known_labels[pos]
                votes = sum(1 for lab in window_labels if lab == cur)
                if votes < temporal_vote_min_count:
                    to_unknown.add(ridx)
            if to_unknown:
                rows = [
                    (ts, fname, face_idx, "unknown", 0.0) if i in to_unknown else (ts, fname, face_idx, lab, sc)
                    for i, (ts, fname, face_idx, lab, sc) in enumerate(rows)
                ]

    if dedupe_session:
        best_row_per_label: dict[str, tuple[str, str, int, str, float]] = {}
        kept_rows = []
        for r in rows:
            ts, fname, face_idx, lab, sc = r
            if lab in ("unknown", "no_face"):
                kept_rows.append(r)
                continue
            prev = best_row_per_label.get(lab)
            if prev is None or sc > prev[4]:
                best_row_per_label[lab] = r
        # Combine known best rows and passthrough rows
        kept_rows.extend(best_row_per_label.values())
        rows = kept_rows

    write_session_csv(csv_path, rows)
    dt_all = time.time() - t0_all
    print(f"Wrote {len(rows)} rows -> {csv_path}")
    print(f"Finished processing in {dt_all:.1f}s")
