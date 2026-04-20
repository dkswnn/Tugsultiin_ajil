from __future__ import annotations

from pathlib import Path
from typing import Dict


def cmd_tune(cfg: Dict, paths: Dict[str, Path]) -> None:
    from PIL import Image
    import numpy as np
    from src.io_utils import load_embeddings_npz, list_captures
    from src.detector import Detector
    from src.embedding import Embedder
    from src.recognizers import CosineGallery, KNNClassifier, HybridKnnCosine

    rec_cfg = cfg.get("recognition", {})
    method = str(rec_cfg.get("method", "cosine")).lower()
    images = list_captures(paths["captures_dir"])
    if not images:
        print(f"No images found in {paths['captures_dir']}")
        return

    # Load galleries
    Emean, Lmean = load_embeddings_npz(paths["embeddings_file"])
    if Emean is None or Lmean is None or len(Emean) == 0:
        print("No gallery means. Run 'python main.py build' first.")
        return
    Eall = Lall = None
    if method in ("knn", "hybrid"):
        Eall, Lall = load_embeddings_npz(paths["embeddings_all_file"])
        if Eall is None or Lall is None or len(Eall) == 0:
            print("No per-image gallery. Run 'python main.py build' first.")
            return

    # Detect once and embed
    det_cfg = cfg.get("detector", {})
    facenet_cfg = cfg.get("facenet", {})
    det = Detector(
        keep_all=True,
        image_size=int(facenet_cfg.get("image_size", 160) or 160),
        margin=int(facenet_cfg.get("margin", 14) or 14),
        backend=str(det_cfg.get("backend", "retinaface") or "retinaface"),
        retina_confidence=float(det_cfg.get("retina_confidence", 0.8) or 0.8),
    )
    emb = Embedder(model_name=str(cfg.get("embedding", {}).get("model", "facenet")))
    min_prob = float(rec_cfg.get("min_detect_prob", 0.9))
    min_box = int((rec_cfg.get("min_box_size", 0) or 0))

    feats: list[np.ndarray] = []
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        faces, probs, boxes = det.detect_with_boxes(img)
        if faces and min_prob is not None:
            keep = [i for i, p in enumerate(probs) if p is not None and float(p) >= min_prob]
            faces = [faces[i] for i in keep]
            boxes = [boxes[i] for i in keep] if boxes else []
        if faces and min_box and boxes:
            def _box_ok(b):
                x1, y1, x2, y2 = b
                return (x2 - x1) >= min_box and (y2 - y1) >= min_box
            keep2 = [i for i, b in enumerate(boxes) if _box_ok(b)]
            faces = [faces[i] for i in keep2]
        if faces:
            Ef = emb.embed_tensors(faces)
            if Ef.size:
                feats.append(Ef)

    if not feats:
        print("No faces found after filtering. Relax detection filters and try again.")
        return
    Q = np.concatenate(feats, axis=0)
    print(f"Tuning on {Q.shape[0]} detected faces.")

    # Grids
    dths = [0.7, 0.75, 0.8, 0.85]
    mths = [0.05, 0.1, 0.2]
    cgs = [None, 0.85, 0.9]
    ks = [3, 5] if method in ("knn", "hybrid") else [None]
    alphas = [0.3, 0.5, 0.7] if method == "hybrid" else [None]
    maha_ws = [0.0, 0.1, 0.2, 0.3] if method == "hybrid" else [None]

    # Precompute Mahalanobis stats once for hybrid
    precomputed_mahal = None
    if method == "hybrid":
        from numpy.linalg import norm as _norm
        Ea = Eall.astype("float32")
        Ea = Ea / (_norm(Ea, axis=1, keepdims=True) + 1e-8)
        labs = [str(l) for l in Lall]
        uniq = sorted(set(labs))
        cm = {}
        ci = {}
        for cls in uniq:
            idx = [i for i, L in enumerate(labs) if L == cls]
            if not idx:
                continue
            X = Ea[idx]
            mu = X.mean(axis=0)
            mu = mu / (_norm(mu) + 1e-8)
            C = np.cov(X.T)
            C = C + (1e-2) * np.eye(C.shape[0], dtype=C.dtype)
            inv = np.linalg.pinv(C)
            cm[cls] = mu.astype("float32")
            ci[cls] = inv.astype("float32")
        precomputed_mahal = {"class_mean": cm, "class_inv": ci}

    results = []
    total_combos = (
        (len(ks) if ks[0] is not None else 1)
        * len(dths)
        * len(mths)
        * (len(alphas) if alphas[0] is not None else 1)
        * (len(maha_ws) if maha_ws[0] is not None else 1)
        * len(cgs)
    )
    processed = 0

    for k in ks:
        for dt in dths:
            for mt in mths:
                for a in alphas:
                    for mw in maha_ws:
                        if method == "knn":
                            rec = KNNClassifier(Eall, Lall, k=int(k), threshold=dt, margin_threshold=mt)
                        elif method == "hybrid":
                            rec = HybridKnnCosine(
                                Eall,
                                Lall,
                                Emean,
                                Lmean,
                                k=int(k),
                                alpha=float(a),
                                threshold=dt,
                                margin_threshold=mt,
                                mahalanobis_weight=(float(mw) if mw is not None else 0.0),
                                precomputed_mahal=precomputed_mahal,
                            )
                        else:
                            rec = CosineGallery(Emean, Lmean, threshold=dt, margin_threshold=mt)
                        labels, _ = rec.predict(Q)
                        total = len(labels)
                        unk = int(np.sum(labels == "unknown"))
                        unk_rate = unk / max(1, total)
                        for cg in cgs:
                            if cg is None:
                                results.append((unk_rate, {"k": k, "decision": dt, "margin": mt, "alpha": a, "mahalanobis_weight": mw, "cosine_gate": None}))
                                processed += 1
                                if processed % 50 == 0 or processed == total_combos:
                                    pct = (processed / total_combos) * 100.0
                                    print(f"Progress: {processed}/{total_combos} ({pct:.1f}%)")
                                continue
                            l2m = {str(label): vec for vec, label in zip(Emean, Lmean)}
                            lab_adj = labels.copy().astype(object)
                            from numpy.linalg import norm as _norm
                            for i in range(total):
                                lab = str(lab_adj[i])
                                if lab == "unknown":
                                    continue
                                cm = l2m.get(lab)
                                if cm is None:
                                    lab_adj[i] = "unknown"
                                    continue
                                cmn = cm / (_norm(cm) + 1e-8)
                                cos = float(np.dot(Q[i].astype("float32"), cmn.astype("float32")))
                                if cos < cg:
                                    lab_adj[i] = "unknown"
                            unk2 = int(np.sum(lab_adj == "unknown"))
                            unk_rate2 = unk2 / max(1, total)
                            results.append((unk_rate2, {"k": k, "decision": dt, "margin": mt, "alpha": a, "mahalanobis_weight": mw, "cosine_gate": cg}))
                            processed += 1
                            if processed % 50 == 0 or processed == total_combos:
                                pct = (processed / total_combos) * 100.0
                                print(f"Progress: {processed}/{total_combos} ({pct:.1f}%)")

    results.sort(key=lambda x: x[0])
    print("Top configs by lowest unknown rate:")
    for r, cfgrow in results[:5]:
        print(f"  unknown_rate={r:.3f}  cfg={cfgrow}")
