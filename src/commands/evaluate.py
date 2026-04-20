from __future__ import annotations

from pathlib import Path
from typing import Dict


def cmd_evaluate(cfg: Dict, paths: Dict[str, Path]) -> None:
    """Leave-one-out evaluation comparing cosine/knn/hybrid/svm (closed-set and gated)."""
    import numpy as np
    from collections import defaultdict
    from src.io_utils import load_embeddings_npz
    from src.recognizers import CosineGallery, KNNClassifier, HybridKnnCosine, SVMClassifier

    Emean, Lmean = load_embeddings_npz(paths["embeddings_file"])
    Eall, Lall = load_embeddings_npz(paths["embeddings_all_file"])
    if Emean is None or Lmean is None or Eall is None or Lall is None:
        print("Missing embeddings. Run 'python main.py build' first.")
        return

    label_to_idx = defaultdict(list)
    for i, lab in enumerate(Lall):
        label_to_idx[str(lab)].append(i)

    rec_cfg = cfg.get("recognition", {})
    thr = rec_cfg.get("decision_threshold", None)
    thr = float(thr) if thr is not None else None
    mthr = rec_cfg.get("margin_threshold", None)
    mthr = float(mthr) if mthr is not None else None
    knn_k = int(rec_cfg.get("knn_k", 5))
    alpha = float(rec_cfg.get("hybrid_alpha", 0.5))
    svm_c = float(rec_cfg.get("svm_c", 8.0) or 8.0)
    svm_kernel = str(rec_cfg.get("svm_kernel", "rbf") or "rbf")
    svm_gamma = rec_cfg.get("svm_gamma", "scale")
    cosine_gate = rec_cfg.get("cosine_gate", None)
    cosine_gate = float(cosine_gate) if cosine_gate is not None else None

    totals = {"cosine": 0, "knn": 0, "hybrid": 0, "svm": 0}
    correct_closed = {"cosine": 0, "knn": 0, "hybrid": 0, "svm": 0}
    correct_gated = {"cosine": 0, "knn": 0, "hybrid": 0, "svm": 0}

    def apply_cosine_gate(pred_labels, feats, mean_vecs, mean_labs):
        if cosine_gate is None:
            return pred_labels
        mv = mean_vecs.astype("float32")
        mv = mv / (np.linalg.norm(mv, axis=1, keepdims=True) + 1e-8)
        l2m = {str(lab): vec for vec, lab in zip(mv, mean_labs)}
        out = pred_labels.copy().astype(object)
        for i in range(len(out)):
            lab = str(out[i])
            if lab == "unknown":
                continue
            cm = l2m.get(lab)
            if cm is None:
                out[i] = "unknown"
                continue
            q = feats[i].astype("float32")
            q = q / (np.linalg.norm(q) + 1e-8)
            cos = float(np.dot(q, cm))
            if cos < cosine_gate:
                out[i] = "unknown"
        return out

    for lab, idxs in label_to_idx.items():
        if len(idxs) < 2:
            continue
        for test_idx in idxs:
            train_idx = [i for i in range(len(Lall)) if i != test_idx]
            Etrain = Eall[train_idx].astype("float32")
            Ltrain = Lall[train_idx]

            per_label = {lbl: [i for i in all_i if i != test_idx] for lbl, all_i in label_to_idx.items()}
            mean_vecs = []
            mean_labs = []
            for lbl, lst in per_label.items():
                if not lst:
                    continue
                M = Eall[lst].astype("float32")
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
                mv = M.mean(axis=0)
                mv = mv / (np.linalg.norm(mv) + 1e-8)
                mean_vecs.append(mv)
                mean_labs.append(lbl)
            if not mean_vecs:
                continue
            Emean_fold = np.stack(mean_vecs, axis=0)
            Lmean_fold = np.array(mean_labs, dtype=object)

            q = Eall[test_idx][None, :]
            y_true = str(Lall[test_idx])

            cos_rec_closed = CosineGallery(Emean_fold, Lmean_fold, threshold=None, margin_threshold=None)
            knn_rec_closed = KNNClassifier(Etrain, Ltrain, k=knn_k, threshold=None, margin_threshold=None)
            hyb_rec_closed = HybridKnnCosine(Etrain, Ltrain, Emean_fold, Lmean_fold, k=knn_k, alpha=alpha, threshold=None, margin_threshold=None)
            svm_rec_closed = SVMClassifier(Etrain, Ltrain, threshold=None, margin_threshold=None, c=svm_c, kernel=svm_kernel, gamma=svm_gamma)
            for name, rec in (("cosine", cos_rec_closed), ("knn", knn_rec_closed), ("hybrid", hyb_rec_closed), ("svm", svm_rec_closed)):
                pred, _ = rec.predict(q)
                totals[name] += 1
                if str(pred[0]) == y_true:
                    correct_closed[name] += 1

            cos_rec = CosineGallery(Emean_fold, Lmean_fold, threshold=thr, margin_threshold=mthr)
            knn_rec = KNNClassifier(Etrain, Ltrain, k=knn_k, threshold=thr, margin_threshold=mthr)
            hyb_rec = HybridKnnCosine(Etrain, Ltrain, Emean_fold, Lmean_fold, k=knn_k, alpha=alpha, threshold=thr, margin_threshold=mthr)
            svm_rec = SVMClassifier(Etrain, Ltrain, threshold=thr, margin_threshold=mthr, c=svm_c, kernel=svm_kernel, gamma=svm_gamma)
            for name, rec in (("cosine", cos_rec), ("knn", knn_rec), ("hybrid", hyb_rec), ("svm", svm_rec)):
                pred, _ = rec.predict(q)
                pred = apply_cosine_gate(pred, q, Emean_fold, Lmean_fold)
                if str(pred[0]) == y_true:
                    correct_gated[name] += 1

    if sum(totals.values()) == 0:
        print("Not enough images per person to evaluate (need >=2). Add more and rebuild.")
        return

    print("\nEvaluation (leave-one-out per person):")
    for name in ("cosine", "knn", "hybrid", "svm"):
        tot = totals[name]
        cc = correct_closed[name]
        cg = correct_gated[name]
        acc_closed = cc / tot if tot else 0.0
        acc_gated = cg / tot if tot else 0.0
        print(f"  {name:7s}  closed-set acc = {acc_closed:.3f}   gated acc = {acc_gated:.3f}   (N={tot})")

    print("\nParameter sweep (gated acc):")
    ks = [3, 5, 7]
    alphas = [0.3, 0.5, 0.7]

    knn_correct = {k: 0 for k in ks}
    hyb_correct = {(k, a): 0 for k in ks for a in alphas}
    total_eval = 0

    for lab, idxs in label_to_idx.items():
        if len(idxs) < 2:
            continue
        for test_idx in idxs:
            train_idx = [i for i in range(len(Lall)) if i != test_idx]
            Etrain = Eall[train_idx].astype("float32")
            Ltrain = Lall[train_idx]

            per_label = {lbl: [i for i in all_i if i != test_idx] for lbl, all_i in label_to_idx.items()}
            mean_vecs = []
            mean_labs = []
            for lbl, lst in per_label.items():
                if not lst:
                    continue
                M = Eall[lst].astype("float32")
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
                mv = M.mean(axis=0)
                mv = mv / (np.linalg.norm(mv) + 1e-8)
                mean_vecs.append(mv)
                mean_labs.append(lbl)
            if not mean_vecs:
                continue
            Emean_fold = np.stack(mean_vecs, axis=0)
            Lmean_fold = np.array(mean_labs, dtype=object)

            q = Eall[test_idx][None, :]
            y_true = str(Lall[test_idx])

            for k in ks:
                rec_knn = KNNClassifier(Etrain, Ltrain, k=int(k), threshold=thr, margin_threshold=mthr)
                pred, _ = rec_knn.predict(q)
                pred = apply_cosine_gate(pred, q, Emean_fold, Lmean_fold)
                if str(pred[0]) == y_true:
                    knn_correct[k] += 1

            for k in ks:
                for a in alphas:
                    rec_h = HybridKnnCosine(Etrain, Ltrain, Emean_fold, Lmean_fold, k=int(k), alpha=float(a), threshold=thr, margin_threshold=mthr)
                    pred, _ = rec_h.predict(q)
                    pred = apply_cosine_gate(pred, q, Emean_fold, Lmean_fold)
                    if str(pred[0]) == y_true:
                        hyb_correct[(k, a)] += 1

            total_eval += 1

    if total_eval:
        print("\nBest KNN settings (gated):")
        knn_rank = sorted(((knn_correct[k] / total_eval, k) for k in ks), reverse=True)
        for acc, k in knn_rank[:3]:
            print(f"  k={k:<2}  acc={acc:.3f}")

        print("\nBest Hybrid settings (gated):")
        hyb_rank = sorted(((hyb_correct[(k, a)] / total_eval, k, a) for k in ks for a in alphas), reverse=True)
        for acc, k, a in hyb_rank[:5]:
            print(f"  k={k:<2}  alpha={a:.2f}  acc={acc:.3f}")

        best_h = hyb_rank[0] if hyb_rank else None
        if best_h:
            acc, bk, ba = best_h
            print("\nSuggested config (hybrid):")
            print("recognition:")
            print("  method: hybrid")
            print(f"  knn_k: {bk}")
            print(f"  hybrid_alpha: {ba}")
            if thr is not None:
                print(f"  decision_threshold: {thr}")
            if mthr is not None:
                print(f"  margin_threshold: {mthr}")
            print(f"  # gated acc≈{acc:.3f} on enrollment LOO")
