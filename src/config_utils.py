from __future__ import annotations

from pathlib import Path
from typing import Dict


DEFAULT_CFG: Dict = {
    "paths": {
        "students_dir": "data/students",
        "embeddings_file": "data/embeddings.npz",
        "embeddings_all_file": "data/embeddings_all.npz",
        "embeddings_aug_file": "data/embeddings_aug.npz",
        "embeddings_all_aug_file": "data/embeddings_all_aug.npz",
        "models_dir": "models",
        "svm_model_file": "models/classifier.joblib",
        "arcface_model_file": "models/arcface_head.pt",
        "captures_dir": "captures",
        "attendance_dir": "attendance",
    },
    "embedding": {
        "model": "facenet",
        "enroll_aug_enabled": False,
        "enroll_aug_count": 2,
        "enroll_aug_brightness_delta": 0.12,
        "enroll_aug_contrast_delta": 0.12,
    },
    "detector": {
        "backend": "retinaface",  # "retinaface" or "mtcnn"
        "retina_confidence": 0.8,
    },
    "training": {
        "enabled": True,
        "epochs": 50,
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 5e-5,
        "hidden_dim": 384,
        "dropout": 0.45,
        "arcface_scale": 30.0,
        "arcface_margin": 0.5,
        "val_ratio": 0.2,
        "seed": 42,
        "augmentations_per_image": 8,
        "min_detect_prob": 0.9,
        "unknown_threshold": 0.62,
        "margin_threshold": 0.08,
        "augmentation": {
            "hflip_prob": 0.8,
            "rotate_deg": 35,
            "affine_prob": 0.9,
            "brightness": 0.5,
            "contrast": 0.5,
            "color_prob": 0.85,
            "blur_prob": 0.5,
            "occlusion_prob": 0.6,
            "max_occlusion_frac": 0.30
        }
    },
    "enrollment": {
        "min_cos_to_center": 0.65,
        "max_per_person": None,
    },
    "capture": {
        "periods": [
            {"name": "1st lesson", "start": "08:00", "end": "09:30"},
            {"name": "2nd lesson", "start": "09:35", "end": "11:05"},
            {"name": "3rd lesson", "start": "11:10", "end": "12:40"},
            {"name": "4th lesson", "start": "13:20", "end": "14:50"},
            {"name": "5th lesson", "start": "14:55", "end": "16:25"},
        ],
    },
    "recognition": {
        "method": "hybrid",  # "cosine", "knn", "hybrid", "svm", "arcface_head"
        "decision_threshold": 0.70,
        "margin_threshold": 0.05,
        "min_detect_prob": 0.85,
        "min_box_size": 24,
        "cosine_gate": None,
        "use_per_class_gate": False,
        "per_class_gate_percentile": 0.10,
        "per_class_gate_min_count": 8,
        "per_class_gate_cap": 0.85,
        "use_cohort_norm": False,
        "cohort_z_threshold": 0.9,
        "use_confusion_pair_margin": False,
        "confusion_pair_topk": 2,
        "confusion_margin_extra": 0.03,
        "confusion_pair_min_cos": 0.70,
        "quality_gate_enabled": False,
        "min_focus_var": 18.0,
        "min_rel_face_area": 0.012,
        "temporal_vote_enabled": False,
        "temporal_vote_window": 2,
        "temporal_vote_min_count": 2,
        "periodic_interval_minutes": 5.0,
        "periodic_required_hits": 3,
        "periodic_min_confirm_score": 0.75,
        "periodic_min_gap_minutes": 2.0,
        "knn_k": 7,
        "hybrid_alpha": 0.7,
        "svm_c": 8.0,
        "svm_kernel": "rbf",
        "svm_gamma": "scale",
        "flip_average": True,
        "multi_centroids": True,
        "max_centroids": 3,
    },
}


def load_config(path: Path) -> Dict:
    try:
        import yaml  # type: ignore
    except Exception:
        print("[info] PyYAML not installed; using default config.")
        return DEFAULT_CFG.copy()
    if not path.exists():
        print(f"[info] Config file not found at {path}; using default config.")
        return DEFAULT_CFG.copy()
    with path.open("r", encoding="utf-8") as f:
        data = (yaml.safe_load(f) or {})
    cfg = DEFAULT_CFG.copy()
    cfg_paths = cfg.get("paths", {}).copy()
    cfg_rec = cfg.get("recognition", {}).copy()
    user_paths = (data.get("paths") or {})
    user_rec = (data.get("recognition") or {})
    cfg_paths.update(user_paths)
    cfg_rec.update(user_rec)
    cfg["paths"] = cfg_paths
    cfg["recognition"] = cfg_rec
    # pass-through any other top-level keys (enrollment, etc.)
    for k, v in data.items():
        if k not in ("paths", "recognition"):
            cfg[k] = v
    return cfg


def resolve_paths(cfg: Dict, root: Path) -> Dict[str, Path]:
    p = cfg.get("paths", {})
    return {
        "students_dir": (root / p.get("students_dir", "data/students")).resolve(),
        "embeddings_file": (root / p.get("embeddings_file", "data/embeddings.npz")).resolve(),
        "embeddings_all_file": (root / p.get("embeddings_all_file", "data/embeddings_all.npz")).resolve(),
        "embeddings_aug_file": (root / p.get("embeddings_aug_file", "data/embeddings_aug.npz")).resolve(),
        "embeddings_all_aug_file": (root / p.get("embeddings_all_aug_file", "data/embeddings_all_aug.npz")).resolve(),
        "models_dir": (root / p.get("models_dir", "models")).resolve(),
        "svm_model_file": (root / p.get("svm_model_file", "models/classifier.joblib")).resolve(),
        "arcface_model_file": (root / p.get("arcface_model_file", "models/arcface_head.pt")).resolve(),
        "captures_dir": (root / p.get("captures_dir", "captures")).resolve(),
        "attendance_dir": (root / p.get("attendance_dir", "attendance")).resolve(),
    }
