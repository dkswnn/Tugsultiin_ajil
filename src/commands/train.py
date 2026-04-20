from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def cmd_train(cfg: Dict, paths: Dict[str, Path]) -> None:
    import json
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from src.detector import Detector
    from src.embedding import Embedder
    from src.io_utils import iter_images, list_person_dirs

    train_cfg = cfg.get("training", {})
    aug_cfg = train_cfg.get("augmentation", {})
    rec_cfg = cfg.get("recognition", {})

    students_dir = paths["students_dir"]
    out_model = paths["arcface_model_file"]
    out_model.parent.mkdir(parents=True, exist_ok=True)

    if not students_dir.exists():
        print(f"Students dir not found: {students_dir}")
        return

    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    min_prob = float(train_cfg.get("min_detect_prob", rec_cfg.get("min_detect_prob", 0.9)))
    aug_n = int(max(0, train_cfg.get("augmentations_per_image", 2)))

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

    color_jitter = None
    blur_aug = None
    try:
        from torchvision.transforms import ColorJitter, GaussianBlur

        color_jitter = ColorJitter(
            brightness=float(aug_cfg.get("brightness", 0.25)),
            contrast=float(aug_cfg.get("contrast", 0.25)),
            saturation=0.15,
            hue=0.03,
        )
        blur_aug = GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
    except Exception:
        pass

    def _to_01(face_t: torch.Tensor) -> torch.Tensor:
        x = face_t.detach().cpu().float()
        if float(x.min()) < 0.0:
            x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    def _rand_occlude(x: torch.Tensor) -> torch.Tensor:
        max_frac = float(aug_cfg.get("max_occlusion_frac", 0.15))
        h = x.shape[1]
        w = x.shape[2]
        oh = max(1, int(h * random.uniform(0.06, max_frac)))
        ow = max(1, int(w * random.uniform(0.06, max_frac)))
        y0 = random.randint(0, max(0, h - oh))
        x0 = random.randint(0, max(0, w - ow))
        x[:, y0 : y0 + oh, x0 : x0 + ow] = 0.0
        return x

    def _augment_face(face_t: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.functional import affine, hflip

        x = _to_01(face_t)
        if random.random() < float(aug_cfg.get("hflip_prob", 0.5)):
            x = hflip(x)

        if random.random() < float(aug_cfg.get("affine_prob", 0.6)):
            deg = float(aug_cfg.get("rotate_deg", 12))
            angle = random.uniform(-deg, deg)
            scale = random.uniform(0.92, 1.08)
            tx = random.uniform(-0.03, 0.03) * x.shape[2]
            ty = random.uniform(-0.03, 0.03) * x.shape[1]
            x = affine(x, angle=angle, translate=[int(tx), int(ty)], scale=scale, shear=[0.0, 0.0])

        if color_jitter is not None and random.random() < float(aug_cfg.get("color_prob", 0.7)):
            x = color_jitter(x)

        if blur_aug is not None and random.random() < float(aug_cfg.get("blur_prob", 0.2)):
            x = blur_aug(x)

        if random.random() < float(aug_cfg.get("occlusion_prob", 0.2)):
            x = _rand_occlude(x)

        return x.clamp(0.0, 1.0)

    X: List[np.ndarray] = []
    Y: List[int] = []
    class_names: List[str] = []

    person_dirs = list_person_dirs(students_dir)
    if not person_dirs:
        print("No person folders found.")
        return

    for cls_idx, person_dir in enumerate(person_dirs):
        count_person = 0
        for img_path in iter_images(person_dir):
            img = Image.open(img_path)
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            img = img.convert("RGB")

            faces, probs = det.detect(img)
            if not faces:
                continue
            best_idx = int(np.argmax(np.asarray(probs)))
            if probs[best_idx] < min_prob:
                continue

            base_face = faces[best_idx]
            emb_base = emb.embed_tensors([base_face])
            if emb_base.size == 0:
                continue
            X.append(emb_base[0].astype("float32"))
            Y.append(len(class_names))  # Use current length as index
            count_person += 1

            for _ in range(aug_n):
                aug_face = _augment_face(base_face)
                emb_aug = emb.embed_tensors([aug_face])
                if emb_aug.size == 0:
                    continue
                X.append(emb_aug[0].astype("float32"))
                Y.append(len(class_names))  # Use current length as index
        
        if count_person > 0:  # Only add class if it has training data
            class_names.append(person_dir.name)
            print(f"[data] {person_dir.name}: {count_person} base faces")
        else:
            print(f"[skip] {person_dir.name}: 0 base faces (no training data)")

    if not X:
        print("No training samples collected. Check student images / detection threshold.")
        return

    Xn = np.asarray(X, dtype="float32")
    Yn = np.asarray(Y, dtype=np.int64)
    num_classes = int(len(class_names))

    if num_classes < 2:
        print("Need at least 2 classes to train.")
        return

    idx = np.arange(len(Yn))
    np.random.shuffle(idx)
    Xn = Xn[idx]
    Yn = Yn[idx]

    val_ratio = float(train_cfg.get("val_ratio", 0.2))
    split = int(len(Yn) * (1.0 - val_ratio))
    split = max(1, min(split, len(Yn) - 1))

    Xtr, Ytr = Xn[:split], Yn[:split]
    Xva, Yva = Xn[split:], Yn[split:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = int(Xtr.shape[1])
    hidden_dim = int(train_cfg.get("hidden_dim", 256))
    dropout = float(train_cfg.get("dropout", 0.2))

    class ArcMarginProduct(nn.Module):
        def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.s = s
            self.m = m
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.xavier_uniform_(self.weight)

        def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
            cosine = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1.0, 1.0)
            if labels is None:
                return cosine * self.s
            theta = torch.acos(cosine)
            target_cos = torch.cos(theta + self.m)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            output = cosine * (1.0 - one_hot) + target_cos * one_hot
            return output * self.s

    class ArcFaceTrainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.arc = ArcMarginProduct(
                hidden_dim,
                num_classes,
                s=float(train_cfg.get("arcface_scale", 30.0)),
                m=float(train_cfg.get("arcface_margin", 0.5)),
            )

        def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
            z = self.backbone(x)
            return self.arc(z, labels)

    model = ArcFaceTrainer().to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    crit = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    Yva_t = torch.from_numpy(Yva).to(device)

    bs = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 25))

    best_acc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(Xtr_t.size(0), device=device)
        total_loss = 0.0
        total = 0
        correct = 0

        for i in range(0, Xtr_t.size(0), bs):
            bidx = perm[i : i + bs]
            xb = Xtr_t[bidx]
            yb = Ytr_t[bidx]

            logits = model(xb, yb)
            loss = crit(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(xb.size(0))

        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, total)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_t, Yva_t)  # Pass labels to get ArcFace logits, not cosine
            val_pred = torch.argmax(val_logits, dim=1)
            val_acc = float((val_pred == Yva_t).float().mean().item())

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"[epoch {ep:02d}] loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    if best_state is None:
        print("Training failed to produce a valid checkpoint.")
        return

    ckpt = {
        "state_dict": best_state,
        "class_names": class_names,
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "arcface_scale": float(train_cfg.get("arcface_scale", 30.0)),
        "arcface_margin": float(train_cfg.get("arcface_margin", 0.5)),
        "unknown_threshold": float(train_cfg.get("unknown_threshold", 0.55)),
        "margin_threshold": float(train_cfg.get("margin_threshold", 0.10)),
    }
    torch.save(ckpt, out_model)

    meta_path = out_model.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_classes": num_classes,
                "num_samples": int(len(Yn)),
                "num_train": int(len(Ytr)),
                "num_val": int(len(Yva)),
                "best_val_acc": float(best_acc),
                "backbone_embedding": str(cfg.get("embedding", {}).get("model", "facenet")),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTraining complete")
    print(f"  model: {out_model}")
    print(f"  meta : {meta_path}")
    print(f"  best val acc: {best_acc:.3f}")
