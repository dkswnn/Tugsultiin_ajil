from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def cmd_build(cfg: Dict, paths: Dict[str, Path], only_ids: Optional[List[str]] = None) -> None:
    from PIL import Image
    import numpy as np
    import os
    from src.detector import Detector
    from src.embedding import Embedder
    from src.io_utils import save_embeddings_npz, load_embeddings_npz
    from src.io_utils import list_person_dirs as _list_person_dirs
    from src.io_utils import iter_images as _iter_images

    students_dir = paths["students_dir"]
    if not students_dir.exists():
        print(f"Students dir not found: {students_dir}")
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

    emb_cfg = cfg.get("embedding", {})
    emb = Embedder(
        model_name=str(emb_cfg.get("model", "facenet")),
        fusion_alpha=float(emb_cfg.get("fusion_alpha", 0.35) or 0.35),
    )

    enroll_aug_enabled = bool(emb_cfg.get("enroll_aug_enabled", False))
    enroll_aug_count = int(emb_cfg.get("enroll_aug_count", 2) or 2)
    enroll_aug_brightness_delta = float(emb_cfg.get("enroll_aug_brightness_delta", 0.12) or 0.12)
    enroll_aug_contrast_delta = float(emb_cfg.get("enroll_aug_contrast_delta", 0.12) or 0.12)

    if enroll_aug_enabled:
        out_npz = paths.get("embeddings_aug_file", paths["embeddings_file"])
        out_all_npz = paths.get("embeddings_all_aug_file", paths["embeddings_all_file"])
    else:
        out_npz = paths["embeddings_file"]
        out_all_npz = paths["embeddings_all_file"]
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    cache_path = paths["embeddings_file"].with_name("embeddings_cache.npz")

    def _load_cache(path: Path):
        try:
            if not path.exists():
                return {}
            data = np.load(path, allow_pickle=True)
            p = data.get("paths")
            t = data.get("mtimes")
            e = data.get("embeddings")
            if p is None or t is None or e is None:
                return {}
            cache = {}
            for fp, mt, vec in zip(p, t, e):
                cache[str(fp)] = (float(mt), vec.astype("float32"))
            return cache
        except Exception:
            return {}

    def _save_cache(path: Path, cache: dict):
        try:
            if not cache:
                return
            keys = list(cache.keys())
            mtimes = [cache[k][0] for k in keys]
            embs = [cache[k][1] for k in keys]
            np.savez_compressed(
                path,
                paths=np.array(keys, dtype=object),
                mtimes=np.array(mtimes, dtype=float),
                embeddings=np.stack(embs, axis=0).astype("float32"),
            )
        except Exception:
            pass

    cache = _load_cache(cache_path)
    cache_hits_total = 0
    cache_new_total = 0

    labels: List[str] = []
    means: List = []
    all_embs: List = []
    all_labels: List[str] = []

    enroll_cfg = cfg.get("enrollment", {})
    min_cos_center = float(enroll_cfg.get("min_cos_to_center", 0.0) or 0.0)
    max_per_person = enroll_cfg.get("max_per_person", None)
    max_per_person = int(max_per_person) if max_per_person else None

    def _pid_from_name(name: str) -> str:
        return name.split("_", 1)[0]

    selected: Optional[set] = set([s.strip() for s in only_ids]) if only_ids else None

    if selected:
        Eold, Lold = load_embeddings_npz(out_npz)
        if Eold is not None and Lold is not None and len(Eold):
            for vec, lab in zip(Eold, Lold):
                if _pid_from_name(str(lab)) not in selected:
                    labels.append(str(lab))
                    means.append(np.asarray(vec, dtype="float32"))
        Eall_old, Lall_old = load_embeddings_npz(out_all_npz)
        if Eall_old is not None and Lall_old is not None and len(Eall_old):
            for vec, lab in zip(Eall_old, Lall_old):
                if _pid_from_name(str(lab)) not in selected:
                    all_embs.append(np.asarray(vec, dtype="float32")[None, :])
                    all_labels.append(str(lab))

    rec_cfg = cfg.get("recognition", {})
    multi = bool(rec_cfg.get("multi_centroids", False))
    max_c = int(rec_cfg.get("max_centroids", 2) or 2)

    def _augment_face_variants(face_tensor):
        variants = [face_tensor]
        if not enroll_aug_enabled or enroll_aug_count <= 0:
            return variants
        try:
            import torch  # type: ignore
        except Exception:
            return variants

        bvals = [1.0 - enroll_aug_brightness_delta, 1.0 + enroll_aug_brightness_delta]
        cvals = [1.0 - enroll_aug_contrast_delta, 1.0 + enroll_aug_contrast_delta]
        max_variants = max(1, enroll_aug_count)

        for bi, ci in zip(bvals, cvals):
            t = face_tensor.clone()
            if float(t.min()) >= 0.0 and float(t.max()) <= 1.0:
                t = torch.clamp(t * float(bi), 0.0, 1.0)
                ch_mean = t.mean(dim=(1, 2), keepdim=True)
                t = torch.clamp((t - ch_mean) * float(ci) + ch_mean, 0.0, 1.0)
            variants.append(t)
            if len(variants) >= (1 + max_variants):
                break
        return variants

    for person_dir in _list_person_dirs(students_dir):
        label = person_dir.name
        pid = _pid_from_name(label)
        if selected and pid not in selected:
            continue

        person_images = list(_iter_images(person_dir))
        total_person_images = len(person_images)
        print(f"[start] {label}: {total_person_images} image(s)")

        person_embs = []
        used = 0

        for idx_img, img_path in enumerate(person_images, start=1):
            if idx_img == 1 or idx_img % 10 == 0 or idx_img == total_person_images:
                print(f"  [{label}] {idx_img}/{total_person_images}: {img_path.name}")
            try:
                img = Image.open(img_path)
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                img = img.convert("RGB")
            except Exception as e:
                print(f"  [skip-img] {label}: {img_path.name} ({e})")
                continue

            fkey = str(Path(img_path).resolve())
            fmtime = float(os.path.getmtime(img_path))

            hit = False
            if fkey in cache:
                mt_cached, vec_cached = cache[fkey]
                if abs(mt_cached - fmtime) < 1e-6:
                    person_embs.append(vec_cached.astype("float32")[None, :])
                    cache_hits_total += 1
                    used += 1
                    hit = True
            if hit:
                continue

            try:
                faces, probs = det.detect(img)
            except Exception as e:
                print(f"  [detect-error] {label}: {img_path.name} ({e})")
                continue

            if not faces:
                continue

            import numpy as _np
            idx = int(_np.argmax(_np.asarray(probs)))
            face_tensor = faces[idx]
            face_variants = _augment_face_variants(face_tensor)

            Ef = emb.embed_tensors(face_variants)
            if Ef.size > 0:
                vec = Ef.mean(axis=0)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                cache[fkey] = (fmtime, vec)
                cache_new_total += 1
                person_embs.append(vec[None, :])
                used += 1

        if not person_embs:
            print(f"  [skip-person] {label}: no faces found")
            continue

        E = np.concatenate(person_embs, axis=0)
        if min_cos_center > 0 and len(E) > 2:
            center = np.mean(E, axis=0, keepdims=True)
            center = center / (np.linalg.norm(center) + 1e-8)
            sims = np.dot(E, center.T).flatten()
            keep_indices = np.where(sims >= min_cos_center)[0]
            if len(keep_indices) < len(E):
                print(f"  [{label}] Filtered {len(E) - len(keep_indices)} outliers (cos < {min_cos_center})")
                E = E[keep_indices]
                person_embs = [E[i : i + 1] for i in range(E.shape[0])]

        if max_per_person and len(E) > max_per_person:
            center = np.mean(E, axis=0, keepdims=True)
            center = center / (np.linalg.norm(center) + 1e-8)
            sims = np.dot(E, center.T).flatten()
            top_indices = np.argsort(sims)[::-1][:max_per_person]
            E = E[top_indices]
            person_embs = [E[i : i + 1] for i in range(E.shape[0])]
            print(f"  [{label}] Limited to {len(E)} closest images to center")

        if E.shape[0] == 0:
            print(f"  [skip-person] {label}: no faces left after filtering")
            continue

        all_embs.extend(person_embs)
        all_labels.extend([label] * len(person_embs))

        if multi:
            from sklearn.cluster import KMeans
            n_clusters = min(len(person_embs), max_c)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(E)
                for i in range(n_clusters):
                    cluster_mean = E[kmeans.labels_ == i].mean(axis=0)
                    cluster_mean = cluster_mean / (np.linalg.norm(cluster_mean) + 1e-8)
                    labels.append(f"{label}#{i+1}")
                    means.append(cluster_mean.astype("float32"))
            else:
                mean_emb = np.mean(E, axis=0)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
                labels.append(label)
                means.append(mean_emb.astype("float32"))
        else:
            mean_emb = np.mean(E, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
            labels.append(label)
            means.append(mean_emb.astype("float32"))

        print(f"  [done] {label}: built mean from {used} images")

    if not means:
        print("No embeddings built.")
        return

    print(f"\nBuilt {len(labels)} mean embeddings.")
    save_embeddings_npz(out_npz, np.array(means), labels)
    print(f"Saved mean embeddings to: {out_npz}")

    if all_embs:
        print(f"Built {len(all_labels)} total per-image embeddings.")
        save_embeddings_npz(out_all_npz, np.concatenate(all_embs, axis=0), all_labels)
        print(f"Saved all embeddings to: {out_all_npz}")

    _save_cache(cache_path, cache)
    print(f"\nCache hits: {cache_hits_total}, New: {cache_new_total}")
    print("Build complete.")
