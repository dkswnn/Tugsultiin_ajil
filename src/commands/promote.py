from __future__ import annotations

from pathlib import Path
from typing import Dict


def cmd_promote(
    cfg: Dict,
    paths: Dict[str, Path],
    source_dir: Path | None = None,
    move_files: bool = False,
    min_face_px: int = 48,
    recent_hours: float | None = None,
    allow_new_labels: bool = False,
) -> None:
    from datetime import datetime, timedelta
    from PIL import Image
    import hashlib
    import shutil

    students_dir = paths["students_dir"]
    src_root = source_dir if source_dir is not None else (students_dir.parent / "harvest_review")

    if not src_root.exists():
        print(f"Source review folder not found: {src_root}")
        return

    cutoff_dt = None
    if recent_hours is not None and float(recent_hours) > 0:
        cutoff_dt = datetime.now() - timedelta(hours=float(recent_hours))

    blocked_labels = {"unknown", "unrecognized", "no_face"}
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

    moved = 0
    skipped = 0
    skipped_duplicate = 0

    def _file_hash(path: Path) -> str | None:
        try:
            h = hashlib.sha1()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    print("Promoting reviewed crops to students dataset")
    print(f"  source       : {src_root}")
    print(f"  destination  : {students_dir}")
    print(f"  action       : {'move' if move_files else 'copy'}")
    print(f"  min_face_px  : {min_face_px}")
    if cutoff_dt is not None:
        print(f"  recent filter: last {recent_hours}h")

    for label_dir in sorted([d for d in src_root.iterdir() if d.is_dir()]):
        label = label_dir.name
        if label.lower() in blocked_labels:
            print(f"  [skip-label] {label}")
            continue

        dst_dir = students_dir / label
        if not dst_dir.exists() and not allow_new_labels:
            print(f"  [skip-label] {label} (no existing student folder)")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Build an exact-duplicate set from existing target files for this label.
        existing_hashes: set[str] = set()
        for ex in sorted([p for p in dst_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext]):
            h = _file_hash(ex)
            if h is not None:
                existing_hashes.add(h)

        for img_path in sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext]):
            try:
                if cutoff_dt is not None and datetime.fromtimestamp(img_path.stat().st_mtime) < cutoff_dt:
                    skipped += 1
                    continue
                with Image.open(img_path) as im:
                    w, h = im.size
                if min(w, h) < int(min_face_px):
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            src_hash = _file_hash(img_path)
            if src_hash is not None and src_hash in existing_hashes:
                skipped += 1
                skipped_duplicate += 1
                continue

            dst_path = dst_dir / img_path.name
            if dst_path.exists():
                stem = img_path.stem
                suf = img_path.suffix
                idx = 1
                while True:
                    cand = dst_dir / f"{stem}_{idx}{suf}"
                    if not cand.exists():
                        dst_path = cand
                        break
                    idx += 1

            try:
                if move_files:
                    shutil.move(str(img_path), str(dst_path))
                else:
                    shutil.copy2(str(img_path), str(dst_path))
                if src_hash is not None:
                    existing_hashes.add(src_hash)
                moved += 1
            except Exception:
                skipped += 1

    print("\nPromote complete")
    print(f"  promoted: {moved}")
    print(f"  skipped : {skipped}")
    print(f"  duplicates_skipped: {skipped_duplicate}")
