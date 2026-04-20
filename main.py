"""Main CLI entrypoint for Face Attendance.

Subcommands:
    - status   Print resolved paths and quick checks
    - build    Build embeddings from data/students -> data/embeddings*.npz
    - process  Recognize faces in captures/ and write CSV under attendance/sessions
    - tune     Sweep thresholds on current captures (no CSV)
    - evaluate Leave-one-out evaluation on enrollment embeddings

Examples (PowerShell):
    python main.py build
    python main.py process --annotate
    python main.py evaluate
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import argparse

from src.config_utils import load_config, resolve_paths


def cmd_status(cfg, paths) -> None:
    print("Paths (resolved):")
    for k, v in paths.items():
        print(f"  {k:15}= {v}")
    # quick checks
    print("\nChecks:")
    print("  students_dir exists:", paths["students_dir"].exists())
    print("  embeddings_file exists:", paths["embeddings_file"].exists())
    print("  captures_dir exists:", paths["captures_dir"].exists())
    print("  attendance_dir exists:", paths["attendance_dir"].exists())
    rec = cfg.get("recognition", {})
    print("\nRecognition:")
    print("  method:", rec.get("method", "cosine"))
    if cfg.get("_config_path"):
        print("\nConfig:")
        print("  file:", cfg.get("_config_path"))
from src.commands.build import cmd_build
from src.commands.process import cmd_process
from src.commands.periodic import cmd_periodic
from src.commands.tune import cmd_tune
from src.commands.evaluate import cmd_evaluate
from src.commands.train import cmd_train
from src.commands.harvest import cmd_harvest
from src.commands.promote import cmd_promote


def _resolve_config_path(root: Path, config_arg: str) -> Path:
    """Resolve config path with friendly fallbacks for profile filenames.

    Supports explicit paths and bare profile names.
    """
    raw = Path(str(config_arg or "config/default.yaml"))

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(root / raw)
        candidates.append(root / "config" / raw)
        candidates.append(root / "config" / "profiles" / raw)
        if raw.suffix == "":
            candidates.append(root / f"{raw}.yaml")
            candidates.append(root / "config" / f"{raw}.yaml")
            candidates.append(root / "config" / "profiles" / f"{raw}.yaml")

    for candidate in candidates:
        p = candidate.resolve()
        if p.exists() and p.is_file():
            return p

    # Preserve previous behavior for unresolved paths; load_config will report fallback.
    return (root / raw).resolve()



def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simple Face Attendance (cosine)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="Print resolved paths and quick checks")
    p_build = sub.add_parser("build", help="Build embeddings from data/students")
    p_build.add_argument("--only", nargs="+", default=None, help="One or more person_ids to rebuild (e.g., s21c086b s21c104b)")
    p_process = sub.add_parser("process", help="Recognize faces in captures/ and write CSV")
    p_process.add_argument("--annotate", action="store_true", help="Save annotated images with boxes and labels to outputs/annotated")
    p_process.add_argument("--out", type=str, default=None, help="Custom directory for annotated outputs")
    p_process.add_argument("--camera", action="store_true", help="Capture one frame from configured camera then recognize it")
    p_periodic = sub.add_parser("periodic", help="Capture one frame periodically and confirm attendance by repeated sightings")
    p_periodic.add_argument("--annotate", action="store_true", help="Save annotated images with boxes and labels to outputs/annotated")
    p_periodic.add_argument("--out", type=str, default=None, help="Custom directory for annotated outputs")
    p_periodic.add_argument("--interval", type=float, default=5.0, help="Minutes between one-frame captures")
    p_periodic.add_argument("--cycles", type=int, default=0, help="Number of cycles to run (0 = run until Ctrl+C)")
    p_periodic.add_argument("--required-hits", type=int, default=3, help="Counted sightings required to mark present")
    p_periodic.add_argument("--min-score", type=float, default=0.75, help="Minimum recognition score to count a sighting")
    p_periodic.add_argument("--min-gap", type=float, default=2.0, help="Minimum minutes between counted sightings per student")
    p_periodic.add_argument("--from-captures", action="store_true", help="Use captures folder queue (one image per cycle) instead of camera")
    p_harvest = sub.add_parser("harvest", help="Extract face crops from captures and save for dataset expansion")
    p_harvest.add_argument("--min-score", type=float, default=0.80, help="Minimum recognition score for known labels to save")
    p_harvest.add_argument("--include-unknown", action="store_true", help="Also save crops predicted as unknown")
    p_harvest.add_argument("--to-students", action="store_true", help="Save accepted known crops directly into data/students/<label>")
    p_harvest.add_argument("--out", type=str, default=None, help="Custom output directory for harvested crops (default: data/harvest_review)")
    p_harvest.add_argument("--max-per-label", type=int, default=100, help="Maximum number of saved crops per label in one run")
    p_harvest.add_argument("--periodic-only", action="store_true", help="Use only periodic raw capture paths (.../<lesson>/captures/*.jpg)")
    p_harvest.add_argument("--outside-period-only", action="store_true", help="Use only outside_period captures for faster review")
    p_harvest.add_argument("--recent-hours", type=float, default=None, help="Only use source images modified in the last N hours")
    p_harvest.add_argument("--min-crop-size", type=int, default=32, help="Minimum crop width/height in pixels to save")
    p_harvest.add_argument("--min-focus-var", type=float, default=18.0, help="Minimum focus measure (blur filter) for saved crops; set 0 to disable")
    p_harvest.add_argument("--crop-expand", type=float, default=0.20, help="Expand each detected face box by this ratio per side before saving (e.g., 0.20)")
    p_harvest.add_argument("--review-only", action="store_true", help="Fast review mode: detect faces and save crops without recognition")
    p_harvest.add_argument("--unknown-folder", type=str, default="unrecognized", help="Folder name for unknown crops in review output")
    p_promote = sub.add_parser("promote", help="Promote reviewed harvest crops into data/students")
    p_promote.add_argument("--source", type=str, default=None, help="Source review folder (default: data/harvest_review)")
    p_promote.add_argument("--move", action="store_true", help="Move files instead of copying them")
    p_promote.add_argument("--min-face-px", type=int, default=48, help="Minimum image width/height in pixels to promote")
    p_promote.add_argument("--recent-hours", type=float, default=None, help="Only promote files modified in the last N hours")
    p_promote.add_argument("--allow-new-labels", action="store_true", help="Allow creating new student label folders if missing")
    sub.add_parser("tune", help="Sweep thresholds on current captures and report unknown rates (no CSV)")
    sub.add_parser("evaluate", help="Cross-validate on enrollment embeddings to compare cosine/knn/hybrid accuracy")
    sub.add_parser("train", help="Train ArcFace-margin classifier head with augmentation")

    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    cfg_path = _resolve_config_path(root, str(getattr(args, "config", "config/default.yaml")))
    cfg = load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    paths = resolve_paths(cfg, root)

    if args.cmd == "status":
        cmd_status(cfg, paths)
    elif args.cmd == "build":
        cmd_build(cfg, paths, only_ids=getattr(args, "only", None))
    elif args.cmd == "process":
        out_dir = Path(args.out).resolve() if getattr(args, "out", None) else None
        cmd_process(
            cfg,
            paths,
            annotate=bool(getattr(args, "annotate", False)),
            out_dir=out_dir,
            camera_once=bool(getattr(args, "camera", False)),
        )
    elif args.cmd == "periodic":
        out_dir = Path(args.out).resolve() if getattr(args, "out", None) else None
        cmd_periodic(
            cfg,
            paths,
            annotate=bool(getattr(args, "annotate", False)),
            out_dir=out_dir,
            interval_minutes=float(getattr(args, "interval", 5.0)),
            cycles=int(getattr(args, "cycles", 0)),
            required_hits=int(getattr(args, "required_hits", 3)),
            min_confirm_score=float(getattr(args, "min_score", 0.75)),
            min_gap_minutes=float(getattr(args, "min_gap", 2.0)),
            source_mode=("captures" if bool(getattr(args, "from_captures", False)) else "camera"),
        )
    elif args.cmd == "harvest":
        out_dir = Path(args.out).resolve() if getattr(args, "out", None) else None
        cmd_harvest(
            cfg,
            paths,
            min_score=float(getattr(args, "min_score", 0.80)),
            include_unknown=bool(getattr(args, "include_unknown", False)),
            to_students=bool(getattr(args, "to_students", False)),
            out_dir=out_dir,
            max_per_label=int(getattr(args, "max_per_label", 100)),
            periodic_only=bool(getattr(args, "periodic_only", False)),
            outside_period_only=bool(getattr(args, "outside_period_only", False)),
            recent_hours=getattr(args, "recent_hours", None),
            min_crop_size=int(getattr(args, "min_crop_size", 32)),
            min_focus_var=float(getattr(args, "min_focus_var", 18.0)),
            unknown_folder=str(getattr(args, "unknown_folder", "unrecognized") or "unrecognized"),
            crop_expand=float(getattr(args, "crop_expand", 0.20)),
            review_only=bool(getattr(args, "review_only", False)),
        )
    elif args.cmd == "promote":
        src = Path(args.source).resolve() if getattr(args, "source", None) else None
        cmd_promote(
            cfg,
            paths,
            source_dir=src,
            move_files=bool(getattr(args, "move", False)),
            min_face_px=int(getattr(args, "min_face_px", 48)),
            recent_hours=getattr(args, "recent_hours", None),
            allow_new_labels=bool(getattr(args, "allow_new_labels", False)),
        )
    elif args.cmd == "tune":
        cmd_tune(cfg, paths)
    elif args.cmd == "evaluate":
        cmd_evaluate(cfg, paths)
    elif args.cmd == "train":
        cmd_train(cfg, paths)
    


if __name__ == "__main__":
    main()

