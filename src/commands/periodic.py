from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import csv
import time
import shutil

from src.commands.process import cmd_process
from src.io_utils import list_captures


@dataclass
class _ConfirmState:
    hits: int = 0
    best_score: float = 0.0
    first_seen: str = ""
    last_seen: str = ""
    last_counted_ts: datetime | None = None


def _latest_session_file(session_dir: Path) -> Path | None:
    files = sorted(session_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _read_session_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_confirm_csv(out_path: Path, states: Dict[str, _ConfirmState], required_hits: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "hits", "present", "best_score", "first_seen", "last_seen"])
        for label in sorted(states.keys()):
            st = states[label]
            w.writerow([
                label,
                st.hits,
                "yes" if st.hits >= required_hits else "no",
                f"{st.best_score:.4f}",
                st.first_seen,
                st.last_seen,
            ])


def _parse_hhmm(v: str) -> tuple[int, int] | None:
    try:
        hh, mm = str(v).split(":", 1)
        h = int(hh)
        m = int(mm)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h, m
    except Exception:
        return None
    return None


def _period_name_from_now(now: datetime, cfg: Dict) -> str:
    cap_cfg = cfg.get("capture", {})
    periods = cap_cfg.get("periods", [])
    cur = now.hour * 60 + now.minute
    for p in periods:
        try:
            name = str(p.get("name", "")).strip()
            s = _parse_hhmm(str(p.get("start", "")))
            e = _parse_hhmm(str(p.get("end", "")))
            if not name or s is None or e is None:
                continue
            s_min = s[0] * 60 + s[1]
            e_min = e[0] * 60 + e[1]
            if s_min <= cur <= e_min:
                return name
        except Exception:
            continue
    return "outside_period"


def _capture_period_dir(base: Path, now: datetime, cfg: Dict) -> Path:
    period_name = _period_name_from_now(now, cfg)
    return base / str(now.year) / str(now.month) / str(now.day) / period_name


def cmd_periodic(
    cfg: Dict,
    paths: Dict[str, Path],
    annotate: bool = False,
    out_dir: Path | None = None,
    interval_minutes: float = 5.0,
    cycles: int = 0,
    required_hits: int = 3,
    min_confirm_score: float = 0.75,
    min_gap_minutes: float = 2.0,
    source_mode: str = "camera",
) -> None:
    """Capture one image at a time periodically and confirm attendance by repeated sightings.

    - Each cycle captures exactly one frame from camera and processes only that image.
    - A student is marked present only after `required_hits` counted sightings.
    - Counted sightings are debounced by `min_gap_minutes` per label.
    """

    interval_minutes = float(interval_minutes)
    required_hits = int(required_hits)
    min_confirm_score = float(min_confirm_score)
    min_gap_minutes = float(min_gap_minutes)

    states_by_lesson: Dict[str, Dict[str, _ConfirmState]] = {}
    started_at = datetime.now()
    min_gap = timedelta(minutes=max(0.0, min_gap_minutes))
    sleep_seconds = max(1.0, interval_minutes * 60.0)

    source_mode = str(source_mode or "camera").lower().strip()
    if source_mode not in ("camera", "captures"):
        source_mode = "camera"

    print("Periodic capture started")
    print(f"  source_mode      : {source_mode}")
    print(f"  interval_minutes : {interval_minutes}")
    print(f"  required_hits    : {required_hits}")
    print(f"  min_confirm_score: {min_confirm_score}")
    print(f"  min_gap_minutes  : {min_gap_minutes}")
    print("  output layout    : captures/YYYY/M/D/<lesson>/{captures,result_csv,finalized_attendance.csv}")
    print("Press Ctrl+C to stop (if cycles=0).")

    session_dir = paths["attendance_dir"] / "sessions"
    count = 0
    empty_cycles = 0
    capture_queue = list_captures(paths["captures_dir"]) if source_mode == "captures" else []
    queue_idx = 0
    queue_tmp_dir = paths["captures_dir"] / ".periodic_queue"
    if source_mode == "captures":
        queue_tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"  captures queued  : {len(capture_queue)}")
        if len(capture_queue) == 0:
            print("No images found in captures directory for --from-captures mode.")
            return

    try:
        while True:
            if cycles > 0 and count >= cycles:
                break

            cycle_now = datetime.now()
            lesson_root = _capture_period_dir(paths["captures_dir"], cycle_now, cfg)
            lesson_key = str(lesson_root)
            lesson_states = states_by_lesson.setdefault(lesson_key, {})
            lesson_capture_dir = lesson_root / "captures"
            lesson_result_dir = lesson_root / "result_csv"
            lesson_confirm_path = lesson_root / "finalized_attendance.csv"
            cycle_out_dir = out_dir if out_dir is not None else (lesson_root / "annotated")

            before = _latest_session_file(session_dir)

            if source_mode == "camera":
                cmd_process(
                    cfg,
                    paths,
                    annotate=annotate,
                    out_dir=cycle_out_dir,
                    camera_once=True,
                    camera_capture_dir=lesson_capture_dir,
                )
            else:
                if queue_idx >= len(capture_queue):
                    print("Capture queue exhausted.")
                    break
                src_img = capture_queue[queue_idx]
                queue_idx += 1
                lesson_capture_dir.mkdir(parents=True, exist_ok=True)
                archived_img = lesson_capture_dir / src_img.name
                shutil.copy2(src_img, archived_img)
                for f in queue_tmp_dir.glob("*"):
                    if f.is_file():
                        try:
                            f.unlink()
                        except Exception:
                            pass
                dst_img = queue_tmp_dir / src_img.name
                shutil.copy2(archived_img, dst_img)
                paths_local = dict(paths)
                paths_local["captures_dir"] = queue_tmp_dir
                cmd_process(
                    cfg,
                    paths_local,
                    annotate=annotate,
                    out_dir=cycle_out_dir,
                    camera_once=False,
                )

            after = _latest_session_file(session_dir)
            if after is None or (before is not None and after == before):
                print("[warn] No new session output produced on this cycle.")
                empty_cycles += 1
                if source_mode == "camera" and empty_cycles >= 3:
                    print("[error] Stopping periodic run after 3 empty camera cycles. Check --config path and camera source.")
                    break
            else:
                empty_cycles = 0
                lesson_result_dir.mkdir(parents=True, exist_ok=True)
                moved_csv = lesson_result_dir / after.name
                if moved_csv.exists():
                    moved_csv = lesson_result_dir / f"{after.stem}_{int(time.time())}{after.suffix}"
                try:
                    shutil.move(str(after), str(moved_csv))
                except Exception:
                    moved_csv = after

                rows = _read_session_rows(moved_csv)
                now = datetime.now()

                for row in rows:
                    lab = str(row.get("label", ""))
                    if lab in ("", "unknown", "no_face"):
                        continue
                    try:
                        score = float(row.get("score", "0") or 0.0)
                    except Exception:
                        score = 0.0
                    if score < min_confirm_score:
                        continue

                    st = lesson_states.get(lab)
                    if st is None:
                        st = _ConfirmState()
                        lesson_states[lab] = st

                    if st.last_counted_ts is not None and (now - st.last_counted_ts) < min_gap:
                        st.last_seen = now.isoformat(timespec="seconds")
                        st.best_score = max(st.best_score, score)
                        continue

                    st.hits += 1
                    st.best_score = max(st.best_score, score)
                    ts = now.isoformat(timespec="seconds")
                    if not st.first_seen:
                        st.first_seen = ts
                    st.last_seen = ts
                    st.last_counted_ts = now

                _write_confirm_csv(lesson_confirm_path, lesson_states, required_hits)

                present = sorted([lab for lab, st in lesson_states.items() if st.hits >= required_hits])
                print(
                    f"[cycle {count + 1}] lesson={lesson_root.name} session={moved_csv.name} "
                    f"known_counted={sum(1 for s in lesson_states.values() if s.hits > 0)} present={len(present)}"
                )
                if present:
                    print("  present:", ", ".join(present))

            count += 1
            if cycles > 0 and count >= cycles:
                break
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    total_present = 0
    for lesson_key, lesson_states in states_by_lesson.items():
        lesson_confirm_path = Path(lesson_key) / "finalized_attendance.csv"
        _write_confirm_csv(lesson_confirm_path, lesson_states, required_hits)
        total_present += sum(1 for st in lesson_states.values() if st.hits >= required_hits)

    print("\nPeriodic capture finished")
    print(f"  cycles_run: {count}")
    print(f"  present(total across lessons): {total_present}")
