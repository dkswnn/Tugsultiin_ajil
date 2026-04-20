# Classroom Attendance with FaceNet + MTCNN (Overhead Camera)

This repo records class attendance using MTCNN for face detection and FaceNet for face embeddings + a simple classifier.
Camera is mounted above the board (top-down/overhead), so data collection and processing are tuned for that view.

python.exe main.py --config hybrid_overhead_precision.yaml periodic --annotate --interval 1 --required-hits 1 --min-score 0.65

## Minimal folder structure (simple and enough)

```
.
├─ src/                         # tiny modules (detector, embedder, recognizer, io_utils)
├─ main.py                      # single entry point (build / process / status)
├─ recognize_me.py              # (optional) verify just yourself quickly (cosine)
├─ config/
│  └─ default.yaml              # optional config (paths + thresholds); defaults used if missing
├─ data/
│  └─ students/                 # enrollment photos: one folder per person
│     ├─ s001_Ariun/
│     └─ s002_Bat/
├─ models/                      # (not used without SVM)
├─ attendance/                  # session CSV(s) saved here
├─ outputs/                     # optional: annotated frames/images
└─ requirements.txt
```

That’s it. Start with `data/students/<person_id>_<name>/` and 10–20 images per person from the actual camera angle.

Quick start (cosine, one command entry):

1. Enrollment — organize photos
   - Put 10–20 images per person in `data/students/<person_id>_<name>/`.

2. Build gallery embeddings (mean per person)
   - `python main.py build`

3. Process captured images (from `captures/`) and write a session CSV
   - `python main.py process`

Optional:

- Check paths and basic status: `python main.py status`
- Config is optional: if `config/default.yaml` is missing or PyYAML isn’t installed, built-in defaults are used.

## CLI commands (full)

General usage:

```powershell
python .\main.py [--config config/default.yaml] <command> [options]
```

Important: `--config` is a global argument and must be placed before the command.

Correct:

```powershell
python .\main.py --config config/profiles/hybrid_overhead_extreme.yaml periodic --interval 1
```

### `status`

Show resolved paths, existence checks, and active recognition method.

```powershell
python .\main.py status
```

### `build`

Build gallery embeddings from `data/students`.

```powershell
python .\main.py build
python .\main.py build --only s21c086b s21c104b
```

### `process`

Run recognition on images in `captures` (or one camera burst with `--camera`) and save session CSV.

```powershell
python .\main.py process
python .\main.py process --annotate
python .\main.py process --annotate --out outputs/annotated_custom
python .\main.py process --camera --annotate
```

### `periodic`

Periodic attendance loop (camera or captures queue mode).

```powershell
python .\main.py periodic
python .\main.py periodic --interval 1 --required-hits 5 --min-score 0.68 --cycles 10
python .\main.py periodic --from-captures --cycles 20 --annotate
```

Options:

- `--interval`: minutes between cycles
- `--cycles`: number of cycles (`0` = until Ctrl+C)
- `--required-hits`: counted sightings needed to mark present
- `--min-score`: minimum recognition score to count a sighting
- `--min-gap`: minimum minutes between counted sightings per student
- `--from-captures`: use capture queue instead of live camera
- `--annotate`, `--out`: save annotated images

### `harvest`

Extract face crops from captures for dataset expansion.

```powershell
python .\main.py harvest --min-score 0.80 --max-per-label 50
python .\main.py harvest --periodic-only --recent-hours 24 --min-score 0.80 --min-crop-size 56 --min-focus-var 22 --crop-expand 0.20 --unknown-folder unrecognized
python .\main.py --config config/profiles/review_fast_mtcnn.yaml harvest --outside-period-only --review-only --min-crop-size 56 --min-focus-var 20 --crop-expand 0.20 --unknown-folder unrecognized --out data/harvest_review_fast_outside_period
python .\main.py harvest --to-students --min-score 0.85 --max-per-label 30
python .\main.py --config config/profiles/review_fast_mtcnn.yaml harvest --periodic-only --include-unknown --min-score 0.70 --min-crop-size 56 --min-focus-var 20 --crop-expand 0.20 --unknown-folder unrecognized --max-per-label 120 --out data/harvest_review_fast_mtcnn
python .\main.py --config config/profiles/review_fast_mtcnn.yaml harvest --periodic-only --review-only --min-crop-size 56 --min-focus-var 20 --crop-expand 0.20 --unknown-folder unrecognized --out data/harvest_review_fast_review_only
```

Options:

- `--min-score`: minimum score for known labels
- `--include-unknown`: include unknown predictions too
- `--review-only`: fast review mode; detect faces and save crops without recognition
- `--to-students`: write known accepted crops directly to `data/students/<label>`
- `--out`: custom review folder (default `data/harvest_review`)
- `--max-per-label`: cap saved crops per label in one run
- `--periodic-only`: only use periodic raw capture paths
- `--outside-period-only`: only use outside_period capture paths
- `--recent-hours`: only use recent source images
- `--min-crop-size`: reject tiny crops by pixel size
- `--min-focus-var`: reject blurry crops (set `0` to disable)
- `--crop-expand`: expand each detected box by ratio per side before saving (e.g., `0.20`)
- `--unknown-folder`: folder name for unknown crops in review output

### `promote`

Promote reviewed crops from harvest into `data/students` safely.

```powershell
python .\main.py promote --min-face-px 72 --recent-hours 24
python .\main.py promote --source data/harvest_review --move --min-face-px 72
```

Options:

- `--source`: source review folder (default `data/harvest_review`)
- `--move`: move files instead of copy
- `--min-face-px`: minimum crop width/height to promote
- `--recent-hours`: only recent reviewed files
- `--allow-new-labels`: allow creating new student folders if missing

### `tune`

Sweep thresholds on current captures (no attendance CSV write).

```powershell
python .\main.py tune
```

### `evaluate`

Cross-validate on enrollment embeddings.

```powershell
python .\main.py evaluate
```

### `train`

Train ArcFace-margin classifier head.

```powershell
python .\main.py train
```

Optional: Switch between cosine, KNN, and Hybrid (no training step)

- Cosine (fast, simple):
  ```yaml
  recognition:
    method: cosine
    decision_threshold: 0.9
    margin_threshold: 0.1
    min_detect_prob: 0.9
    min_box_size: 40
    cosine_gate: 0.9
  ```
- KNN (uses all per-image embeddings; run `build` first):
  ```yaml
  recognition:
    method: knn
    knn_k: 5
    decision_threshold: 0.85
    margin_threshold: 0.2
    min_detect_prob: 0.9
    min_box_size: 40
    cosine_gate: 0.9
  ```
- Hybrid (KNN + cosine-to-mean blend):
  ```yaml
  recognition:
    method: hybrid
    knn_k: 3
    hybrid_alpha: 0.5 # 0..1; 1=KNN only, 0=cosine-to-mean only
    decision_threshold: 0.7
    margin_threshold: 0.1
    min_detect_prob: 0.85
    min_box_size: 24
    cosine_gate: null # optional; enable (e.g., 0.9) to be stricter
  ```

### Recognition thresholds and filtering

- Cosine:
  - `decision_threshold`: minimum cosine similarity to accept a match; else `unknown`.
  - `margin_threshold`: minimum gap between top1 and top2 cosine scores.

- KNN:
  - `decision_threshold`: minimum cosine of the best single neighbor.
  - `margin_threshold`: minimum (top_class_sum - second_class_sum) from top-k votes.

- Hybrid:
  - Blends KNN class-sum (normalized by k) with cosine-to-mean: combined = alpha*(knn_sum/k) + (1-alpha)*cosine_mean.
  - `decision_threshold` applies to the best single neighbor (like KNN).
  - `margin_threshold` applies to the combined top1-top2 difference.

- Detection filtering (both modes):
  - `min_detect_prob` drops faces from the detector below a confidence (default 0.9). Lower it if you miss faces; raise it to avoid spurious detections.
  - `min_box_size` removes tiny detections by width/height (default 40 in config). Increase to ignore small/background faces.
  - `cosine_gate` (extra open-set guard): require a minimum cosine similarity to the predicted class mean (e.g., 0.9). Helps reject look-alikes.

Optional (single-person sanity check): - `python recognize_me.py --build` - `python recognize_me.py --image d:/Tugsultiin_Ajil/captures/sample.jpg`

If later you want more control (crops, embeddings cache, logs, etc.), you can grow into the fuller layout below.

## Full layout (optional, more granular)

```
.
├─ face_detection/                # your existing quick test code (can keep for experiments)
├─ config/
│  └─ default.yaml                # runtime settings (thresholds, paths, camera)
├─ data/
│  ├─ raw/                        # original images/videos from camera
│  │  └─ 2025-11-03/              # optional date-based subfolders for ingestion
│  ├─ faces/                      # cropped/aligned faces for each person
│  │  └─ <person_id>_<name>/
│  └─ embeddings/                 # persistent embeddings store
│     ├─ embeddings.parquet       # or .npz/.pkl; choose one
│     └─ meta.json                # versioning, model, classifier info
├─ models/
│  ├─ facenet/                    # FaceNet weights/checkpoints if needed
│  └─ classifiers/                # (unused without SVM)
├─ scripts/
│  ├─ enroll_from_images.py       # build dataset per person from images
│  ├─ run_attendance.py           # main runtime loop: detect → embed → classify → log
│  └─ evaluate.py                 # optional: accuracy/eer/roc by session
├─ attendance/
│  ├─ class_roster.csv            # person_id,name,email,role
│  ├─ sessions/                   # per-session outputs
│  │  └─ 2025-11-03_period1.csv
│  └─ rollup.csv                  # cumulative attendance ledger
├─ calibration/
│  ├─ camera.yaml                 # intrinsics/extrinsics if known; or homography
│  └─ roi_mask.png                # mask for board/irrelevant area (top-down)
├─ outputs/
│  ├─ visualizations/             # annotated frames/images for debugging
│  └─ debug_frames/               # saved frames around detections
├─ logs/
│  └─ app.log
├─ requirements.txt
└─ README.md
```

## Rationale (overhead camera)

- Enrollment needs more variation in pitch/roll due to top-down view. Aim for 15–30 images/person (from above angle if possible).
- Consider masking the board/ceiling area with `calibration/roi_mask.png` to reduce false detections.
- If the camera is fixed, you can crop to a static region where student faces usually appear.
- Classifier choice: cosine or KNN; start with cosine, move to KNN if you have many enrollment images per person.
- Persist embeddings to `data/embeddings` so you can retrain quickly without re-encoding.

## Files you can add later (optional)

- `scripts/run_attendance.py`: open camera → MTCNN detect → FaceNet embed → classify → write CSV to `attendance/sessions/DATE.csv`.
- `scripts/enroll_from_images.py`: read `data/raw/<person>` folders → detect/aligned faces → save to `data/faces/<person>` → compute embeddings → append to `data/embeddings/`.

## Config keys (see config/default.yaml)

- detector thresholds (min_face_size, thresholds, device)
- embedder model (FaceNet variant), image size, margin, alignment
- classifier path and type
- camera index/URL, FPS, ROI mask
- output paths for logs, visualizations, attendance

Tip: Don’t move your existing `face_detection/` yet—use scripts/ for production flow and keep `face_detection/` for experiments.
