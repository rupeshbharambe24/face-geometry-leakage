# Face Geometry Leakage

**Do facial landmark geometry features identify people?**

Many "privacy-friendly" face apps send only geometry instead of images: triangle angles, landmark distances, or MediaPipe's 52 blendshape scores. This project measures how much **identity** those features leak, and tests defences that keep expression information while removing identity.

It began as the feature pipeline of a face liveness system: MediaPipe FaceMesh with 478 landmarks, 141 triplet vertex angles and a small MLP. The question here is whether that same 141-number vector works as a fingerprint.

## Research questions

| # | Question | Protocol |
|---|---|---|
| RQ1 | Can an attacker re-identify enrolled people from geometry alone? | `identify_match` (cross-session, neutral gallery), `identify_loeo` |
| RQ2 | Can an attacker trained only on public data link *unseen* people? | `verify_open_set` (CelebA → KDEF / CK+ / JAFFE) |
| RQ3 | How does head pose change leakage? | `identify_match` frontal → half-profile (KDEF) |
| RQ4 | How much expression information do the same features carry? | `utility` (subject-independent folds) |
| RQ5 | Which defences cut identity leakage at the least utility cost? | `defense` (delta-neutral, noise, quantisation, feature selection, adversarial) |

## Feature representations

| Name | What it is | Dims |
|---|---|---|
| `legacy_angles` | Exact replica of the deployed liveness features: width/height normalisation and arccos | 141 |
| `angles2d` / `angles3d` | Unique legacy triplets, numerically stable `atan2` angles in pixel space | 139 |
| `delaunay_angles2d` | All vertex angles of a Delaunay triangulation of the mean face | ~2.7k |
| `distances2d` | Pairwise distances among the triplet landmarks, divided by inter-ocular distance | ~4k |
| `coords2d` | Similarity-aligned 2D coordinates | 956 |
| `coords3d` | 3D Procrustes-aligned coordinates (head rotation removed) | 1434 |
| `blendshapes` | MediaPipe's 52 expression scores | 52 |

## Layout

```
configs/        smoke.yaml (synthetic check), main.yaml (full study)
data/           raw/ (your datasets, git-ignored), manifests/, landmarks/ (generated)
docs/           research_plan.md
src/fgl/        datasets, extract (MediaPipe), geometry, features, protocols,
                defenses, models (PyTorch), metrics, runner, report, cli
tests/          geometry invariances, metrics, end-to-end synthetic run
```

## Setup on Windows (RTX 3050 Ti)

Run these in PowerShell. Replace `E:` with whatever drive letter Windows gives the SSD.

```powershell
# 1. Install uv once (Python manager). Restart PowerShell afterwards.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Go to the project and install everything (Python 3.12, CUDA PyTorch, MediaPipe)
E:
cd E:\Projects\face-geometry-leakage
uv sync --extra dev

# 3. Check the GPU and run the tests
uv run fgl check-gpu        # expect: CUDA available: True, NVIDIA GeForce RTX 3050 Ti
uv run pytest -q
```

If `check-gpu` prints `False`, update the NVIDIA driver (version 560 or newer supports CUDA 12.6) and run `uv sync` again.

## Smoke test (synthetic data, about 1–2 minutes)

```powershell
uv run fgl synth --out data/landmarks/synthetic.npz --subjects 30 --identity-seed 0 --prefix S
uv run fgl synth --out data/landmarks/synthetic_public.npz --subjects 60 --identity-seed 1 --prefix P
uv run fgl run configs/smoke.yaml
uv run fgl report results/smoke
```

`results/smoke/report.md` and `results/smoke/figures/` should appear. The numbers are meaningless; this only checks the pipeline.

## Real data

See [data/README.md](data/README.md) for where to request each dataset and where to unpack it.

### 1. Download the MediaPipe model

```powershell
uv run fgl download-model
```

### 2. Index each dataset, then extract landmarks

Extraction runs on the CPU (MediaPipe's Python package has no Windows GPU support). Set `--workers` to your CPU core count minus 2.

```powershell
uv run fgl manifest kdef   --root data/raw/KDEF
uv run fgl extract --manifest data/manifests/kdef.csv   --root data/raw/KDEF   --out data/landmarks/kdef.npz   --workers 6

uv run fgl manifest jaffe  --root data/raw/jaffe
uv run fgl extract --manifest data/manifests/jaffe.csv  --root data/raw/jaffe  --out data/landmarks/jaffe.npz  --workers 6

uv run fgl manifest ckplus --root data/raw/CK+
uv run fgl extract --manifest data/manifests/ckplus.csv --root data/raw/CK+    --out data/landmarks/ckplus.npz --workers 6

# CelebA: the attacker's public data. 2000 people is about 50k images.
uv run fgl manifest celeba --root data/raw/CelebA --min-images 20 --max-identities 2000
uv run fgl extract --manifest data/manifests/celeba.csv --root data/raw/CelebA --out data/landmarks/celeba.npz --workers 6

uv run fgl info data/landmarks/kdef.npz      # faces found, people, expressions, poses
```

### 3. Run the study

```powershell
# KDEF experiments first; they don't need the other datasets
uv run fgl run configs/main.yaml --only kdef_cross_session kdef_neutral_gallery kdef_loeo kdef_pose_shift kdef_utility kdef_defense
uv run fgl report results/main

# then everything
uv run fgl run configs/main.yaml
uv run fgl report results/main
```

Each run rewrites `results/<name>/results.csv`. Copy the folder first if you want to keep an old run.

## Notes

- **Licensed data:** datasets and extracted landmarks are git-ignored. Check each dataset's licence before sharing derived features.
- **Pixel units:** landmarks are stored as x·W, y·H, z·W. The image aspect ratio therefore doesn't distort angles; MediaPipe's normalised x and y are scaled by width and height separately.
- **Test people are always unseen:** utility uses folds split by person, and defences are fitted on different people from the ones they are evaluated on.
