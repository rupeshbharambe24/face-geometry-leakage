"""Run MediaPipe FaceLandmarker over a manifest and cache landmarks, blendshapes, pose.

Output .npz keys:
  landmarks  (N, 478, 3) float32, pixel units: x*W, y*H, z*W
  blendshapes (N, 52) float32
  matrix     (N, 4, 4) float32 facial transformation matrix
  image_size (N, 2) int32 (width, height)
  ok         (N,) bool, False when no face was found
  blendshape_names (52,) str
plus the manifest itself, saved next to it as <name>.manifest.csv.
"""

from __future__ import annotations

import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
NUM_LANDMARKS, NUM_BLENDSHAPES = 478, 52

_detector = None


def download_model(dest: str | Path = "models/face_landmarker.task") -> Path:
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL} -> {dest}")
    urllib.request.urlretrieve(MODEL_URL, dest)
    return dest


def _init_worker(model_path: str) -> None:
    global _detector
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    _detector = vision.FaceLandmarker.create_from_options(options)


def _read_rgb(path: str) -> np.ndarray | None:
    import cv2

    # np.fromfile + imdecode handles non-ASCII Windows paths; IMREAD_COLOR makes grey images RGB
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _process(path: str):
    import mediapipe as mp

    rgb = _read_rgb(path)
    if rgb is None:
        return None
    h, w = rgb.shape[:2]
    result = _detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not result.face_landmarks:
        return (w, h, None, None, None, None)
    lm = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]], np.float32)
    cats = result.face_blendshapes[0]
    bs = np.array([c.score for c in cats], np.float32)
    names = [c.category_name for c in cats]
    mat = np.asarray(result.facial_transformation_matrixes[0], np.float32)
    return (w, h, lm, bs, mat, names)


def extract(manifest: pd.DataFrame, root: str | Path, model_path: str | Path, out: str | Path,
            workers: int = 4) -> Path:
    root, out = Path(root), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [str(root / p) for p in manifest["image"]]
    n = len(paths)
    landmarks = np.full((n, NUM_LANDMARKS, 3), np.nan, np.float32)
    blendshapes = np.full((n, NUM_BLENDSHAPES), np.nan, np.float32)
    matrix = np.full((n, 4, 4), np.nan, np.float32)
    size = np.zeros((n, 2), np.int32)
    ok = np.zeros(n, bool)
    names = None

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(model_path),)) as pool:
        results = pool.map(_process, paths, chunksize=16)
        for i, r in enumerate(tqdm(results, total=n, desc=f"extract {out.stem}")):
            if r is None:
                continue
            w, h, lm, bs, mat, nm = r
            size[i] = (w, h)
            if lm is None or lm.shape[0] != NUM_LANDMARKS:
                continue
            landmarks[i], matrix[i] = lm, mat
            blendshapes[i, : len(bs)] = bs[:NUM_BLENDSHAPES]
            names = names or nm
            ok[i] = True

    np.savez_compressed(out, landmarks=landmarks, blendshapes=blendshapes, matrix=matrix,
                        image_size=size, ok=ok,
                        blendshape_names=np.array(names or [], dtype=str))
    manifest.to_csv(out.with_suffix(".manifest.csv"), index=False)
    print(f"{ok.sum()}/{n} faces detected -> {out}")
    return out
