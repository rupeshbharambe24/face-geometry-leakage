"""Synthetic landmark dataset for smoke tests (no real faces, no licences).

Each person = template + a fixed identity offset; each expression = a fixed
deformation; each image adds noise and a random similarity transform. Geometry
features should therefore identify people well above chance, which checks that the
attack pipeline works end to end before you spend time on real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import COLUMNS

EXPRESSIONS = ["neutral", "happy", "sad", "anger", "surprise", "fear", "disgust"]


def make_synthetic(out: str | Path, n_subjects: int = 30, sessions=("A", "B"),
                   images_per_cell: int = 1, seed: int = 0, identity_seed: int = 0,
                   prefix: str = "S") -> Path:
    """``seed`` fixes the shared face template and expressions; ``identity_seed`` the people."""
    world = np.random.default_rng(seed)
    rng = np.random.default_rng(10_000 + identity_seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    template = world.uniform(0, 1, (478, 3)) * np.array([200.0, 240.0, 40.0])
    identity = rng.normal(0, 4.0, (n_subjects, 478, 3))
    expression = {e: (0 if e == "neutral" else world.normal(0, 6.0, (478, 3))) for e in EXPRESSIONS}
    lm, bs, rows = [], [], []
    for s in range(n_subjects):
        for sess in sessions:
            for ei, e in enumerate(EXPRESSIONS):
                for k in range(images_per_cell):
                    pts = template + identity[s] + expression[e] + rng.normal(0, 1.0, (478, 3))
                    theta = rng.normal(0, 0.15)
                    c, si = np.cos(theta), np.sin(theta)
                    rot = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1]])
                    pts = rng.uniform(0.8, 1.2) * pts @ rot.T + np.r_[rng.uniform(100, 300, 2), 0]
                    lm.append(pts)
                    b = np.clip(rng.normal(0.05, 0.03, 52), 0, 1)
                    b[(ei * 7) % 52] += 0.7
                    bs.append(b)
                    rows.append(["synthetic", f"{prefix}{s:03d}_{sess}_{e}_{k}.png",
                                 f"{prefix}{s:03d}", sess, e, "S"])
    n = len(rows)
    np.savez_compressed(out, landmarks=np.asarray(lm, np.float32), blendshapes=np.asarray(bs, np.float32),
                        matrix=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
                        image_size=np.tile(np.array([640, 480], np.int32), (n, 1)),
                        ok=np.ones(n, bool), blendshape_names=np.array([f"b{i}" for i in range(52)]))
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out.with_suffix(".manifest.csv"), index=False)
    return out
