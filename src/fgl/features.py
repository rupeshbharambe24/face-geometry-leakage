"""Load cached landmarks and turn them into feature representations.

Every representation has ``fit`` (learns any dataset-level reference, e.g. a mean shape)
and ``transform``. In cross-dataset experiments the representation is fitted on the
attacker's data and applied unchanged to the target data, so features line up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as geo
from .triplets import LEGACY_TRIPLETS, triplet_landmarks, unique_triplets


@dataclass
class FaceData:
    name: str
    landmarks: np.ndarray   # (N, 478, 3) pixels
    blendshapes: np.ndarray  # (N, 52)
    meta: pd.DataFrame       # dataset, image, subject, session, expression, pose

    def __len__(self) -> int:
        return len(self.meta)

    def select(self, mask: np.ndarray) -> FaceData:
        mask = np.asarray(mask)
        return FaceData(self.name, self.landmarks[mask], self.blendshapes[mask],
                        self.meta[mask].reset_index(drop=True))


def apply_filters(meta: pd.DataFrame, filters: dict | None) -> np.ndarray:
    """``{"pose": ["S"], "expression": ["happy", "sad"]}`` -> boolean row mask."""
    mask = np.ones(len(meta), bool)
    for col, values in (filters or {}).items():
        values = values if isinstance(values, (list, tuple)) else [values]
        mask &= meta[col].astype(str).isin([str(v) for v in values]).to_numpy()
    return mask


def load(path: str | Path, filters: dict | None = None) -> FaceData:
    path = Path(path)
    arrays = np.load(path, allow_pickle=False)
    meta = pd.read_csv(path.with_suffix(".manifest.csv"), dtype=str, keep_default_na=False)
    ok = arrays["ok"] & np.isfinite(arrays["landmarks"]).all(axis=(1, 2))
    data = FaceData(path.stem, arrays["landmarks"].astype(np.float64),
                    arrays["blendshapes"].astype(np.float64), meta)
    return data.select(ok & apply_filters(meta, filters))


class Representation:
    name = "base"

    def fit(self, data: FaceData) -> Representation:
        return self

    def transform(self, data: FaceData) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, data: FaceData) -> np.ndarray:
        return self.fit(data).transform(data)


class LegacyAngles(Representation):
    """Exact replica of the deployed liveness features: anisotropic norm + arccos, 141 dims."""

    name = "legacy_angles"

    def transform(self, data):
        pts = geo.legacy_normalize(data.landmarks)
        return geo.triplet_angles(pts, np.asarray(LEGACY_TRIPLETS), method="arccos")


class TripletAngles(Representation):
    """atan2 vertex angles on pixel-space landmarks (2D or 3D)."""

    def __init__(self, dim: int = 2, triplets: str = "legacy"):
        self.dim, self.triplet_set = dim, triplets
        self.name = f"{'delaunay_' if triplets == 'delaunay' else ''}angles{dim}d"
        self.triplets = unique_triplets() if triplets == "legacy" else None

    def fit(self, data):
        if self.triplet_set == "delaunay":
            mean_shape = geo.similarity_align_2d(data.landmarks).mean(axis=0)
            self.triplets = geo.delaunay_triplets(mean_shape)
        return self

    def transform(self, data):
        return geo.triplet_angles(data.landmarks[..., : self.dim], self.triplets)


class Distances(Representation):
    """Pairwise 2D distances among the landmarks used by the legacy triplets, / IOD."""

    name = "distances2d"

    def __init__(self):
        self.indices = triplet_landmarks()

    def transform(self, data):
        pts = data.landmarks[..., :2]
        return geo.pairwise_distances(pts, self.indices, geo.interocular(pts))


class Coords2D(Representation):
    name = "coords2d"

    def transform(self, data):
        return geo.similarity_align_2d(data.landmarks).reshape(len(data), -1)


class Coords3D(Representation):
    """3D Procrustes-aligned coordinates (head rotation removed)."""

    name = "coords3d"

    def fit(self, data):
        _, self.reference = geo.procrustes_align_3d(data.landmarks)
        return self

    def transform(self, data):
        aligned, _ = geo.procrustes_align_3d(data.landmarks, reference=self.reference)
        return aligned.reshape(len(data), -1)


class Blendshapes(Representation):
    name = "blendshapes"

    def transform(self, data):
        return data.blendshapes


_FACTORIES = {
    "legacy_angles": LegacyAngles,
    "angles2d": lambda: TripletAngles(2, "legacy"),
    "angles3d": lambda: TripletAngles(3, "legacy"),
    "delaunay_angles2d": lambda: TripletAngles(2, "delaunay"),
    "distances2d": Distances,
    "coords2d": Coords2D,
    "coords3d": Coords3D,
    "blendshapes": Blendshapes,
}
REPRESENTATIONS = tuple(_FACTORIES)


def make_representation(name: str) -> Representation:
    if name not in _FACTORIES:
        raise KeyError(f"unknown representation {name!r}; choose from {REPRESENTATIONS}")
    return _FACTORIES[name]()
