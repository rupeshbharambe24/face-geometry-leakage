"""Vectorised landmark geometry.

All functions take landmark arrays shaped (N, K, D): N faces, K landmarks, D = 2 or 3.
Landmarks are stored in pixel units (x*W, y*H, z*W) so that the x/y aspect ratio of
the image does not distort angles (MediaPipe's normalised x and y are scaled by width
and height separately).
"""

from __future__ import annotations

import numpy as np

# MediaPipe FaceMesh indices
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 33, 263
LEGACY_WIDTH = (93, 323)    # cheek to cheek, used by the original liveness system
LEGACY_HEIGHT = (10, 152)   # forehead to chin
NUM_FACE_LANDMARKS = 468    # 468 face points; 468..477 are the refined irises


def _norm(v: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(v, axis=axis, keepdims=keepdims)


def interocular(landmarks: np.ndarray) -> np.ndarray:
    """Outer-eye-corner distance per face, shape (N,)."""
    return _norm(landmarks[:, LEFT_EYE_OUTER] - landmarks[:, RIGHT_EYE_OUTER])


def triplet_angles(points: np.ndarray, triplets: np.ndarray, method: str = "atan2") -> np.ndarray:
    """Unsigned angle at vertex B of each triplet (A, B, C), in degrees, shape (N, T).

    ``atan2`` uses atan2(|u x v|, u . v), which stays accurate near 0 and 180 degrees.
    ``arccos`` reproduces the original liveness code (clipped arccos of the cosine).
    """
    t = np.asarray(triplets, dtype=np.int64)
    u = points[:, t[:, 0]] - points[:, t[:, 1]]
    v = points[:, t[:, 2]] - points[:, t[:, 1]]
    dot = np.sum(u * v, axis=-1)
    if method == "arccos":
        denom = _norm(u) * _norm(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(denom > 0, dot / denom, 1.0)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    if method != "atan2":
        raise ValueError(f"unknown angle method {method!r}")
    if points.shape[-1] == 2:
        cross = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    else:
        cross = _norm(np.cross(u, v))
    return np.degrees(np.arctan2(cross, dot))


def legacy_normalize(landmarks: np.ndarray) -> np.ndarray:
    """Normalisation of the deployed liveness system (2D, separate x and y scales).

    x' = (x - x93) / |x323 - x93|,  y' = (y - y10) / |y152 - y10|.
    Dividing by image width/height first cancels out, so pixel input is fine.
    """
    xy = landmarks[..., :2]
    w1, w2 = xy[:, LEGACY_WIDTH[0], 0], xy[:, LEGACY_WIDTH[1], 0]
    h1, h2 = xy[:, LEGACY_HEIGHT[0], 1], xy[:, LEGACY_HEIGHT[1], 1]
    width = np.abs(w2 - w1)[:, None]
    height = np.abs(h2 - h1)[:, None]
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - w1[:, None]) / width
    out[..., 1] = (xy[..., 1] - h1[:, None]) / height
    return out


def similarity_align_2d(landmarks: np.ndarray) -> np.ndarray:
    """Remove 2D translation, scale and in-plane rotation (eye line horizontal, IOD = 1)."""
    xy = landmarks[..., :2].astype(np.float64)
    xy = xy - xy.mean(axis=1, keepdims=True)
    eye = xy[:, RIGHT_EYE_OUTER] - xy[:, LEFT_EYE_OUTER]
    theta = np.arctan2(eye[:, 1], eye[:, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N, 2, 2)
    xy = np.einsum("nij,nkj->nki", rot, xy)
    return xy / _norm(eye)[:, None, None]


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best similarity transform of centred ``source`` (K, 3) onto centred ``target``."""
    h = source.T @ target
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    corr = np.diag([1.0, 1.0, d])
    r = vt.T @ corr @ u.T
    scale = np.trace(np.diag(s) @ corr) / np.sum(source**2)
    return scale * source @ r.T


def procrustes_align_3d(landmarks: np.ndarray, reference: np.ndarray | None = None,
                        iterations: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Procrustes alignment in 3D (removes head rotation, position, size).

    Returns (aligned landmarks, reference shape). Pass a ``reference`` to align new
    faces to an existing frame. The reference is unit-size and centred.
    """
    pts = landmarks[..., :3].astype(np.float64)
    pts = pts - pts.mean(axis=1, keepdims=True)
    pts = pts / _norm(pts.reshape(len(pts), -1), axis=1)[:, None, None]
    fixed = reference is not None
    ref = reference if fixed else pts[0]
    for _ in range(1 if fixed else iterations):
        aligned = np.stack([_kabsch(p, ref) for p in pts])
        if fixed:
            break
        ref = aligned.mean(axis=0)
        ref = ref / np.linalg.norm(ref)
    return aligned, ref


def delaunay_triplets(mean_shape_2d: np.ndarray, num_points: int = NUM_FACE_LANDMARKS) -> np.ndarray:
    """Vertex-angle triplets from a Delaunay triangulation of the mean face (3 per triangle)."""
    from scipy.spatial import Delaunay

    tri = Delaunay(mean_shape_2d[:num_points]).simplices
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return np.concatenate([np.stack([b, a, c], 1), np.stack([a, b, c], 1),
                           np.stack([a, c, b], 1)]).astype(np.int64)


def pairwise_distances(points: np.ndarray, indices: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Upper-triangle distances among ``indices``, divided by a per-face ``scale``."""
    sub = points[:, indices]
    iu, ju = np.triu_indices(len(indices), k=1)
    d = _norm(sub[:, iu] - sub[:, ju])
    return d / scale[:, None]
