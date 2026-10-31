import numpy as np
import pytest

from fgl import geometry as geo
from fgl.triplets import LEGACY_TRIPLETS, unique_triplets

rng = np.random.default_rng(0)


def faces(n=5, dim=3):
    base = rng.uniform(0, 200, (478, dim))
    return base + rng.normal(0, 3, (n, 478, dim))


def rotation_3d(a, b, c):
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_legacy_triplets_match_deployed_system():
    assert len(LEGACY_TRIPLETS) == 141
    assert len(unique_triplets()) == 139


def test_right_angle():
    pts = np.array([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
    for method in ("atan2", "arccos"):
        assert geo.triplet_angles(pts, [(0, 1, 2)], method)[0, 0] == pytest.approx(90.0)


def test_atan2_matches_arccos_away_from_degenerate():
    x = faces(4, 3)
    t = unique_triplets()
    np.testing.assert_allclose(geo.triplet_angles(x, t, "atan2"),
                               geo.triplet_angles(x, t, "arccos"), atol=1e-6)


def test_atan2_accurate_near_zero_degrees():
    eps = 1e-7  # nearly collinear: arccos loses precision, atan2 does not
    pts = np.array([[[1.0, eps], [0.0, 0.0], [1.0, -eps]]])
    expected = np.degrees(2 * np.arctan(eps))
    assert geo.triplet_angles(pts, [(0, 1, 2)], "atan2")[0, 0] == pytest.approx(expected, rel=1e-6)


def test_angles_invariant_to_3d_similarity_transform():
    x = faces(3, 3)
    r = rotation_3d(0.3, -0.5, 1.1)
    y = 2.5 * x @ r.T + np.array([10.0, -4.0, 7.0])
    t = unique_triplets()
    np.testing.assert_allclose(geo.triplet_angles(x, t), geo.triplet_angles(y, t), atol=1e-6)


def test_2d_angles_change_with_head_yaw():
    """The pose sensitivity the paper measures: 2D projected angles are not yaw-invariant."""
    x = faces(2, 3)
    y = x @ rotation_3d(0.0, 0.5, 0.0).T
    t = unique_triplets()
    diff = np.abs(geo.triplet_angles(x[..., :2], t) - geo.triplet_angles(y[..., :2], t))
    assert diff.mean() > 1.0


def test_similarity_align_2d_removes_rotation_scale_translation():
    x = faces(2, 2)
    th = 0.7
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    y = 3.0 * x @ r.T + 50
    np.testing.assert_allclose(geo.similarity_align_2d(x), geo.similarity_align_2d(y), atol=1e-8)


def test_procrustes_3d_removes_rotation():
    x = faces(1, 3)
    y = 0.5 * x @ rotation_3d(0.4, 0.9, -0.2).T + 3
    stacked = np.concatenate([x, y])
    aligned, _ = geo.procrustes_align_3d(stacked)
    np.testing.assert_allclose(aligned[0], aligned[1], atol=1e-8)


def test_legacy_normalize_matches_original_code():
    """Same numbers as extract_landmarks() + compute_cosineSim_angles() in gui_liveness.py."""
    x = faces(1, 2)[0]
    w1, w2, h1, h2 = x[93, 0], x[323, 0], x[10, 1], x[152, 1]
    flat = [c for lm in [((p[0] - w1) / abs(w2 - w1), (p[1] - h1) / abs(h2 - h1)) for p in x]
            for c in lm]
    expected = []
    for a, b, c in LEGACY_TRIPLETS:
        ab = np.array([flat[a * 2] - flat[b * 2], flat[a * 2 + 1] - flat[b * 2 + 1]])
        bc = np.array([flat[c * 2] - flat[b * 2], flat[c * 2 + 1] - flat[b * 2 + 1]])
        cos = np.clip(ab @ bc / (np.linalg.norm(ab) * np.linalg.norm(bc)), -1, 1)
        expected.append(np.degrees(np.arccos(cos)))
    got = geo.triplet_angles(geo.legacy_normalize(x[None]), np.asarray(LEGACY_TRIPLETS), "arccos")
    np.testing.assert_allclose(got[0], expected, atol=1e-9)


def test_delaunay_triplets_valid():
    mean = geo.similarity_align_2d(faces(3, 2)).mean(axis=0)
    t = geo.delaunay_triplets(mean)
    assert t.shape[1] == 3 and t.max() < geo.NUM_FACE_LANDMARKS and len(t) % 3 == 0
