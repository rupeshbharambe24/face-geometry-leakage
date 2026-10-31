import numpy as np
import pytest

from fgl import metrics


def test_rank_k():
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.5, 0.7], [0.1, 0.8, 0.3]])
    r = metrics.rank_k(scores, np.array([0, 1, 1]), ks=(1, 2))
    assert r["rank1"] == pytest.approx(2 / 3)
    assert r["rank2"] == pytest.approx(1.0)


def test_class_scores_takes_max_per_class():
    sim = np.array([[0.1, 0.9, 0.3]])
    out = metrics.class_scores(sim, np.array(["a", "a", "b"]), np.array(["a", "b"]))
    np.testing.assert_allclose(out, [[0.9, 0.3]])


def test_verification_separable_and_random():
    rng = np.random.default_rng(0)
    perfect = metrics.verification(rng.uniform(0.6, 1, 500), rng.uniform(0, 0.4, 500))
    assert perfect["eer"] == pytest.approx(0.0, abs=1e-9) and perfect["auc"] == 1.0
    noise = metrics.verification(rng.normal(size=5000), rng.normal(size=5000))
    assert noise["eer"] == pytest.approx(0.5, abs=0.03)
