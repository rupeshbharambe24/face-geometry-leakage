"""Feature-level privacy defences. Each is fitted on the defender's subjects and
applied to unseen evaluation subjects: ``fit(X, meta)`` then ``transform(X, meta)``
returning (features, keep_mask)."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from .models import AdversarialEncoder


def neutral_reference(X: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row mean of the same person's neutral samples (same session if possible)."""
    ref = np.full_like(X, np.nan, dtype=np.float64)
    neutral = (meta["expression"] == "neutral").to_numpy()
    subj, sess = meta["subject"].to_numpy(), meta["session"].to_numpy()
    for s in np.unique(subj):
        rows = subj == s
        subj_mean = X[rows & neutral].mean(axis=0) if (rows & neutral).any() else None
        for se in np.unique(sess[rows]):
            r = rows & (sess == se)
            n = r & neutral
            if n.any():
                ref[r] = X[n].mean(axis=0)
            elif subj_mean is not None:
                ref[r] = subj_mean
    has_ref = np.isfinite(ref).all(axis=1)
    return ref, has_ref


class Defense:
    label = "none"

    def fit(self, X, meta):
        return self

    def transform(self, X, meta):
        return X, np.ones(len(X), bool)


class DeltaNeutral(Defense):
    """Subtract the user's own neutral face: keeps the expression change, removes static shape."""

    label = "delta_neutral"

    def transform(self, X, meta):
        ref, has_ref = neutral_reference(X, meta)
        return np.nan_to_num(X - ref), has_ref


class GaussianNoise(Defense):
    def __init__(self, sigma: float, seed: int = 0):
        self.sigma, self.seed = sigma, seed
        self.label = f"noise(sigma={sigma:g})"

    def fit(self, X, meta):
        self.std = X.std(axis=0) + 1e-12
        return self

    def transform(self, X, meta):
        rng = np.random.default_rng(self.seed)
        return X + rng.normal(size=X.shape) * self.sigma * self.std, np.ones(len(X), bool)


class Quantize(Defense):
    """Round each feature to a grid of ``step`` x its standard deviation."""

    def __init__(self, step: float):
        self.step = step
        self.label = f"quantize(step={step:g})"

    def fit(self, X, meta):
        self.q = self.step * (X.std(axis=0) + 1e-12)
        return self

    def transform(self, X, meta):
        return np.round(X / self.q) * self.q, np.ones(len(X), bool)


class SelectLowIdentity(Defense):
    """Drop the features whose identity F-score is highest relative to expression F-score."""

    def __init__(self, drop_fraction: float):
        self.drop_fraction = drop_fraction
        self.label = f"select(drop={drop_fraction:g})"

    def fit(self, X, meta):
        labelled = (meta["expression"] != "unlabelled").to_numpy()
        f_id, _ = f_classif(X, meta["subject"].to_numpy())
        f_ex, _ = f_classif(X[labelled], meta["expression"].to_numpy()[labelled])
        ratio = np.nan_to_num(f_id, nan=0.0) / (np.nan_to_num(f_ex, nan=0.0) + 1e-9)
        n_keep = max(1, round(X.shape[1] * (1 - self.drop_fraction)))
        self.keep = np.sort(np.argsort(ratio)[:n_keep])
        return self

    def transform(self, X, meta):
        return X[:, self.keep], np.ones(len(X), bool)


class Adversarial(Defense):
    def __init__(self, lam: float, seed: int = 0, **model_kwargs):
        self.lam, self.seed, self.model_kwargs = lam, seed, model_kwargs
        self.label = f"adversarial(lam={lam:g})"

    def fit(self, X, meta):
        labelled = (meta["expression"] != "unlabelled").to_numpy()
        self.scaler = StandardScaler().fit(X[labelled])
        self.encoder = AdversarialEncoder(lam=self.lam, seed=self.seed, **self.model_kwargs).fit(
            self.scaler.transform(X[labelled]), meta["expression"].to_numpy()[labelled],
            meta["subject"].to_numpy()[labelled])
        return self

    def transform(self, X, meta):
        return self.encoder.transform(self.scaler.transform(X)), np.ones(len(X), bool)


_CLASSES = {
    "none": Defense,
    "delta_neutral": DeltaNeutral,
    "noise": GaussianNoise,
    "quantize": Quantize,
    "select": SelectLowIdentity,
    "adversarial": Adversarial,
}


def make_defenses(specs: list, seed: int = 0, adversarial_kwargs: dict | None = None) -> list[Defense]:
    """Expand specs like ``{"name": "noise", "sigma": [0.25, 0.5, 1]}`` into instances."""
    out = []
    for spec in specs:
        spec = {"name": spec} if isinstance(spec, str) else dict(spec)
        cls = _CLASSES[spec.pop("name")]
        grid = {k: v if isinstance(v, list) else [v] for k, v in spec.items()}
        for values in product(*grid.values()) if grid else [()]:
            kwargs = dict(zip(grid, values, strict=True))
            if cls in (GaussianNoise, Adversarial):
                kwargs["seed"] = seed
            if cls is Adversarial:
                kwargs.update(adversarial_kwargs or {})
            out.append(cls(**kwargs))
    return out
