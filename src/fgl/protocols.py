"""Attack and utility protocols. Each returns a list of result dicts (one per row).

Threat model: an app collects only geometry features (angles, distances, blendshapes),
never images. Can an attacker who holds those features tell who the user is?
  - identify_match: closed-set matching of probes against an enrolled gallery
  - identify_loeo: attacker has labelled samples of each person, trains a classifier,
    and is tested on an expression it never saw (leave-one-expression-out)
  - verify_open_set: attacker learns an identity embedding on a public dataset and
    links samples of people it never saw (cross-dataset)
  - utility: expression recognition with subject-independent folds
  - defense: privacy/utility trade-off of each defence on held-out subjects
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import metrics
from .defenses import neutral_reference
from .features import apply_filters
from .models import CosFaceEmbedder, MLPClassifier

UNLABELLED = "unlabelled"


def _scale(fit_X, *Xs):
    scaler = StandardScaler().fit(fit_X)
    return [np.nan_to_num(scaler.transform(x)) for x in Xs]


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _make_classifier(name: str, cfg: dict, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, C=cfg.get("logreg_C", 1.0))
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "mlp":
        return MLPClassifier(seed=seed, device=cfg.get("device", "auto"), **cfg.get("mlp", {}))
    raise KeyError(f"unknown model {name!r}")


def _lda_projection(X, y, max_dim=128):
    n_comp = int(min(len(np.unique(y)) - 1, X.shape[1], max_dim))
    return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=n_comp).fit(X, y)


def identify_match(X, meta, gallery: dict, probe: dict | None = None,
                   attackers=("cosine", "lda"), **_) -> list[dict]:
    g = apply_filters(meta, gallery)
    p = apply_filters(meta, probe) if probe else ~g
    subj = meta["subject"].to_numpy()
    p &= np.isin(subj, subj[g])
    classes = np.unique(subj[g])
    if p.sum() == 0 or len(classes) < 2:
        return []
    Xg, Xp = _scale(X[g], X[g], X[p])
    true_idx = np.searchsorted(classes, subj[p])
    rows = []
    for attacker in attackers:
        if attacker == "lda":
            counts = pd.Series(subj[g]).value_counts()
            if counts.min() < 2:
                continue  # LDA needs within-class scatter
            proj = _lda_projection(Xg, subj[g])
            Eg, Ep = proj.transform(Xg), proj.transform(Xp)
        else:
            Eg, Ep = Xg, Xp
        sim = _l2(Ep) @ _l2(Eg).T
        scores = metrics.class_scores(sim, subj[g], classes)
        rows.append({"attacker": attacker, **metrics.rank_k(scores, true_idx),
                     "n_probe": int(p.sum()), "n_subjects": len(classes),
                     "chance": 1 / len(classes)})
    return rows


def identify_loeo(X, meta, models=("logreg", "lda"), cfg=None, seed=0, **_) -> list[dict]:
    """Train identity on all expressions but one, test on the held-out expression."""
    cfg = cfg or {}
    expr, subj = meta["expression"].to_numpy(), meta["subject"].to_numpy()
    held_out = [e for e in np.unique(expr) if e != UNLABELLED]
    rows = []
    for model_name in models:
        hits1 = hits5 = total = 0
        for e in held_out:
            train, test = expr != e, expr == e
            test &= np.isin(subj, subj[train])
            if test.sum() == 0:
                continue
            Xtr, Xte = _scale(X[train], X[train], X[test])
            clf = _make_classifier(model_name, cfg, seed).fit(Xtr, subj[train])
            proba = clf.predict_proba(Xte)
            r = metrics.rank_k(proba, np.searchsorted(clf.classes_, subj[test]))
            n = int(test.sum())
            hits1, hits5, total = hits1 + r["rank1"] * n, hits5 + r["rank5"] * n, total + n
            rows.append({"attacker": model_name, "heldout": e, **r, "n_probe": n})
        if total:
            n_subj = len(np.unique(subj))
            rows.append({"attacker": model_name, "heldout": "ALL", "rank1": hits1 / total,
                         "rank5": hits5 / total, "n_probe": total, "n_subjects": n_subj,
                         "chance": 1 / n_subj})
    return rows


def _pair_scores(E, labels, max_impostor=2_000_000, seed=0):
    sim = _l2(E) @ _l2(E).T
    iu, ju = np.triu_indices(len(E), k=1)
    same = labels[iu] == labels[ju]
    genuine, impostor = sim[iu[same], ju[same]], sim[iu[~same], ju[~same]]
    if len(impostor) > max_impostor:
        impostor = np.random.default_rng(seed).choice(impostor, max_impostor, replace=False)
    return genuine, impostor


def verify_open_set(Xa, meta_a, Xt, meta_t, attackers=("cosine", "lda", "cosface"),
                    cfg=None, seed=0, max_target=6000, **_) -> list[dict]:
    """Attacker trains on dataset A only; verification is measured on target people."""
    cfg = cfg or {}
    ya = meta_a["subject"].to_numpy()
    if len(Xt) > max_target:
        keep = np.sort(np.random.default_rng(seed).choice(len(Xt), max_target, replace=False))
        Xt, meta_t = Xt[keep], meta_t.iloc[keep].reset_index(drop=True)
    yt = meta_t["subject"].to_numpy()
    Xa_s, Xt_s = _scale(Xa, Xa, Xt)
    rows = []
    for attacker in attackers:
        if attacker == "cosine":
            E = Xt_s
        elif attacker == "lda":
            E = _lda_projection(Xa_s, ya).transform(Xt_s)
        elif attacker == "cosface":
            emb = CosFaceEmbedder(seed=seed, device=cfg.get("device", "auto"), **cfg.get("cosface", {}))
            E = emb.fit(Xa_s, ya).transform(Xt_s)
        else:
            raise KeyError(f"unknown attacker {attacker!r}")
        genuine, impostor = _pair_scores(E, yt, seed=seed)
        rows.append({"attacker": attacker, **metrics.verification(genuine, impostor),
                     "n_genuine": len(genuine), "n_impostor": len(impostor),
                     "n_subjects": len(np.unique(yt))})
    return rows


def utility(X, meta, models=("logreg",), folds=5, cfg=None, seed=0,
            exclude=(UNLABELLED,), **_) -> list[dict]:
    """Expression recognition; folds are split by person so test people are unseen."""
    cfg = cfg or {}
    keep = ~meta["expression"].isin(exclude).to_numpy()
    X, y, groups = X[keep], meta["expression"].to_numpy()[keep], meta["subject"].to_numpy()[keep]
    n_folds = min(folds, len(np.unique(groups)))
    if n_folds < 2 or len(np.unique(y)) < 2:
        return []
    rows = []
    for model_name in models:
        pred = np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
            Xtr, Xte = _scale(X[tr], X[tr], X[te])
            clf = _make_classifier(model_name, cfg, seed).fit(Xtr, y[tr])
            pred[te] = clf.classes_[clf.predict_proba(Xte).argmax(1)]
        rows.append({"model": model_name, "accuracy": accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro"),
                     "n_classes": len(np.unique(y)), "n_samples": len(y)})
    return rows


def _nn_identity_rank1(X, meta):
    """Nearest neighbour among the same people's *other* expressions."""
    expr, subj = meta["expression"].to_numpy(), meta["subject"].to_numpy()
    (Xs,) = _scale(X, X)
    sim = _l2(Xs) @ _l2(Xs).T
    sim[expr[:, None] == expr[None, :]] = -np.inf
    valid = np.isfinite(sim).any(axis=1)
    nn = sim.argmax(axis=1)
    return float(np.mean(subj[nn][valid] == subj[valid]))


def defense(X, meta, defenses, eval_fraction=0.5, cfg=None, seed=0, **_) -> list[dict]:
    """Fit each defence on defender subjects; measure utility and identity leakage on the rest.

    Evaluation uses non-neutral samples of people that have a neutral reference, so every
    defence (including delta_neutral) is scored on the same rows.
    """
    cfg = cfg or {}
    subj = meta["subject"].to_numpy()
    people = np.unique(subj)
    rng = np.random.default_rng(seed)
    eval_people = rng.choice(people, round(len(people) * eval_fraction), replace=False)
    ev = np.isin(subj, eval_people)
    meta_d, meta_e = meta[~ev].reset_index(drop=True), meta[ev].reset_index(drop=True)
    _, has_ref = neutral_reference(X[ev], meta_e)
    rows = []
    for d in defenses:
        d.fit(X[~ev], meta_d)
        Xe, keep = d.transform(X[ev], meta_e)
        mask = keep & has_ref & ~meta_e["expression"].isin(["neutral", UNLABELLED]).to_numpy()
        Xm, mm = Xe[mask], meta_e[mask].reset_index(drop=True)
        util = utility(Xm, mm, models=("logreg",), cfg=cfg, seed=seed)
        loeo = [r for r in identify_loeo(Xm, mm, models=("logreg",), cfg=cfg, seed=seed)
                if r["heldout"] == "ALL"]
        n_subj = len(np.unique(mm["subject"]))
        rows.append({
            "defense": d.label,
            "utility_macro_f1": util[0]["macro_f1"] if util else np.nan,
            "utility_accuracy": util[0]["accuracy"] if util else np.nan,
            "id_rank1_loeo": loeo[0]["rank1"] if loeo else np.nan,
            "id_rank1_nn": _nn_identity_rank1(Xm, mm),
            "chance": 1 / max(n_subj, 1), "n_eval_subjects": n_subj, "n_eval_samples": len(mm),
            "n_features": Xe.shape[1],
        })
    return rows
