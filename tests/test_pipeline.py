"""End-to-end run on tiny synthetic data (CPU, a few seconds)."""

import pandas as pd
import yaml

from fgl.datasets import build_manifest
from fgl.features import REPRESENTATIONS, load, make_representation
from fgl.report import build_report
from fgl.runner import Runner
from fgl.synth import make_synthetic


def test_every_representation_runs(tmp_path):
    data = load(make_synthetic(tmp_path / "s.npz", n_subjects=4))
    for name in REPRESENTATIONS:
        X = make_representation(name).fit_transform(data)
        assert X.shape[0] == len(data) and X.shape[1] > 0
        assert pd.DataFrame(X).notna().all().all(), name


def test_runner_and_report(tmp_path):
    make_synthetic(tmp_path / "t.npz", n_subjects=8, identity_seed=0, prefix="S")
    make_synthetic(tmp_path / "p.npz", n_subjects=12, identity_seed=1, prefix="P")
    cfg = {
        "name": "t", "results_dir": str(tmp_path / "res"), "device": "cpu", "seeds": [0],
        "datasets": {"t": {"path": str(tmp_path / "t.npz")}, "p": {"path": str(tmp_path / "p.npz")}},
        "representations": ["angles2d", "blendshapes"],
        "models": {"mlp": {"epochs": 2}, "cosface": {"epochs": 2}, "adversarial": {"epochs": 2}},
        "experiments": [
            {"type": "identify_match", "tag": "xs", "dataset": "t",
             "gallery": {"session": ["A"]}, "probe": {"session": ["B"]}},
            {"type": "identify_loeo", "tag": "lo", "dataset": "t", "models": ["logreg", "mlp"]},
            {"type": "verify_open_set", "tag": "os", "attacker_dataset": "p", "target": "t",
             "attackers": ["cosine", "lda", "cosface"]},
            {"type": "utility", "tag": "ut", "dataset": "t"},
            {"type": "defense", "tag": "df", "dataset": "t", "representations": ["angles2d"],
             "defenses": ["none", "delta_neutral", {"name": "noise", "sigma": [1.0]},
                          {"name": "select", "drop_fraction": [0.5]},
                          {"name": "adversarial", "lam": [1.0]}]},
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    df = pd.read_csv(Runner(path).run())
    assert set(df["experiment"]) == {"xs", "lo", "os", "ut", "df"}
    # synthetic people have distinct shapes, so identification must beat chance
    xs = df[(df["experiment"] == "xs") & (df["representation"] == "angles2d")]
    assert (xs["rank1"] > 2 * xs["chance"]).all()
    assert build_report(tmp_path / "res" / "t").exists()


def test_kdef_manifest_parsing(tmp_path):
    for name in ["AF01ANS.JPG", "BM12HAHL.JPG", "AF01NES.jpg", "readme.txt"]:
        p = tmp_path / name[:4] / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"")
    df = build_manifest("kdef", tmp_path).sort_values("image")
    assert list(df["subject"]) == ["F01", "F01", "M12"]
    assert list(df["expression"]) == ["anger", "neutral", "happy"]
    assert list(df["pose"]) == ["S", "S", "HL"]
    assert list(df["session"]) == ["A", "A", "B"]
