"""Run the experiments described in a YAML config and write a tidy results.csv."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import protocols
from .defenses import make_defenses
from .features import FaceData, load, make_representation
from .models import get_device

SINGLE_DATASET = {"identify_match", "identify_loeo", "utility", "defense"}


class Runner:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.name = self.cfg.get("name", self.config_path.stem)
        self.out_dir = Path(self.cfg.get("results_dir", "results")) / self.name
        self.seeds = self.cfg.get("seeds", [0])
        self._data: dict[str, FaceData] = {}
        self._features: dict[tuple[str, str], np.ndarray] = {}

    def data(self, key: str) -> FaceData:
        if key not in self._data:
            spec = self.cfg["datasets"][key]
            self._data[key] = load(spec["path"], spec.get("filters"))
            d = self._data[key]
            print(f"[data] {key}: {len(d)} faces, {d.meta['subject'].nunique()} people")
        return self._data[key]

    def features(self, key: str, rep_name: str) -> np.ndarray:
        if (key, rep_name) not in self._features:
            self._features[(key, rep_name)] = make_representation(rep_name).fit_transform(self.data(key))
        return self._features[(key, rep_name)]

    def _run_one(self, exp: dict, rep_name: str, seed: int) -> list[dict]:
        kind = exp["type"]
        model_cfg = {"device": self.cfg.get("device", "auto"), **self.cfg.get("models", {})}
        params = {k: v for k, v in exp.items() if k not in {"type", "tag", "dataset",
                                                            "attacker_dataset", "target",
                                                            "representations", "defenses",
                                                            "stochastic"}}
        if kind in SINGLE_DATASET:
            data = self.data(exp["dataset"])
            X = self.features(exp["dataset"], rep_name)
            fn = getattr(protocols, kind)
            if kind == "defense":
                params["defenses"] = make_defenses(exp["defenses"], seed=seed,
                                                   adversarial_kwargs=model_cfg.get("adversarial"))
            return fn(X, data.meta, cfg=model_cfg, seed=seed, **params)
        if kind == "verify_open_set":
            attacker, target = self.data(exp["attacker_dataset"]), self.data(exp["target"])
            rep = make_representation(rep_name).fit(attacker)
            return protocols.verify_open_set(rep.transform(attacker), attacker.meta,
                                             rep.transform(target), target.meta,
                                             cfg=model_cfg, seed=seed, **params)
        raise KeyError(f"unknown experiment type {kind!r}")

    def run(self, only: list[str] | None = None) -> Path:
        print(f"[run] {self.name}  device={get_device(self.cfg.get('device', 'auto'))}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_path, self.out_dir / "config.yaml")
        csv_path = self.out_dir / "results.csv"
        if csv_path.exists():
            csv_path.unlink()
        all_rows: list[dict] = []
        for exp in self.cfg["experiments"]:
            tag = exp.get("tag", exp["type"])
            if only and tag not in only:
                continue
            reps = exp.get("representations", self.cfg["representations"])
            seeds = self.seeds if exp.get("stochastic", True) else self.seeds[:1]
            for rep_name in reps:
                for seed in seeds:
                    t0 = time.time()
                    rows = self._run_one(exp, rep_name, seed)
                    dataset = exp.get("dataset") or f"{exp['attacker_dataset']}->{exp['target']}"
                    for r in rows:
                        r.update(experiment=tag, type=exp["type"], dataset=dataset,
                                 representation=rep_name, seed=seed)
                    if rows:
                        # experiments have different columns, so rewrite the whole table
                        all_rows.extend(rows)
                        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
                    print(f"[{tag}] {rep_name} seed={seed}: {len(rows)} rows "
                          f"({time.time() - t0:.1f}s) {_headline(rows)}")
        (self.out_dir / "done.json").write_text(json.dumps({"finished": time.ctime()}))
        print(f"[run] results -> {csv_path}")
        return csv_path


def _headline(rows: list[dict]) -> str:
    for key in ("rank1", "eer", "macro_f1", "id_rank1_loeo"):
        vals = [r[key] for r in rows if key in r and r.get("heldout", "ALL") == "ALL"]
        if vals:
            return f"{key}={max(vals) if key != 'eer' else min(vals):.3f}"
    return ""
