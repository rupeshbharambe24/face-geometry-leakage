"""Turn results.csv into markdown tables and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HEADLINE = {
    "identify_match": "rank1",
    "identify_loeo": "rank1",
    "verify_open_set": "eer",
    "utility": "macro_f1",
}


def _fmt(mean: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    out = mean.map(lambda v: f"{v:.3f}")
    if std is not None and std.notna().any().any():
        out = out + std.map(lambda v: "" if pd.isna(v) or v == 0 else f" ± {v:.3f}")
    return out


def build_report(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "results.csv")
    if "heldout" in df:
        df = df[df["heldout"].isna() | (df["heldout"] == "ALL")]
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    lines = [f"# Results: {results_dir.name}", ""]

    for (tag, kind), sub in df.groupby(["experiment", "type"], sort=False):
        if kind == "defense":
            continue
        metric = HEADLINE[kind]
        col = "attacker" if "attacker" in sub and sub["attacker"].notna().any() else "model"
        g = sub.groupby(["representation", col])[metric]
        mean, std = g.mean().unstack(), g.std().unstack()
        lines += [f"## {tag} ({kind}, {metric}, mean ± std over seeds)", "",
                  _fmt(mean, std).to_markdown(), ""]
        if "chance" in sub:
            lines += [f"Chance level: {sub['chance'].mean():.4f}", ""]
        ax = mean.plot.bar(figsize=(9, 4), yerr=std, capsize=2)
        ax.set_ylabel(metric)
        ax.set_title(tag)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{tag}.png", dpi=150)
        plt.close()

    for tag, sub in df[df["type"] == "defense"].groupby("experiment", sort=False):
        cols = ["utility_macro_f1", "id_rank1_loeo", "id_rank1_nn"]
        g = sub.groupby(["representation", "defense"], sort=False)[cols]
        lines += [f"## {tag} (defences: utility vs identity leakage)", "",
                  _fmt(g.mean(), g.std()).to_markdown(), "",
                  f"Chance level: {sub['chance'].mean():.4f}", ""]
        _, ax = plt.subplots(figsize=(7, 5))
        for rep, r in sub.groupby("representation"):
            m = r.groupby("defense", sort=False)[cols].mean()
            ax.scatter(m["id_rank1_loeo"], m["utility_macro_f1"], label=rep)
            for name, row in m.iterrows():
                ax.annotate(name, (row["id_rank1_loeo"], row["utility_macro_f1"]), fontsize=6)
        ax.axvline(sub["chance"].mean(), ls="--", c="grey", lw=1)
        ax.set_xlabel("identity rank-1 (attacker, lower = more private)")
        ax.set_ylabel("expression macro-F1 (higher = more useful)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{tag}.png", dpi=150)
        plt.close()

    out = results_dir / "report.md"
    out.write_text("\n".join(lines))
    return out
