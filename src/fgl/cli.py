"""Command line: ``fgl <command> ...`` (or ``python -m fgl``)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _manifest(args):
    from .datasets import build_manifest

    kwargs = {}
    if args.dataset == "celeba":
        kwargs = {"min_images": args.min_images, "max_identities": args.max_identities}
    df = build_manifest(args.dataset, args.root, **kwargs)
    out = Path(args.out or f"data/manifests/{args.dataset}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} images, {df['subject'].nunique()} people -> {out}")
    print(df.groupby("expression").size().to_string())


def _extract(args):
    import pandas as pd

    from .extract import download_model, extract

    model = download_model(args.model)
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    if args.limit:
        manifest = manifest.head(args.limit)
    extract(manifest, args.root, model, args.out, workers=args.workers)


def _info(args):
    from .features import load

    d = load(args.npz)
    m = d.meta
    print(f"{args.npz}: {len(d)} usable faces, {m['subject'].nunique()} people")
    for col in ("session", "expression", "pose"):
        print(f"\n{col}:\n{m[col].value_counts().to_string()}")


def _synth(args):
    from .synth import make_synthetic

    out = make_synthetic(args.out, n_subjects=args.subjects, seed=args.seed,
                         identity_seed=args.identity_seed, prefix=args.prefix)
    print(f"synthetic dataset -> {out}")


def _run(args):
    from .runner import Runner

    Runner(args.config).run(only=args.only)


def _report(args):
    from .report import build_report

    print(f"report -> {build_report(args.results_dir)}")


def _check_gpu(args):
    import torch

    print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fgl", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download-model", help="fetch MediaPipe face_landmarker.task")
    s.add_argument("--dest", default="models/face_landmarker.task")
    s.set_defaults(fn=lambda a: __import__("fgl.extract", fromlist=["x"]).download_model(a.dest))

    s = sub.add_parser("manifest", help="index a dataset folder into a CSV")
    s.add_argument("dataset", choices=["kdef", "jaffe", "ckplus", "celeba", "folder"])
    s.add_argument("--root", required=True)
    s.add_argument("--out")
    s.add_argument("--min-images", type=int, default=20, help="celeba: min images per person")
    s.add_argument("--max-identities", type=int, default=None, help="celeba: cap #people")
    s.set_defaults(fn=_manifest)

    s = sub.add_parser("extract", help="run MediaPipe and cache landmarks/blendshapes")
    s.add_argument("--manifest", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="models/face_landmarker.task")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=_extract)

    s = sub.add_parser("info", help="summarise an extracted .npz")
    s.add_argument("npz")
    s.set_defaults(fn=_info)

    s = sub.add_parser("synth", help="write a synthetic dataset for smoke tests")
    s.add_argument("--out", default="data/landmarks/synthetic.npz")
    s.add_argument("--subjects", type=int, default=30)
    s.add_argument("--seed", type=int, default=0, help="shared face template")
    s.add_argument("--identity-seed", type=int, default=0, help="which synthetic people")
    s.add_argument("--prefix", default="S", help="subject id prefix")
    s.set_defaults(fn=_synth)

    s = sub.add_parser("run", help="run the experiments in a YAML config")
    s.add_argument("config")
    s.add_argument("--only", nargs="*", help="run only these experiment tags")
    s.set_defaults(fn=_run)

    s = sub.add_parser("report", help="tables and figures from a results folder")
    s.add_argument("results_dir")
    s.set_defaults(fn=_report)

    s = sub.add_parser("check-gpu", help="print torch/CUDA status")
    s.set_defaults(fn=_check_gpu)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
