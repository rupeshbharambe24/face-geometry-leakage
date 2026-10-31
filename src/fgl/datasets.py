"""Manifest builders: one CSV row per image with identity and expression labels.

Manifest columns: dataset, image (path relative to the dataset root), subject,
session, expression, pose. Datasets are licensed, so you obtain them yourself and
point ``--root`` at the extracted folder (see data/README.md).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLUMNS = ["dataset", "image", "subject", "session", "expression", "pose"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

KDEF_EXPR = {"AF": "fear", "AN": "anger", "DI": "disgust", "HA": "happy",
             "NE": "neutral", "SA": "sad", "SU": "surprise"}
JAFFE_EXPR = {"AN": "anger", "DI": "disgust", "FE": "fear", "HA": "happy",
              "NE": "neutral", "SA": "sad", "SU": "surprise"}
CKPLUS_EXPR = {0: "neutral", 1: "anger", 2: "contempt", 3: "disgust", 4: "fear",
               5: "happy", 6: "sad", 7: "surprise"}


def _images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_kdef(root: Path) -> pd.DataFrame:
    """KDEF / KDEF-AM: files like AF01ANS.JPG = session A, female 01, anger, straight.

    Two photo sessions (A, B) of 70 people, 7 expressions, 5 head angles.
    """
    pat = re.compile(r"^([AB])([FM])(\d{2})(AF|AN|DI|HA|NE|SA|SU)(FL|FR|HL|HR|S)$", re.IGNORECASE)
    rows = []
    for p in _images(root):
        m = pat.match(p.stem)
        if not m:
            continue
        session, sex, num, expr, pose = (g.upper() for g in m.groups())
        rows.append(["kdef", _rel(p, root), f"{sex}{num}", session, KDEF_EXPR[expr], pose])
    return pd.DataFrame(rows, columns=COLUMNS)


def build_jaffe(root: Path) -> pd.DataFrame:
    """JAFFE: files like KA.AN1.39.tiff = subject KA, anger. 10 subjects, one session."""
    pat = re.compile(r"^([A-Z]{2})\.([A-Z]{2})\d*\.\d+$", re.IGNORECASE)
    rows = []
    for p in _images(root):
        m = pat.match(p.stem)
        if m and m.group(2).upper() in JAFFE_EXPR:
            rows.append(["jaffe", _rel(p, root), m.group(1).upper(), "1",
                         JAFFE_EXPR[m.group(2).upper()], "S"])
    return pd.DataFrame(rows, columns=COLUMNS)


def build_ckplus(root: Path, peak_frames: int = 1) -> pd.DataFrame:
    """CK+: first frame of each sequence is neutral, last ``peak_frames`` are the apex.

    Expects ``cohn-kanade-images/Sxxx/yyy/*.png`` and optionally ``Emotion/`` labels.
    Apex frames of unlabelled sequences get expression "unlabelled".
    """
    img_root = root / "cohn-kanade-images"
    emo_root = root / "Emotion"
    rows = []
    for seq in sorted(p for p in img_root.glob("S*/*") if p.is_dir()):
        frames = sorted(f for f in seq.iterdir() if f.suffix.lower() == ".png")
        if len(frames) < 2:
            continue
        subject, sequence = seq.parent.name, seq.name
        label = "unlabelled"
        emo = list((emo_root / subject / sequence).glob("*_emotion.txt")) if emo_root.exists() else []
        if emo:
            label = CKPLUS_EXPR[int(float(emo[0].read_text().split()[0]))]
        rows.append(["ckplus", _rel(frames[0], root), subject, sequence, "neutral", "S"])
        for f in frames[-peak_frames:]:
            rows.append(["ckplus", _rel(f, root), subject, sequence, label, "S"])
    return pd.DataFrame(rows, columns=COLUMNS)


def build_celeba(root: Path, min_images: int = 20, max_identities: int | None = None) -> pd.DataFrame:
    """CelebA aligned images with identity labels; expression = smiling / not_smiling.

    Expects ``img_align_celeba/``, ``identity_CelebA.txt`` and optionally
    ``list_attr_celeba.txt``. Used as the attacker's public training set.
    """
    ident = pd.read_csv(root / "identity_CelebA.txt", sep=r"\s+", header=None,
                        names=["file", "subject"])
    counts = ident["subject"].value_counts()
    keep = counts[counts >= min_images].index
    if max_identities is not None:
        keep = keep[:max_identities]
    ident = ident[ident["subject"].isin(keep)]
    expr = pd.Series("unlabelled", index=ident["file"])
    attr_path = root / "list_attr_celeba.txt"
    if attr_path.exists():
        attrs = pd.read_csv(attr_path, sep=r"\s+", skiprows=1)
        smiling = attrs["Smiling"].reindex(ident["file"])
        expr = smiling.map({1: "smiling", -1: "not_smiling"}).fillna("unlabelled")
    return pd.DataFrame({
        "dataset": "celeba",
        "image": "img_align_celeba/" + ident["file"].values,
        "subject": "C" + ident["subject"].astype(str).values,
        "session": "1",
        "expression": expr.values,
        "pose": "unknown",
    })[COLUMNS]


def build_folder(root: Path, name: str = "folder") -> pd.DataFrame:
    """Generic layout: root/<subject>/<expression>/<image> (session = "1")."""
    rows = []
    for p in _images(root):
        parts = p.relative_to(root).parts
        if len(parts) >= 3:
            rows.append([name, _rel(p, root), parts[0], "1", parts[1].lower(), "S"])
    return pd.DataFrame(rows, columns=COLUMNS)


BUILDERS = {
    "kdef": build_kdef,
    "jaffe": build_jaffe,
    "ckplus": build_ckplus,
    "celeba": build_celeba,
    "folder": build_folder,
}


def build_manifest(dataset: str, root: str | Path, **kwargs) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")
    df = BUILDERS[dataset](root, **kwargs)
    if df.empty:
        raise RuntimeError(f"no images matched the {dataset} layout under {root}")
    return df.reset_index(drop=True)
