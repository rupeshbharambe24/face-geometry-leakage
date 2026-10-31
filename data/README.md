# Datasets

None of these can be committed or redistributed. Request each one yourself for non-commercial research and unpack it under `data/raw/`. Re-check the licence terms on the official page before publishing anything derived from them, including landmarks or features.

| Dataset | Role | People | Why it is useful | Where to get it |
|---|---|---|---|---|
| **KDEF** | Main target | 70 | 2 photo sessions (A, B) × 7 expressions × 5 head angles; supports cross-session and pose tests | kdef.se (registration form) |
| **CK+** | Target | 123 | Posed expression sequences with labelled apex frames | Request from the dataset owners (Jeffrey Cohn's lab, University of Pittsburgh) |
| **JAFFE** | Small target | 10 | Classic posed expressions | Zenodo record for JAFFE (request access) |
| **CelebA** | Attacker's public data | 10k+ | In-the-wild photos with identity labels; trains the open-set attacker | mmlab.ie.cuhk.edu.hk/projects/CelebA.html |

## Expected folder layout

```
data/raw/
  KDEF/            AF01/AF01ANS.JPG ...   (any depth; files are found by name)
  jaffe/           KA.AN1.39.tiff ...
  CK+/             cohn-kanade-images/S005/001/*.png
                   Emotion/S005/001/*_emotion.txt
  CelebA/          img_align_celeba/000001.jpg ...
                   identity_CelebA.txt
                   list_attr_celeba.txt
```

Any other dataset laid out as `root/<person>/<expression>/<image>` works with `fgl manifest folder --root ...`.

## Generated files (git-ignored)

- `data/manifests/<name>.csv`: one row per image (subject, session, expression, pose)
- `data/landmarks/<name>.npz` + `<name>.manifest.csv`: MediaPipe landmarks, blendshapes and pose matrices
