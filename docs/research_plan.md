# Research plan

## Working title

*Geometry Is a Fingerprint: Identity Leakage from Facial Landmark Angles and Blendshapes, and How to Remove It*

## Problem

Face-interaction systems (liveness checks, facial-gesture input, expression-driven avatars, rehab trackers) increasingly send only **geometry features** instead of images, and present that as privacy-preserving. Examples are triangle vertex angles, landmark distances and MediaPipe/ARKit blendshapes. Whether these features identify people has not been measured. The searches behind this plan (Oct 2026) found:

- **Facial geometry for expression recognition:** angle features from MediaPipe or tracked landmarks are well established, e.g. Siam et al. 2022 and Ghimire et al. 2017. Neither measured identity leakage.
- **Motion re-identifies people:** head and hand motion in VR identified 94% of about 55k users (USENIX Security 2023). Facial dynamics with appearance removed still carry identity (arXiv 2510.11223).
- **Image-based privacy-preserving expression recognition:** de-identification and adversarial representations exist, but they work on images, not on landmark geometry.
- **No audit of FaceMesh angles or blendshapes:** no study was found. Run a fresh Scholar/arXiv check before submission.

## Threat model

The app or server stores feature vectors only. The attacker:

1. **Has labelled samples of enrolled users.** Typical case: an app database with a few sessions per user (closed-set identification).
2. **Has only public face data** (CelebA). They learn an identity embedding and use it to link records of people they have never seen (open-set linkage).

The attacker never sees images of the target people.

## Experiments (implemented in `configs/main.yaml`)

| RQ | Experiment | Metric |
|---|---|---|
| 1 | KDEF session A → B; neutral gallery → expressive probes; leave-one-expression-out (KDEF, CK+, JAFFE) | Rank-1 / rank-5 vs chance |
| 2 | CelebA-trained attackers (no training, LDA, CosFace MLP) → KDEF / CK+ / JAFFE | EER, AUC, TAR@FAR 1% / 0.1% |
| 3 | KDEF frontal gallery → half-profile probes | Rank-1 drop |
| 4 | Expression recognition, subject-independent 5-fold | Macro-F1 |
| 5 | Defences on held-out people: delta-neutral, Gaussian noise, quantisation, identity-aware feature selection, adversarial encoder (gradient reversal) | Utility F1 vs identity rank-1 (Pareto front) |

Representations compared: legacy liveness angles, `atan2` 2D/3D angles, Delaunay angles, distances, aligned 2D/3D coordinates, blendshapes. Seeds 0–2 for stochastic models.

## Expected contributions

1. The first measurement of identity leakage from FaceMesh triplet angles and MediaPipe blendshapes, under closed-set and open-set attackers.
2. A comparison of how the feature representation affects the privacy-utility trade-off.
3. Defences evaluated on unseen people, with a recommended privacy-preserving default (expected candidate: delta-neutral with adversarial training).
4. Open-source code, with no redistributed face data.

## Checks before writing

- Report chance level next to every identification number.
- Fit defences on different people from the evaluation set (already enforced).
- Attack the defences with a fresh attacker trained after the defence, not the adversary head (already enforced).
- Add confidence intervals: bootstrap over people.
- Stronger attackers that reviewers will ask for: a small graph network over landmarks, or a 1D transformer.
- Add a temporal (video) dataset if available, e.g. CK+ full sequences, to test whether dynamics leak more than single frames.

## Target venues

| Tier | Venues |
|---|---|
| Journals | IEEE Transactions on Information Forensics and Security; IEEE Transactions on Dependable and Secure Computing; Pattern Recognition (Elsevier); Computers & Security (Elsevier); IEEE Transactions on Biometrics, Behavior, and Identity Science |
| Conferences | IJCB, PETS (Privacy Enhancing Technologies Symposium), ACM CCS / USENIX Security workshops |
