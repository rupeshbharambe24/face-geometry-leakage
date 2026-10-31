"""Small PyTorch models: MLP identity attacker, CosFace embedding attacker, and the
adversarial (gradient-reversal) encoder used as a defence. Inputs are standardised
numpy arrays; outputs are numpy arrays."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, dropout: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _batches(n: int, batch_size: int, gen: torch.Generator, device):
    perm = torch.randperm(n, generator=gen).to(device)
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        if len(idx) > 1:  # BatchNorm needs >1 sample
            yield idx


def _train(params, step_fn, n, epochs, lr, weight_decay, batch_size, seed, device):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in _batches(n, batch_size, gen, device):
            opt.zero_grad(set_to_none=True)
            step_fn(idx).backward()
            opt.step()
        sched.step()


def _tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


class MLPClassifier:
    """sklearn-style classifier: fit(X, y) / predict_proba(X)."""

    def __init__(self, hidden=(256, 128), dropout=0.3, epochs=200, lr=1e-3,
                 weight_decay=1e-4, batch_size=256, device="auto", seed=0):
        self.hidden, self.dropout, self.epochs, self.lr = tuple(hidden), dropout, epochs, lr
        self.weight_decay, self.batch_size, self.seed = weight_decay, batch_size, seed
        self.device = get_device(device)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        Xt, yt = _tensor(X, self.device), _tensor(y_idx, self.device, torch.long)
        self.net = mlp(X.shape[1], self.hidden, len(self.classes_), self.dropout).to(self.device)
        loss = nn.CrossEntropyLoss()
        self.net.train()
        _train(self.net.parameters(), lambda i: loss(self.net(Xt[i]), yt[i]), len(Xt),
               self.epochs, self.lr, self.weight_decay, self.batch_size, self.seed, self.device)
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        return torch.softmax(self.net(_tensor(X, self.device)), 1).cpu().numpy()


class CosFaceEmbedder:
    """Metric-learning attacker: learns an identity embedding on the attacker's data
    (CosFace / large-margin cosine loss) and is then applied to unseen people."""

    def __init__(self, dim=128, hidden=(512, 256), dropout=0.2, scale=30.0, margin=0.25,
                 epochs=100, lr=1e-3, weight_decay=5e-4, batch_size=512, device="auto", seed=0):
        self.dim, self.hidden, self.dropout = dim, tuple(hidden), dropout
        self.scale, self.margin, self.epochs, self.lr = scale, margin, epochs, lr
        self.weight_decay, self.batch_size, self.seed = weight_decay, batch_size, seed
        self.device = get_device(device)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        classes, y_idx = np.unique(y, return_inverse=True)
        Xt, yt = _tensor(X, self.device), _tensor(y_idx, self.device, torch.long)
        self.net = mlp(X.shape[1], self.hidden, self.dim, self.dropout).to(self.device)
        weight = nn.Parameter(torch.randn(len(classes), self.dim, device=self.device) * 0.01)
        ce = nn.CrossEntropyLoss()

        def step(i):
            emb = nn.functional.normalize(self.net(Xt[i]), dim=1)
            cos = emb @ nn.functional.normalize(weight, dim=1).T
            onehot = nn.functional.one_hot(yt[i], len(classes)).float()
            return ce(self.scale * (cos - self.margin * onehot), yt[i])

        self.net.train()
        _train(list(self.net.parameters()) + [weight], step, len(Xt), self.epochs, self.lr,
               self.weight_decay, self.batch_size, self.seed, self.device)
        return self

    @torch.no_grad()
    def transform(self, X):
        self.net.eval()
        return nn.functional.normalize(self.net(_tensor(X, self.device)), dim=1).cpu().numpy()


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class AdversarialEncoder:
    """Encoder trained to keep expression information while a gradient-reversed
    adversary head tries to recover identity. ``lam`` sets the privacy pressure."""

    def __init__(self, lam=1.0, dim=64, hidden=(256,), dropout=0.2, epochs=200, lr=1e-3,
                 weight_decay=1e-4, batch_size=256, device="auto", seed=0):
        self.lam, self.dim, self.hidden, self.dropout = lam, dim, tuple(hidden), dropout
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.batch_size, self.seed = batch_size, seed
        self.device = get_device(device)

    def fit(self, X, y_utility, y_identity):
        torch.manual_seed(self.seed)
        _, yu = np.unique(y_utility, return_inverse=True)
        _, yi = np.unique(y_identity, return_inverse=True)
        Xt = _tensor(X, self.device)
        yu_t, yi_t = _tensor(yu, self.device, torch.long), _tensor(yi, self.device, torch.long)
        self.encoder = mlp(X.shape[1], self.hidden, self.dim, self.dropout).to(self.device)
        util_head = nn.Linear(self.dim, int(yu.max()) + 1).to(self.device)
        adv_head = mlp(self.dim, (128,), int(yi.max()) + 1, 0.0).to(self.device)
        ce = nn.CrossEntropyLoss()

        def step(i):
            z = self.encoder(Xt[i])
            return ce(util_head(z), yu_t[i]) + ce(adv_head(_GradReverse.apply(z, self.lam)), yi_t[i])

        params = [*self.encoder.parameters(), *util_head.parameters(), *adv_head.parameters()]
        self.encoder.train()
        _train(params, step, len(Xt), self.epochs, self.lr, self.weight_decay,
               self.batch_size, self.seed, self.device)
        return self

    @torch.no_grad()
    def transform(self, X):
        self.encoder.eval()
        return self.encoder(_tensor(X, self.device)).cpu().numpy()
