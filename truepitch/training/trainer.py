"""Generic CPU/CUDA training loop: class-weighted CE, AdamW, ReduceLROnPlateau,
gradient clipping, early stopping and checkpoint selection on VALIDATION LOSS
(more stable than F1 on ~70 validation samples; documented decision)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ..evaluation.metrics import macro_f1

HP = dict(hidden=64, dropout=0.5, in_dropout=0.2, lr=1e-3, weight_decay=1e-2, batch_size=32,
          max_epochs=100, patience=10, clip=1.0, fusion_dim=64, fusion_dropout=0.3,
          svm_C=[0.1, 1.0, 10.0], fusion_speech="concat")


class Standardizer:
    """Fit on TRAIN rows only."""

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


def class_weights(y: np.ndarray) -> torch.Tensor:
    n = len(y)
    return torch.tensor([n / (2 * max((y == c).sum(), 1)) for c in (0, 1)], dtype=torch.float32)


def fit_torch(model, X_tr, y_tr, X_va, y_va, hp, seed, device="cpu"):
    """X_*: tuples of float tensors (already standardised); y_*: long tensors."""
    torch.manual_seed(seed)
    model.to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(y_tr.numpy()).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    X_tr = [x.to(device) for x in X_tr]; X_va = [x.to(device) for x in X_va]
    y_tr_d, y_va_d = y_tr.to(device), y_va.to(device)
    n, bs = len(y_tr), hp["batch_size"]
    best_loss, best_state, best_ep, bad, hist = float("inf"), None, 0, 0, []
    for ep in range(hp["max_epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(*[x[idx] for x in X_tr]), y_tr_d[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["clip"])
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            lg = model(*X_va)
            vl = crit(lg, y_va_d).item()
        vf1 = macro_f1(y_va.numpy(), lg.argmax(1).cpu().numpy())
        sched.step(vl)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_loss": vl, "val_macro_f1": vf1})
        if vl < best_loss - 1e-4:
            best_loss, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return best_ep, hist


def predict_proba(model, X, device="cpu") -> np.ndarray:
    """P(sarcastic)."""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(*[x.to(device) for x in X]), dim=-1)[:, 1].cpu().numpy()
