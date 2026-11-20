"""Small heads on frozen features. All forward() take tensors and return logits (B, 2)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5, in_dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(in_dropout), nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


class AttentionFusion(nn.Module):
    """Two modality vectors -> shared dim d -> softmax attention over {text, speech} -> weighted sum.
    Optional `extra` features (mismatch etc.) are embedded and concatenated before the classifier.
    Attention weights are stored in `last_attn` (B, 2): descriptive only, NOT a causal explanation."""

    def __init__(self, d_text: int, d_speech: int, d: int = 64, extra_dim: int = 0,
                 dropout: float = 0.3, in_dropout: float = 0.2):
        super().__init__()

        def proj(i):
            return nn.Sequential(nn.Dropout(in_dropout), nn.Linear(i, d), nn.LayerNorm(d),
                                 nn.GELU(), nn.Dropout(dropout))

        self.pt, self.ps = proj(d_text), proj(d_speech)
        self.score = nn.Linear(d, 1)
        self.extra = nn.Sequential(nn.Linear(extra_dim, 8), nn.GELU()) if extra_dim else None
        self.cls = nn.Linear(d + (8 if extra_dim else 0), 2)
        self.last_attn = None

    def forward(self, xt, xs, extra=None):
        H = torch.stack([self.pt(xt), self.ps(xs)], dim=1)          # (B, 2, d)
        a = torch.softmax(self.score(H).squeeze(-1), dim=1)           # (B, 2)
        self.last_attn = a.detach()
        fused = (a.unsqueeze(-1) * H).sum(1)
        if self.extra is not None:
            if extra is None:
                raise ValueError("model built with extra features but none given")
            fused = torch.cat([fused, self.extra(extra)], dim=1)
        return self.cls(fused)
