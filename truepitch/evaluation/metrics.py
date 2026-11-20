from __future__ import annotations

import numpy as np


def _prf(y, yhat, c):
    tp = int(((y == c) & (yhat == c)).sum()); fp = int(((y != c) & (yhat == c)).sum())
    fn = int(((y == c) & (yhat != c)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, int((y == c).sum())


def macro_f1(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean([_prf(y, yhat, c)[2] for c in (0, 1)]))


def full_metrics(y, yhat) -> dict:
    """Positive class = sarcastic (1). FPR = FP/(FP+TN); FNR = FN/(FN+TP)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    s = [_prf(y, yhat, c) for c in (0, 1)]
    n = len(y)
    tp = int(((y == 1) & (yhat == 1)).sum()); tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "macro_precision": float(np.mean([x[0] for x in s])),
        "macro_recall": float(np.mean([x[1] for x in s])),
        "macro_f1": float(np.mean([x[2] for x in s])),
        "weighted_f1": float(sum(x[2] * x[3] for x in s) / n) if n else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "fnr": fn / (fn + tp) if fn + tp else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
