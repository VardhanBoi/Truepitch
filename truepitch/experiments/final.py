"""Train the FINAL models served by the app, on all labelled data except one speaker-disjoint
validation split used for early stopping. These models have NO held-out test set: their
reported performance is the cross-validated performance of the same recipe (E1/E3/E6)."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.cache import W2V_LAYERS, load_features
from ..data.mustard import MustardDataset
from ..data.splits import load_folds
from ..evaluation.metrics import full_metrics
from ..training.trainer import HP
from ..utils.seed import set_global_seed
from .runner import EXPERIMENTS, build_extra, fit_predict, mismatch_features

FINAL = {"text": "E1", "speech": "E3", "multimodal": "E6"}


def train_final(cfg: dict, n_seeds: int = 5, max_epochs: int | None = None) -> Path:
    hp = {**HP, **cfg.get("training", {})}
    if max_epochs:
        hp["max_epochs"] = max_epochs
    out_root = Path(cfg["paths"]["output_dir"])
    feats = load_features(cfg["paths"]["cache_dir"])
    smp = {s.id: s for s in MustardDataset(cfg["paths"]["mustard_annotations"]).load()}
    y = np.array([smp[i].label for i in feats.ids], dtype=np.int64)
    row = {i: k for k, i in enumerate(feats.ids)}
    folds = load_folds(out_root / "splits" / "mustard_speaker_folds.json")
    vf = max(folds, key=lambda f: len(f.val))                       # largest speaker-disjoint val split
    va = np.array([row[i] for i in vf.val])
    tr = np.array(sorted(set(range(len(y))) - set(va.tolist())))
    s_a, m, d = mismatch_features(feats.s_t, feats.s_a_raw, tr)
    mu_t, sd_t = float(feats.s_t[tr].mean()), float(feats.s_t[tr].std() + 1e-8)
    mu_a, sd_a = float(feats.s_a_raw[tr].mean()), float(feats.s_a_raw[tr].std() + 1e-8)
    extra = build_extra("abs", feats.s_t, s_a, m, d)
    cols_p = Path(cfg["paths"]["cache_dir"]) / "egemaps_columns.json"
    val_f1 = {}
    for name, eid in FINAL.items():
        spec = EXPERIMENTS[eid]
        mdir = out_root / "models" / name
        mdir.mkdir(parents=True, exist_ok=True)
        f1s = []
        for seed in range(n_seeds):
            set_global_seed(seed)
            r = fit_predict(spec, feats, y, tr, va, va, extra if spec.extra else None, hp, seed, "cpu",
                            mdir / f"seed{seed}.pt")
            f1s.append(full_metrics(y[va], (r["p_val"] > 0.5).astype(int))["macro_f1"])
        val_f1[name] = float(np.mean(f1s))
    cv = {}
    for name, eid in FINAL.items():
        p = out_root / "experiments" / eid / "preds.csv"
        if p.exists():
            g = pd.read_csv(p).groupby("id").agg(label=("label", "first"), prob=("prob", "mean"))
            cv[name] = {"experiment": eid, **{k: v for k, v in full_metrics(g.label, (g.prob > 0.5).astype(int)).items()
                                              if k in ("macro_f1", "accuracy", "fpr", "fnr")}}
    meta = {
        "status": "TRAINED_ON_ALL_DATA_EXCEPT_ONE_VALIDATION_SPLIT (no held-out test; see cv_metrics)",
        "created": datetime.datetime.now().isoformat(timespec="seconds"), "n_seeds": n_seeds,
        "n_train": int(len(tr)), "n_val": int(len(va)), "fusion_speech": hp["fusion_speech"],
        "w2v_layers": list(W2V_LAYERS), "models": FINAL,
        "calibration": {"mu_t": mu_t, "sd_t": sd_t, "mu_a": mu_a, "sd_a": sd_a},
        "m_median_train": float(np.median(m[tr])),
        "egemaps_mean": feats.egemaps[tr].mean(0).tolist(), "egemaps_std": (feats.egemaps[tr].std(0) + 1e-8).tolist(),
        "egemaps_columns": json.loads(cols_p.read_text()) if cols_p.exists() else None,
        "val_macro_f1_final_models": val_f1, "cv_metrics": cv,
    }
    (out_root / "models" / "meta.json").write_text(json.dumps(meta, indent=1))
    return out_root / "models" / "meta.json"
