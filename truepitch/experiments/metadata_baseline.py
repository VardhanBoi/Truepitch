"""Shortcut control: can trivial clip metadata (duration, loudness, word count) predict sarcasm?
Same speaker-independent folds as E1-E9. If this reaches F1 near the real models, the real
models may be partly exploiting dataset artefacts rather than sarcasm cues."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import pandas as pd

from ..data.cache import load_features
from ..data.mustard import MustardDataset
from ..data.splits import load_folds
from ..evaluation.metrics import full_metrics, macro_f1
from ..training.trainer import Standardizer
from .runner import mismatch_features


def _qc(cfg, samples):
    p = Path(cfg["paths"]["cache_dir"]) / "audio_qc.csv"
    if p.exists():
        rows = {Path(r["path"]).stem: r for r in csv.DictReader(open(p))}
    else:
        from ..speech.features import audio_qc
        rows = {Path(r["path"]).stem: r for r in audio_qc([s.audio_path for s in samples])}
    return rows


def run(cfg: dict) -> str:
    ds = MustardDataset(cfg["paths"]["mustard_annotations"], cfg["paths"].get("audio_dir"))
    samples = ds.load()
    qc = _qc(cfg, samples)
    ids = [s.id for s in samples]
    y = np.array([s.label for s in samples])
    words = np.array([len((s.text or "").split()) for s in samples], float)
    dur = np.array([float(qc[i]["duration_s"]) for i in ids]); rms = np.array([float(qc[i]["rms"]) for i in ids])
    peak = np.array([float(qc[i]["peak"]) for i in ids])
    sets = {"duration only": dur[:, None], "word count only": words[:, None],
            "duration+rms+peak+words": np.stack([dur, rms, peak, words], 1)}
    row = {i: k for k, i in enumerate(ids)}
    folds = load_folds(Path(cfg["paths"]["output_dir"]) / "splits" / "mustard_speaker_folds.json")
    shows = np.array([s.show or "?" for s in samples])
    out = ["Metadata-only control (logistic regression, same speaker-independent folds)",
           f"{'features':<28}{'pooled macro-F1':>16}   per-show macro-F1 (n/a = single class)"]
    feats = load_features(cfg["paths"]["cache_dir"])
    assert feats.ids == ids, "cache order differs from annotation order"
    full = "duration+rms+peak+words"
    for name, X in sets.items():
        pred = np.zeros(len(y), int)
        runs, preds = [], []
        for f in folds:
            tr = np.array([row[i] for i in f.train]); va = np.array([row[i] for i in f.val])
            te = np.array([row[i] for i in f.test])
            sc = Standardizer().fit(X[tr])
            clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc(X[tr]), y[tr])
            pred[te] = clf.predict(sc(X[te]))
            if name == full:   # saved as experiment E0 (same format as E1-E9)
                _, m, _ = mismatch_features(feats.s_t, feats.s_a_raw, tr)
                thr = float(np.median(m[tr])); pt = clf.predict_proba(sc(X[te]))[:, 1]
                tm = full_metrics(y[te], (pt > 0.5).astype(int))
                runs.append({"exp": "E0", "fold": f.fold, "seed": 0, "best_epoch": -1,
                             "val_macro_f1": macro_f1(y[va], clf.predict(sc(X[va]))),
                             **{f"test_{k}": v for k, v in tm.items()},
                             "attn_text": np.nan, "attn_speech": np.nan})
                for j, i in enumerate(te):
                    preds.append({"exp": "E0", "fold": f.fold, "seed": 0, "id": ids[i], "label": int(y[i]),
                                  "prob": float(pt[j]), "pred": int(pt[j] > 0.5), "m": float(m[i]),
                                  "high_dis": int(m[i] > thr)})
        if name == full:
            d0 = Path(cfg["paths"]["output_dir"]) / "experiments" / "E0"; d0.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(runs).to_csv(d0 / "runs.csv", index=False); pd.DataFrame(preds).to_csv(d0 / "preds.csv", index=False)
        per = []
        for sh in sorted(set(shows)):
            m_ = shows == sh
            per.append(f"{sh}={macro_f1(y[m_], pred[m_]):.3f}" if len(set(y[m_])) == 2 else f"{sh}=n/a")
        out.append(f"{name:<28}{macro_f1(y, pred):>16.3f}   " + "  ".join(per))
    txt = "\n".join(out)
    (Path(cfg["paths"]["output_dir"]) / "metadata_baseline.txt").write_text(txt)
    return txt
