"""Runs E1-E9 on the frozen speaker-independent folds. Every experiment uses identical
folds, seeds, feature cache and training protocol.

E1 text | E2 eGeMAPS+SVM | E3 wav2vec2+MLP | E4 early | E5 late | E6 attn+|s_t-s_a| (primary)
E7 attn without mismatch | E8 (extra ablation) attn + raw s_t,s_a | E9 (extra) attn + signed s_t-s_a
"""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.svm import SVC

from ..data.cache import Features, load_features
from ..data.mustard import MustardDataset
from ..data.splits import load_folds
from ..evaluation.metrics import full_metrics, macro_f1
from ..fusion.models import MLP, AttentionFusion
from ..training.trainer import HP, Standardizer, fit_torch, predict_proba
from ..utils.seed import set_global_seed

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
log = logging.getLogger("truepitch.runner")


@dataclass
class ExpSpec:
    id: str
    title: str
    kind: str                  # mlp | svm | late | attn
    blocks: tuple
    extra: str | None = None   # None | abs | raw | signed
    deterministic: bool = False


EXPERIMENTS = {
    "E1": ExpSpec("E1", "Text: RoBERTa (frozen) + MLP", "mlp", ("text",)),
    "E2": ExpSpec("E2", "Speech: eGeMAPS prosody + SVM", "svm", ("egemaps",), deterministic=True),
    "E3": ExpSpec("E3", "Speech: wav2vec 2.0 (frozen) + MLP", "mlp", ("w2v",)),
    "E4": ExpSpec("E4", "Early fusion (concat) + MLP", "mlp", ("text", "speech")),
    "E5": ExpSpec("E5", "Late fusion (weighted probabilities)", "late", ("text", "speech")),
    "E6": ExpSpec("E6", "Attention fusion + mismatch |s_t-s_a|  [PRIMARY]", "attn", ("text", "speech"), "abs"),
    "E7": ExpSpec("E7", "Attention fusion, no mismatch", "attn", ("text", "speech"), None),
    "E8": ExpSpec("E8", "[extra] Attention + raw s_t, s_a (no explicit difference)", "attn", ("text", "speech"), "raw"),
    "E9": ExpSpec("E9", "[extra] Attention + signed s_t-s_a", "attn", ("text", "speech"), "signed"),
}


def get_block(f: Features, name: str, fusion_speech: str) -> np.ndarray:
    if name == "text":
        return f.text
    if name == "egemaps":
        return f.egemaps
    if name == "w2v":
        return f.w2v
    if name == "speech":
        return {"concat": np.hstack([f.egemaps, f.w2v]), "w2v": f.w2v, "egemaps": f.egemaps}[fusion_speech]
    raise KeyError(name)


def mismatch_features(s_t, s_a_raw, train_idx):
    """Calibrate s_a to the s_t scale using TRAIN statistics only (label-free moment matching),
    clip to [-1, 1], then m = |s_t - s_a|, d = s_t - s_a."""
    mu_t, sd_t = s_t[train_idx].mean(), s_t[train_idx].std() + 1e-8
    mu_a, sd_a = s_a_raw[train_idx].mean(), s_a_raw[train_idx].std() + 1e-8
    s_a = np.clip(mu_t + (s_a_raw - mu_a) / sd_a * sd_t, -1.0, 1.0)
    return s_a, np.abs(s_t - s_a), s_t - s_a


def build_extra(mode, s_t, s_a, m, d):
    if mode is None:
        return None
    return {"abs": m[:, None], "raw": np.stack([s_t, s_a], 1), "signed": d[:, None]}[mode].astype(np.float32)


def _T(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _prep(blocks, tr, va, te):
    """Standardise each block with train statistics. Returns (scalers, tensors per split)."""
    sc = [Standardizer().fit(b[tr]) for b in blocks]
    return sc, [[_T(s(b[idx])) for s, b in zip(sc, blocks)] for idx in (tr, va, te)]


def _mlp(in_dim, hp):
    return MLP(in_dim, hp["hidden"], hp["dropout"], hp["in_dropout"])


def fit_predict(spec: ExpSpec, feats: Features, y, tr, va, te, extra, hp, seed, device, ckpt_path):
    """Returns dict(p_val, p_test, best_epoch, history, attn)."""
    yt, yv = torch.from_numpy(y[tr]).long(), torch.from_numpy(y[va]).long()
    out = {"history": None, "attn": None, "best_epoch": -1}
    blocks = [get_block(feats, b, hp["fusion_speech"]) for b in spec.blocks]

    if spec.kind == "svm":
        X = blocks[0]; sc = Standardizer().fit(X[tr])
        best = None
        for C in hp["svm_C"]:
            clf = SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
            clf.fit(sc(X[tr]), y[tr])
            f = macro_f1(y[va], clf.predict(sc(X[va])))
            if best is None or f > best[0]:
                best = (f, C)
        clf = SVC(C=best[1], kernel="rbf", gamma="scale", class_weight="balanced",
                  probability=True, random_state=seed).fit(sc(X[tr]), y[tr])
        out.update(p_val=clf.predict_proba(sc(X[va]))[:, 1], p_test=clf.predict_proba(sc(X[te]))[:, 1],
                   best_C=best[1])
        joblib.dump({"clf": clf, "scaler": sc}, ckpt_path.with_suffix(".joblib"))
        return out

    if spec.kind == "mlp":
        X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
        sc, (Xtr, Xva, Xte) = _prep([X], tr, va, te)
        model = _mlp(X.shape[1], hp)
        out["best_epoch"], out["history"] = fit_torch(model, Xtr, yt, Xva, yv, hp, seed, device)
        out.update(p_val=predict_proba(model, Xva, device), p_test=predict_proba(model, Xte, device))
        torch.save({"state_dict": model.state_dict(), "mu": sc[0].mu, "sd": sc[0].sd, "hp": hp}, ckpt_path)
        return out

    if spec.kind == "late":
        ps_val, ps_test, states, hist = [], [], [], {}
        for name, blk in zip(("text", "speech"), blocks):
            sc, (Xtr, Xva, Xte) = _prep([blk], tr, va, te)
            model = _mlp(blk.shape[1], hp)
            ep, h = fit_torch(model, Xtr, yt, Xva, yv, hp, seed, device)
            ps_val.append(predict_proba(model, Xva, device)); ps_test.append(predict_proba(model, Xte, device))
            states.append(model.state_dict()); hist[name] = h; out["best_epoch"] = max(out["best_epoch"], ep)
        # one scalar weight chosen on VALIDATION only
        ws = np.linspace(0, 1, 11)
        scores = [macro_f1(y[va], ((w * ps_val[0] + (1 - w) * ps_val[1]) > 0.5).astype(int)) for w in ws]
        best = max(range(len(ws)), key=lambda i: (scores[i], -abs(ws[i] - 0.5)))
        w = float(ws[best])
        out.update(p_val=w * ps_val[0] + (1 - w) * ps_val[1], p_test=w * ps_test[0] + (1 - w) * ps_test[1],
                   history=hist, late_w_text=w)
        torch.save({"state_dicts": states, "w_text": w, "hp": hp}, ckpt_path)
        return out

    if spec.kind == "attn":
        sc, (Xtr, Xva, Xte) = _prep(blocks, tr, va, te)
        ex = extra
        ex_tr, ex_va, ex_te = ([_T(ex[i])] if ex is not None else [] for i in (tr, va, te))
        model = AttentionFusion(blocks[0].shape[1], blocks[1].shape[1], hp["fusion_dim"],
                                0 if ex is None else ex.shape[1], hp["fusion_dropout"], hp["in_dropout"])
        out["best_epoch"], out["history"] = fit_torch(model, Xtr + ex_tr, yt, Xva + ex_va, yv, hp, seed, device)
        out["p_val"] = predict_proba(model, Xva + ex_va, device)
        out["p_test"] = predict_proba(model, Xte + ex_te, device)
        out["attn"] = model.last_attn.mean(0).cpu().numpy().tolist()   # [text, speech] on test; descriptive
        torch.save({"state_dict": model.state_dict(), "mu": [s.mu for s in sc], "sd": [s.sd for s in sc],
                    "hp": hp, "extra": spec.extra}, ckpt_path)
        return out
    raise ValueError(spec.kind)


def run_experiments(cfg: dict, exps: list[str], n_seeds: int = 5, max_epochs: int | None = None,
                    device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hp = {**HP, **cfg.get("training", {})}
    if max_epochs:
        hp["max_epochs"] = max_epochs
    paths = cfg["paths"]
    feats = load_features(paths["cache_dir"])
    ds = MustardDataset(paths["mustard_annotations"], paths.get("audio_dir"))
    smp = {s.id: s for s in ds.load()}
    y = np.array([smp[i].label for i in feats.ids], dtype=np.int64)
    folds_path = Path(paths["output_dir"]) / "splits" / "mustard_speaker_folds.json"
    folds = load_folds(folds_path)
    split_hash = json.loads(folds_path.read_text()).get("sha256_12")
    row = {i: k for k, i in enumerate(feats.ids)}

    for eid in exps:
        spec = EXPERIMENTS[eid]
        out_dir = Path(paths["output_dir"]) / "experiments" / eid
        (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "train.log", mode="w"); log.addHandler(fh)
        seeds = [0] if spec.deterministic else list(range(n_seeds))
        log.info("=== %s | %s | seeds=%s | device=%s ===", eid, spec.title, seeds, device)
        t0 = time.time()
        runs, preds, histories = [], [], {}
        for fold in folds:
            tr, va, te = (np.array([row[i] for i in part]) for part in (fold.train, fold.val, fold.test))
            s_a, m, d = mismatch_features(feats.s_t, feats.s_a_raw, tr)
            thr = float(np.median(m[tr]))
            extra = build_extra(spec.extra, feats.s_t, s_a, m, d)
            for seed in seeds:
                set_global_seed(1000 * seed + fold.fold)
                r = fit_predict(spec, feats, y, tr, va, te, extra, hp, 1000 * seed + fold.fold, device,
                                out_dir / "ckpt" / f"fold{fold.fold}_seed{seed}.pt")
                pv, pt = r["p_val"], r["p_test"]
                vm = full_metrics(y[va], (pv > 0.5).astype(int)); tm = full_metrics(y[te], (pt > 0.5).astype(int))
                runs.append({"exp": eid, "fold": fold.fold, "seed": seed, "best_epoch": r["best_epoch"],
                             "val_macro_f1": vm["macro_f1"], **{f"test_{k}": v for k, v in tm.items()},
                             "attn_text": (r["attn"] or [np.nan, np.nan])[0],
                             "attn_speech": (r["attn"] or [np.nan, np.nan])[1],
                             "late_w_text": r.get("late_w_text", np.nan), "svm_C": r.get("best_C", np.nan)})
                for j, i in enumerate(te):
                    preds.append({"exp": eid, "fold": fold.fold, "seed": seed, "id": feats.ids[i],
                                  "label": int(y[i]), "prob": float(pt[j]), "pred": int(pt[j] > 0.5),
                                  "m": float(m[i]), "high_dis": int(m[i] > thr)})
                if r["history"] is not None:
                    histories[f"fold{fold.fold}_seed{seed}"] = r["history"]
                log.info("%s fold %d seed %d | best_ep %d | val F1 %.3f | TEST macro-F1 %.3f",
                         eid, fold.fold, seed, r["best_epoch"], vm["macro_f1"], tm["macro_f1"])
        pd.DataFrame(runs).to_csv(out_dir / "runs.csv", index=False)
        pd.DataFrame(preds).to_csv(out_dir / "preds.csv", index=False)
        (out_dir / "history.json").write_text(json.dumps(histories))
        (out_dir / "config.json").write_text(json.dumps({
            "experiment": eid, "title": spec.title, "kind": spec.kind, "blocks": spec.blocks,
            "extra": spec.extra, "hyperparameters": hp, "seeds": seeds, "split_hash": split_hash,
            "device": device, "torch": torch.__version__, "sklearn": sklearn.__version__,
            "python": platform.python_version(), "runtime_s": round(time.time() - t0, 1),
            "feature_shapes": {"text": feats.text.shape, "egemaps": feats.egemaps.shape, "w2v": feats.w2v.shape},
            "status": "TRAINED"}, indent=1, default=str))
        r_df = pd.DataFrame(runs)
        log.info("%s done in %.0fs | mean TEST macro-F1 %.3f (std over runs %.3f)", eid, time.time() - t0,
                 r_df.test_macro_f1.mean(), r_df.test_macro_f1.std())
        log.removeHandler(fh); fh.close()
