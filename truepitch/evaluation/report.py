"""Aggregates experiment outputs into tables, figures and statistics.
Anything not trained is reported as NOT RUN. Test metrics only; validation is shown separately."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.mustard import MustardDataset
from ..experiments.runner import EXPERIMENTS
from .metrics import full_metrics, macro_f1

UNIMODAL = ["E1", "E2", "E3"]
TITLES = {"E0": "[control] Metadata only: duration, loudness, word count (LR)",
          **{e: s.title for e, s in EXPERIMENTS.items()}}


def _fmt(x, nd=3):
    return "NOT RUN" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _md(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_results(out_root: Path):
    runs, preds = {}, {}
    for eid in TITLES:
        d = out_root / "experiments" / eid
        if (d / "runs.csv").exists() and (d / "preds.csv").exists():
            runs[eid], preds[eid] = pd.read_csv(d / "runs.csv"), pd.read_csv(d / "preds.csv")
    return runs, preds


def pooled(preds: pd.DataFrame) -> pd.DataFrame:
    """Seed-ensemble: average P(sarcastic) over seeds per sample (each sample is a test sample once)."""
    g = preds.groupby("id").agg(label=("label", "first"), prob=("prob", "mean"), m=("m", "first"),
                                high_dis=("high_dis", "first"), fold=("fold", "first")).reset_index()
    g["pred"] = (g.prob > 0.5).astype(int)
    return g


def cluster_bootstrap(y, pa, pb, groups, stat, n_boot=2000, seed=0):
    """Speaker-level (cluster) bootstrap of stat(y, pa, pb, idx). Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    point = stat(np.arange(len(y)))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def build_report(cfg: dict, n_boot: int = 2000) -> Path:
    out_root = Path(cfg["paths"]["output_dir"])
    fig_dir = out_root / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    runs, preds = load_results(out_root)
    if not runs:
        raise SystemExit("No experiment results found. Run scripts/run_experiments.py first.")
    ds = MustardDataset(cfg["paths"]["mustard_annotations"], cfg["paths"].get("audio_dir"))
    spk = {s.id: s.speaker for s in ds.load()}
    show = {s.id: s.show for s in ds.load()}
    lines = ["# TruePitch results (auto-generated; test metrics are speaker-independent)\n"]

    # ---- main table
    rows, pool = [], {}
    for eid, title in TITLES.items():
        if eid not in runs:
            rows.append([eid, title, "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN"]); continue
        r, p = runs[eid], pooled(preds[eid]); pool[eid] = p
        per_fold = r.groupby("fold").test_macro_f1.mean()
        pm = full_metrics(p.label, p.pred)
        rows.append([eid, title, _fmt(r.val_macro_f1.mean()),
                     f"{per_fold.mean():.3f} ± {per_fold.std():.3f}", _fmt(pm["macro_f1"]),
                     _fmt(pm["accuracy"]), _fmt(pm["fpr"]), _fmt(pm["fnr"])])
    lines += ["## 1. Main comparison\n",
              "Val F1 = validation macro-F1 (model selection split, NOT a result). "
              "Test F1 = mean ± std over the 5 speaker-held-out folds (seed-averaged within fold). "
              "Pooled = macro-F1 over all out-of-fold test predictions (seed-ensembled).\n",
              _md(rows, ["ID", "Model", "Val F1", "Test macro-F1 (mean ± std over folds)", "Pooled macro-F1",
                         "Pooled acc", "FPR", "FNR"]), ""]

    # ---- additional metrics
    rows = []
    for eid, p in pool.items():
        m = full_metrics(p.label, p.pred)
        w = runs[eid].test_weighted_f1.mean()
        rows.append([eid, _fmt(m["macro_precision"]), _fmt(m["macro_recall"]), _fmt(m["macro_f1"]), _fmt(w)])
    lines += ["## 2. Precision / recall / F1 (pooled)\n",
              _md(rows, ["ID", "Macro precision", "Macro recall", "Macro F1", "Weighted F1 (mean over runs)"]), ""]

    # ---- disagreement analysis
    rows = []
    for eid, p in pool.items():
        hi, lo = p[p.high_dis == 1], p[p.high_dis == 0]
        rows.append([eid, len(hi), _fmt(macro_f1(hi.label, hi.pred)) if len(hi) else "n/a",
                     len(lo), _fmt(macro_f1(lo.label, lo.pred)) if len(lo) else "n/a"])
    lines += ["## 3. Text-speech disagreement analysis (H2)\n",
              "High disagreement = m above the median of m in that fold's TRAINING set (threshold never "
              "uses test data). Subsets are small: interpret with the confidence intervals in section 4.\n",
              _md(rows, ["ID", "n high-disagree", "macro-F1 high", "n low-disagree", "macro-F1 low"]), ""]

    # ---- bootstrap comparisons
    uni = [e for e in UNIMODAL if e in runs]
    best_val = max(uni, key=lambda e: runs[e].val_macro_f1.mean()) if uni else None
    # H1 asks about the BEST unimodal model. We pick it on TEST pooled F1, which is deliberately
    # conservative (it favours the baseline), and we also report every unimodal comparison.
    best_uni = max(uni, key=lambda e: full_metrics(pool[e].label, pool[e].pred)["macro_f1"]) if uni else None
    pairs = []
    for f in ("E4", "E5", "E6", "E7"):
        for u in uni:
            if f in pool:
                pairs.append((f, u, "H1 (strongest unimodal)" if u == best_uni else "H1"))
    pairs += [(a, b, h) for a, b, h in (("E6", "E7", "H3"), ("E6", "E8", "H3 (strict)"),
                                          ("E6", "E9", "exploratory")) if a in pool and b in pool]
    pairs += [(a, "E0", "control (artifact baseline)") for a in ("E1", "E3", "E4", "E5", "E6", "E7")
              if a in pool and "E0" in pool]
    rows = []
    for a, b, tag in pairs:
        ids = pool[a].id.values
        pb = pool[b].set_index("id").loc[ids]
        y = pool[a].label.values; ya, yb = pool[a].pred.values, pb.pred.values
        g = np.array([spk[i] for i in ids])
        stat = lambda idx: macro_f1(y[idx], ya[idx]) - macro_f1(y[idx], yb[idx])
        pt, lo, hi = cluster_bootstrap(y, ya, yb, g, stat, n_boot)
        rows.append([f"{a} vs {b}", tag, f"{pt:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]",
                     "yes" if lo > 0 or hi < 0 else "NO"])
    lines += ["## 4. Paired speaker-level bootstrap (difference in pooled macro-F1)\n",
              f"Strongest unimodal model on test = **{best_uni}** (conservative comparator for H1). "
              f"Highest validation F1 = {best_val}, but SVM validation F1 is inflated by selecting C on it, "
              f"so it is not used as the comparator. {n_boot} speaker-level resamples; 95% interval. "
              f"'CI excludes 0' = evidence beyond noise. Many comparisons, no multiplicity correction: "
              f"treat as exploratory.\n",
              _md(rows, ["Comparison (A vs B)", "Hypothesis", "Δ macro-F1 (A−B)", "95% CI", "CI excludes 0?"]), ""]

    # H2 gain difference
    rows = []
    for a, b in ([("E6", best_uni)] if best_uni and "E6" in pool else []) + ([("E6", "E7")] if "E6" in pool and "E7" in pool else []):
        ids = pool[a].id.values
        pbb = pool[b].set_index("id").loc[ids]
        y = pool[a].label.values; ya, yb = pool[a].pred.values, pbb.pred.values
        hi_m = pool[a].high_dis.values == 1
        g = np.array([spk[i] for i in ids])

        def gain_diff(idx):
            h, l = idx[hi_m[idx]], idx[~hi_m[idx]]
            if len(h) < 10 or len(l) < 10:
                return np.nan
            return (macro_f1(y[h], ya[h]) - macro_f1(y[h], yb[h])) - (macro_f1(y[l], ya[l]) - macro_f1(y[l], yb[l]))
        pt, lo, hi = cluster_bootstrap(y, ya, yb, g, gain_diff, n_boot)
        rows.append([f"{a} vs {b}", f"{pt:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", "yes" if lo > 0 else "NO"])
    if rows:
        lines += ["## 5. H2 test: is the gain larger on high-disagreement samples?\n",
                  "Statistic = (gain on high-disagreement) − (gain on low-disagreement), speaker bootstrap.\n",
                  _md(rows, ["Comparison", "Δgain (high−low)", "95% CI", "CI > 0?"]), ""]

    # ---- per-show breakdown (shortcut diagnostic)
    if any(show.values()):
        lab = dict(zip(next(iter(pool.values())).id, next(iter(pool.values())).label))
        cnt = []
        for sh in sorted({v for v in show.values() if v}):
            ys = [lab[i] for i in lab if show[i] == sh]
            cnt.append([sh, len(ys), _fmt(float(np.mean(ys)))])
        lines += ["## 6a. Class balance per show\n", _md(cnt, ["Show", "n", "sarcastic rate"]), ""]
        rows = []
        for eid, p in pool.items():
            row = [eid]
            for sh in sorted({v for v in show.values() if v}):
                q = p[[show[i] == sh for i in p.id]]
                row.append(_fmt(macro_f1(q.label, q.pred)) if len(q) > 5 and q.label.nunique() == 2 else "n/a")
            rows.append(row)
        lines += ["## 6b. Per-show macro-F1 (shortcut diagnostic; n/a = a single class in that show)\n",
                  _md(rows, ["ID"] + sorted({v for v in show.values() if v})), ""]

    # ---- attention weights
    att = [[e, _fmt(runs[e].attn_text.mean()), _fmt(runs[e].attn_speech.mean())]
           for e in runs if runs[e].attn_text.notna().any()]
    if att:
        lines += ["## 7. Mean attention weights (descriptive, not causal)\n",
                  _md(att, ["ID", "text", "speech"]), ""]

    # ---- figures
    ids_ = [e for e in TITLES if e in pool]
    fig, ax = plt.subplots(figsize=(9, 4))
    means = [runs[e].groupby("fold").test_macro_f1.mean().mean() for e in ids_]
    stds = [runs[e].groupby("fold").test_macro_f1.mean().std() for e in ids_]
    ax.bar(ids_, means, yerr=stds, capsize=4, color="#4C78A8")
    ax.axhline(0.5, ls="--", c="gray", lw=1); ax.set_ylim(0, 1)
    ax.set_ylabel("Test macro-F1 (mean ± std over folds)"); ax.set_title("TruePitch: E1–E9")
    fig.tight_layout(); fig.savefig(fig_dir / "f1_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4)); w = 0.38; x = np.arange(len(ids_))
    hi_f = [macro_f1(pool[e][pool[e].high_dis == 1].label, pool[e][pool[e].high_dis == 1].pred) for e in ids_]
    lo_f = [macro_f1(pool[e][pool[e].high_dis == 0].label, pool[e][pool[e].high_dis == 0].pred) for e in ids_]
    ax.bar(x - w / 2, hi_f, w, label="high disagreement"); ax.bar(x + w / 2, lo_f, w, label="low disagreement")
    ax.set_xticks(x); ax.set_xticklabels(ids_); ax.set_ylim(0, 1); ax.legend(); ax.set_ylabel("Pooled macro-F1")
    fig.tight_layout(); fig.savefig(fig_dir / "disagreement.png", dpi=150); plt.close(fig)

    n = len(ids_); cols = min(n, 5); rws = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(3 * cols, 3 * rws), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, e in zip(axs.ravel(), ids_):
        m = full_metrics(pool[e].label, pool[e].pred); cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.axis("on"); ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_title(e); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
    fig.tight_layout(); fig.savefig(fig_dir / "confusion_matrices.png", dpi=150); plt.close(fig)

    lines.append("Figures: outputs/figures/f1_comparison.png, disagreement.png, confusion_matrices.png")
    out = out_root / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
