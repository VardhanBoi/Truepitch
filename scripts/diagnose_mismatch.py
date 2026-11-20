"""Is the mismatch feature informative on its own? Usage: python scripts/diagnose_mismatch.py
Calibration is label-free, so using all rows here is a descriptive diagnostic (not a model result)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truepitch.data.cache import load_features
from truepitch.data.mustard import MustardDataset
from truepitch.experiments.runner import mismatch_features

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
cfg = yaml.safe_load(open(ap.parse_args().config))
f = load_features(cfg["paths"]["cache_dir"])
lab = {s.id: s.label for s in MustardDataset(cfg["paths"]["mustard_annotations"]).load()}
y = np.array([lab[i] for i in f.ids])
s_a, m, d = mismatch_features(f.s_t, f.s_a_raw, np.arange(len(y)))
out = ["Single-feature AUC for predicting sarcasm (0.50 = no information; <0.5 means higher value => less sarcastic)",
       f"{'feature':<22}{'AUC':>7}{'mean|sarc':>12}{'mean|non':>11}"]
for name, v in [("s_t (text sentiment)", f.s_t), ("s_a (acoustic valence)", s_a),
                ("m = |s_t - s_a|", m), ("d = s_t - s_a", d)]:
    out.append(f"{name:<22}{roc_auc_score(y, v):>7.3f}{v[y == 1].mean():>12.3f}{v[y == 0].mean():>11.3f}")
rho = spearmanr(f.s_t, s_a)[0]
out.append(f"\nSpearman corr(s_t, s_a) = {rho:.3f}  (near 0 = the two scores barely agree about anything)")
out.append("\nSarcasm rate by mismatch quartile (low -> high):")
q = np.quantile(m, [0.25, 0.5, 0.75]); b = np.digitize(m, q)
out.append("  " + "  ".join(f"Q{k+1}: {y[b == k].mean():.2f} (n={int((b == k).sum())})" for k in range(4)))
txt = "\n".join(out); print(txt)
Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
(Path(cfg["paths"]["output_dir"]) / "mismatch_diagnostics.txt").write_text(txt)