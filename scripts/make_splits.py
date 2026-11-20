"""Create and save speaker-independent folds. Usage: python scripts/make_splits.py"""
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truepitch.data.mustard import MustardDataset
from truepitch.data.splits import describe_folds, make_speaker_grouped_folds, save_folds
from truepitch.utils.seed import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
cfg = yaml.safe_load(open("configs/default.yaml"))
set_global_seed(cfg["seed"])

ds = MustardDataset(cfg["paths"]["mustard_annotations"], cfg["paths"]["audio_dir"])
samples = ds.load()
print(ds.validate(samples).summary())

sp = cfg["splits"]
folds = make_speaker_grouped_folds([s.id for s in samples], [s.speaker for s in samples],
                                   [s.label for s in samples], sp["n_folds"], sp["val_folds"], sp["seed"],
                                   sp.get("max_val_speaker_frac", 0.5))
for row in describe_folds(folds, {s.id: s.label for s in samples}, {s.id: s.speaker for s in samples}):
    print(row, flush=True)
    print()
out = Path(cfg["paths"]["output_dir"]) / "splits" / "mustard_speaker_folds.json"
print("saved", out, "hash", save_folds(folds, out, {"dataset": ds.name, **sp}))