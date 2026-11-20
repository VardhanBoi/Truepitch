"""Usage: python scripts/run_experiments.py --exp all --seeds 5
       python scripts/run_experiments.py --exp E1 E2 E3
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truepitch.experiments.runner import EXPERIMENTS, run_experiments

ap = argparse.ArgumentParser()
ap.add_argument("--exp", nargs="+", default=["all"])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--max-epochs", type=int, default=None)
ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
exps = list(EXPERIMENTS) if a.exp == ["all"] else a.exp
bad = [e for e in exps if e not in EXPERIMENTS]
if bad:
    sys.exit(f"Unknown experiments {bad}. Valid: {list(EXPERIMENTS)}")
run_experiments(yaml.safe_load(open(a.config)), exps, a.seeds, a.max_epochs)
