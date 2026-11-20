"""Usage: python scripts/evaluate.py   -> outputs/results.md + outputs/figures/*.png"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truepitch.evaluation.report import build_report

out = build_report(yaml.safe_load(open("configs/default.yaml")))
print(Path(out).read_text(encoding="utf-8"))
