"""Train the models the app serves. Usage: python scripts/train_final.py [--seeds 5]"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truepitch.experiments.final import train_final

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
print("saved", train_final(yaml.safe_load(open("configs/default.yaml")), a.seeds))
