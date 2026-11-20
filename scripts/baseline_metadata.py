import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truepitch.experiments.metadata_baseline import run

print(run(yaml.safe_load(open("configs/default.yaml"))))
