"""Extract 16 kHz mono wav for every annotated utterance (cross-platform; needs ffmpeg on PATH).

Usage: python scripts/extract_audio.py --video-root data/mustard
It searches --video-root recursively for .mp4 files whose filename stem equals an
utterance id from the annotation file (context clips are ignored automatically).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(open("configs/default.yaml"))
ap = argparse.ArgumentParser()
ap.add_argument("--video-root", default="data/mustard")
ap.add_argument("--annotations", default=cfg["paths"]["mustard_annotations"])
ap.add_argument("--out", default=cfg["paths"]["audio_dir"])
a = ap.parse_args()

if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg not found on PATH. Install it first (see docs/DATASET_SETUP.md).")
ids = set(json.load(open(a.annotations, encoding="utf-8")).keys())
found = {}
for p in Path(a.video_root).rglob("*.mp4"):
    if p.stem in ids:
        found.setdefault(p.stem, []).append(p)
print(f"annotated ids: {len(ids)} | matching videos: {len(found)}")
dups = {k: v for k, v in found.items() if len(v) > 1}
if dups:
    print(f"WARNING: {len(dups)} ids match several files; first match used, e.g. {list(dups.items())[0]}")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ok = bad = 0
for uid, paths in sorted(found.items()):
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(paths[0]), "-vn",
                        "-ac", "1", "-ar", "16000", str(out / f"{uid}.wav")], capture_output=True, text=True)
    if r.returncode: bad += 1; print("FAILED", uid, r.stderr.strip()[:200])
    else: ok += 1
missing = sorted(ids - set(found))
print(f"extracted={ok} failed={bad} no_video_found={len(missing)} {missing[:10]}")
