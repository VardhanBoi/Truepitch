"""Extract and cache all features (resumable: existing cache files are skipped).
Usage: python scripts/extract_features.py [--only text|egemaps|w2v|valence] [--force]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truepitch.data.cache import save_feature
from truepitch.data.mustard import MustardDataset

ap = argparse.ArgumentParser()
ap.add_argument("--only", choices=["qc", "text", "egemaps", "w2v", "valence"])
ap.add_argument("--force", action="store_true")
ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
cache = Path(cfg["paths"]["cache_dir"])
ds = MustardDataset(cfg["paths"]["mustard_annotations"], cfg["paths"]["audio_dir"])
samples = ds.load()
rep = ds.validate(samples)
print(rep.summary())
if rep.n_missing_text or rep.n_missing_audio:
    sys.exit("Missing text/audio: fix before extracting features (see report above).")
ids = [s.id for s in samples]
texts = [s.text for s in samples]
paths = [s.audio_path for s in samples]


def todo(name):
    return a.force or not (cache / f"{name}.npy").exists()


def want(step):
    return a.only in (None, step)


t0 = time.time()
if want("qc"):
    from truepitch.speech.features import audio_qc
    rows = audio_qc(paths)
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "audio_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    d = [r["duration_s"] for r in rows]
    print(f"audio QC ok: {len(rows)} clips, duration min/mean/max = {min(d):.2f}/{sum(d)/len(d):.2f}/{max(d):.2f} s")
if want("text") and todo("text_roberta"):
    from truepitch.text.embeddings import extract_text_embeddings
    save_feature(cache, "text_roberta", extract_text_embeddings(texts, cfg["text"]["primary_encoder"]), ids)
if want("text") and todo("sentiment_text"):
    from truepitch.text.embeddings import extract_text_sentiment
    save_feature(cache, "sentiment_text", extract_text_sentiment(texts), ids)
if want("egemaps") and todo("egemaps"):
    from truepitch.speech.features import extract_egemaps
    arr, cols = extract_egemaps(paths)
    save_feature(cache, "egemaps", arr, ids)
    (cache / "egemaps_columns.json").write_text(json.dumps(cols))
if want("w2v") and todo("w2v_layers"):
    from truepitch.speech.features import extract_w2v_layer_means
    save_feature(cache, "w2v_layers", extract_w2v_layer_means(paths, cfg["speech"]["ssl_encoder"]), ids)
if want("valence") and todo("valence_audio"):
    from truepitch.speech.valence import extract_valence
    save_feature(cache, "valence_audio", extract_valence(paths), ids)
print(f"done in {(time.time()-t0)/60:.1f} min. cache contents: {sorted(p.name for p in cache.glob('*.npy'))}")
