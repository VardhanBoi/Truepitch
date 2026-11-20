"""Acoustic feature extraction. No silence trimming on purpose (pauses carry prosody)."""
from __future__ import annotations

import numpy as np

SR = 16000


def load_wav(path, sr: int = SR) -> np.ndarray:
    import soundfile as sf

    y, fs = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if fs != sr:
        import librosa
        y = librosa.resample(y, orig_sr=fs, target_sr=sr)
    if y.size < int(0.1 * sr):
        raise ValueError(f"{path}: audio shorter than 0.1 s")
    if not np.isfinite(y).all():
        raise ValueError(f"{path}: non-finite samples")
    return y


def audio_qc(paths) -> list[dict]:
    rows = []
    for p in paths:
        y = load_wav(p)
        rows.append({"path": str(p), "duration_s": round(len(y) / SR, 3),
                     "rms": float(np.sqrt(np.mean(y ** 2))), "peak": float(np.abs(y).max())})
    return rows


def extract_egemaps(paths):
    """openSMILE eGeMAPSv02 functionals (88 features: F0, loudness, jitter, shimmer, ...)."""
    import opensmile

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    rows, cols = [], None
    for k, p in enumerate(paths):
        df = smile.process_signal(load_wav(p), SR)
        cols = list(df.columns)
        rows.append(df.values[0])
        if k % 50 == 0:
            print(f"  eGeMAPS {k}/{len(paths)}", flush=True)
    return np.stack(rows), cols


def extract_w2v_layer_means(paths, model_name: str = "facebook/wav2vec2-base") -> np.ndarray:
    """Frozen wav2vec 2.0. Returns (N, n_layers+1, hidden): time-mean of every layer."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).eval()
    rows = []
    for k, p in enumerate(paths):
        inp = fe(load_wav(p), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            hs = model(**inp, output_hidden_states=True).hidden_states
        rows.append(torch.stack([h[0].mean(0) for h in hs]).numpy())
        if k % 50 == 0:
            print(f"  wav2vec2 {k}/{len(paths)}", flush=True)
    return np.stack(rows)
