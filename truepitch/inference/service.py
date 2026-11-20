"""Inference: loads the trained final models + frozen encoders and produces predictions with
descriptive (non-causal) explanations. Missing modalities are reported, never faked."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import torch

from ..fusion.models import MLP, AttentionFusion

KEY_PROSODY = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "mean pitch",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "pitch variability",
    "loudness_sma3_amean": "mean loudness",
    "loudness_sma3_stddevNorm": "loudness variability",
    "jitterLocal_sma3nz_amean": "jitter (pitch irregularity)",
    "shimmerLocaldB_sma3nz_amean": "shimmer (amplitude irregularity)",
    "VoicedSegmentsPerSec": "voiced segments per second (speaking-rate proxy)",
}


class HFEncoders:
    """Real frozen encoders (lazy-loaded, cached). Same processing as scripts/extract_features.py."""

    def __init__(self, cfg: dict):
        self.cfg, self._c = cfg, {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def text_embedding(self, text: str) -> np.ndarray:
        from transformers import AutoModel, AutoTokenizer
        name = self.cfg["text"]["primary_encoder"]
        tok, model = self._get("roberta", lambda: (AutoTokenizer.from_pretrained(name),
                                                   AutoModel.from_pretrained(name).eval()))
        enc = tok([text], truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return ((h * mask).sum(1) / mask.sum(1))[0].numpy()

    def text_sentiment(self, text: str) -> float:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = "distilbert-base-uncased-finetuned-sst-2-english"
        tok, model = self._get("sst2", lambda: (AutoTokenizer.from_pretrained(name),
                                                AutoModelForSequenceClassification.from_pretrained(name).eval()))
        id2 = {int(k): v.upper() for k, v in model.config.id2label.items()}
        pos = [k for k, v in id2.items() if v.startswith("POS")][0]
        neg = [k for k, v in id2.items() if v.startswith("NEG")][0]
        with torch.no_grad():
            p = torch.softmax(model(**tok([text], truncation=True, max_length=128, return_tensors="pt")).logits, -1)[0]
        return float(p[pos] - p[neg])

    def egemaps(self, wav: np.ndarray) -> np.ndarray:
        import opensmile
        smile = self._get("smile", lambda: opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                                           feature_level=opensmile.FeatureLevel.Functionals))
        return smile.process_signal(wav, 16000).values[0]

    def w2v_layer_means(self, wav: np.ndarray) -> np.ndarray:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        name = self.cfg["speech"]["ssl_encoder"]
        fe, model = self._get("w2v", lambda: (Wav2Vec2FeatureExtractor.from_pretrained(name),
                                              Wav2Vec2Model.from_pretrained(name).eval()))
        with torch.no_grad():
            hs = model(**fe(wav, sampling_rate=16000, return_tensors="pt"), output_hidden_states=True).hidden_states
        return torch.stack([h[0].mean(0) for h in hs]).numpy()

    def valence(self, wav: np.ndarray) -> np.ndarray:
        from ..speech.valence import load_valence_model, predict_avd
        proc, model = self._get("val", load_valence_model)
        return predict_avd(proc, model, wav)

    def transcribe(self, wav: np.ndarray) -> str:
        from transformers import pipeline
        asr = self._get("whisper", lambda: pipeline("automatic-speech-recognition", model="openai/whisper-base"))
        return asr({"raw": wav, "sampling_rate": 16000})["text"].strip()


def _load_models(d: Path):
    out = []
    for p in sorted(d.glob("seed*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        hp = ck["hp"]
        if "extra" in ck:
            ed = {"abs": 1, "signed": 1, "raw": 2}.get(ck["extra"], 0)
            m = AttentionFusion(len(ck["mu"][0]), len(ck["mu"][1]), hp["fusion_dim"], ed,
                                hp["fusion_dropout"], hp["in_dropout"])
        else:
            m = MLP(len(ck["mu"]), hp["hidden"], hp["dropout"], hp["in_dropout"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        out.append((m, ck))
    if not out:
        raise FileNotFoundError(f"No trained checkpoints in {d}. Run: python scripts/train_final.py")
    return out


def _t(x, mu, sd):
    return torch.from_numpy(((np.asarray(x) - mu) / sd).astype(np.float32))[None]


class Predictor:
    def __init__(self, cfg: dict, encoders=None, models_dir=None):
        root = Path(models_dir or Path(cfg["paths"]["output_dir"]) / "models")
        if not (root / "meta.json").exists():
            raise FileNotFoundError(f"{root / 'meta.json'} not found. Run: python scripts/train_final.py")
        self.meta = json.loads((root / "meta.json").read_text())
        self.models = {n: _load_models(root / n) for n in ("text", "speech", "multimodal")}
        self.enc = encoders or HFEncoders(cfg)
        self.lock = threading.Lock()
        self.w2v_layers = list(self.meta["w2v_layers"])

    def transcribe(self, wav):
        with self.lock:
            return self.enc.transcribe(wav)

    def predict(self, text: str | None, wav: np.ndarray | None, transcript_source: str = "user") -> dict:
        text = (text or "").strip() or None
        has_t, has_a = text is not None, wav is not None
        if not (has_t or has_a):
            raise ValueError("Provide a sentence and/or an audio file.")
        cal, M = self.meta["calibration"], self.meta
        warnings, res = [], {}
        with self.lock:
            if has_t:
                s_t, t_emb = float(self.enc.text_sentiment(text)), self.enc.text_embedding(text)
            if has_a:
                avd = np.asarray(self.enc.valence(wav), float)
                w2v = np.asarray(self.enc.w2v_layer_means(wav))[self.w2v_layers].mean(0)
                eg = np.asarray(self.enc.egemaps(wav))
                s_a = float(np.clip(cal["mu_t"] + (avd[2] - cal["mu_a"]) / cal["sd_a"] * cal["sd_t"], -1, 1))
        if has_t and has_a:
            name, m_val = "multimodal", abs(s_t - s_a)
        else:
            name = "text" if has_t else "speech"
            miss = "audio" if has_t else "text"
            warnings.append(f"No {miss} provided: this is a {name}-only prediction, not a multimodal one.")
        probs, attn = [], []
        with torch.no_grad():
            for m, ck in self.models[name]:
                if name == "multimodal":
                    sp = np.hstack([eg, w2v]) if M["fusion_speech"] == "concat" else (w2v if M["fusion_speech"] == "w2v" else eg)
                    lg = m(_t(t_emb, ck["mu"][0], ck["sd"][0]), _t(sp, ck["mu"][1], ck["sd"][1]),
                           torch.tensor([[m_val]], dtype=torch.float32))
                    attn.append(m.last_attn[0].numpy())
                else:
                    lg = m(_t(t_emb if name == "text" else w2v, ck["mu"], ck["sd"]))
                probs.append(torch.softmax(lg, -1)[0, 1].item())
        p = float(np.mean(probs))
        cv = M.get("cv_metrics", {}).get(name, {})
        res.update({
            "label": "sarcastic" if p > 0.5 else "not sarcastic", "p_sarcastic": round(p, 4),
            "confidence": round(max(p, 1 - p), 4), "seed_std": round(float(np.std(probs)), 4),
            "modality": {"text": "used" if has_t else "unavailable", "audio": "used" if has_a else "unavailable"},
            "model": {"name": name, "cv_macro_f1": cv.get("macro_f1"), "cv_experiment": cv.get("experiment"),
                      "n_ensemble": len(probs)},
            "scores": {"text_sentiment": round(s_t, 3) if has_t else None,
                       "acoustic_valence_native": round(float(avd[2]), 3) if has_a else None,
                       "acoustic_valence_calibrated": round(s_a, 3) if has_a else None,
                       "mismatch": round(m_val, 3) if name == "multimodal" else None},
            "transcript": {"text": text, "source": transcript_source} if has_t else None,
            "warnings": warnings,
        })
        ex = {"textual": [], "acoustic": [], "cross_modal": [], "model": [], "caveats": []}
        if has_t:
            lab = "positive" if s_t > 0.3 else "negative" if s_t < -0.3 else "neutral/mixed"
            ex["textual"].append(f"The wording reads as {lab} (sentiment {s_t:+.2f} on a −1…+1 scale). "
                                 "This comes from a movie-review sentiment model and is noisy for TV dialogue.")
        if has_a:
            ex["acoustic"].append(f"Predicted acoustic valence {s_a:+.2f} (calibrated to the text-sentiment scale); "
                                  f"arousal {avd[0]:.2f}.")
            cols = M.get("egemaps_columns")
            if cols:
                mu, sd = np.array(M["egemaps_mean"]), np.array(M["egemaps_std"])
                for c, nice in KEY_PROSODY.items():
                    if c in cols:
                        z = (eg[cols.index(c)] - mu[cols.index(c)]) / sd[cols.index(c)]
                        side = "higher" if z > 0.5 else "lower" if z < -0.5 else "typical"
                        note = (" Unusually extreme: may reflect background noise, laughter or pitch-tracking "
                                "problems, so interpret with caution.") if abs(z) > 3 else ""
                        ex["acoustic"].append(f"{nice}: {side} than the typical training clip (z = {z:+.1f}).{note}")
        if name == "multimodal":
            rel = "above" if m_val > M["m_median_train"] else "below"
            ex["cross_modal"].append(f"Text sentiment {s_t:+.2f} vs acoustic valence {s_a:+.2f}: mismatch {m_val:.2f}, "
                                     f"{rel} the training median ({M['m_median_train']:.2f}). In our experiments the "
                                     "mismatch feature did not significantly improve accuracy, so treat it as context, "
                                     "not as a sarcasm rule.")
            a = np.mean(attn, 0)
            ex["model"].append(f"Attention weights: text {a[0]:.2f}, speech {a[1]:.2f} (descriptive only).")
            res["attention"] = {"text": round(float(a[0]), 3), "speech": round(float(a[1]), 3)}
        ex["caveats"] += ["These are descriptive signals, not a causal explanation of the prediction.",
                          "Utterance-level model without conversational context; trained on ~690 TV clips.",
                          f"Cross-validated macro-F1 of this model type: {cv['macro_f1']:.2f}. A content-free metadata "
                          "baseline performs similarly (see the Experiments tab)." if cv.get("macro_f1") is not None
                          else "No cross-validated metric available for this model."]
        res["explanation"] = ex
        return res
