"""Backend tests with FAKE encoders (no model downloads). Real encoders are exercised only on your machine."""
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app import create_app
from tests.test_pipeline import make_fixture
from truepitch.experiments.final import train_final
from truepitch.experiments.runner import EXPERIMENTS, run_experiments
from truepitch.inference.service import Predictor


class FakeEncoders:
    rng = np.random.default_rng(0)

    def text_embedding(self, t): return self.rng.normal(size=32)
    def text_sentiment(self, t): return -0.8 if "great" in t else 0.5
    def egemaps(self, w): return self.rng.normal(size=88)
    def w2v_layer_means(self, w): return self.rng.normal(size=(13, 24))
    def valence(self, w): return np.array([0.4, 0.5, 0.6])
    def transcribe(self, w): return "this is a fake transcript"


def wav_bytes(sec=2.0, sr=16000):
    t = np.arange(int(sec * sr)) / sr
    y = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype("float32")
    b = io.BytesIO(); sf.write(b, y, sr, format="WAV"); return b.getvalue()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    cfg = make_fixture(tmp)
    cfg["paths"]["mustard_annotations"] = str(tmp / "a.json")
    run_experiments(cfg, ["E1", "E6"], n_seeds=1, max_epochs=3, device="cpu")
    train_final(cfg, n_seeds=2, max_epochs=3)
    return cfg, TestClient(create_app(cfg, Predictor(cfg, FakeEncoders())))


def test_status_and_experiments(env):
    _, c = env
    s = c.get("/api/status").json()
    assert s["final_models"] == {"text": 2, "speech": 2, "multimodal": 2} and s["experiments_trained"] == 2
    ex = {e["id"]: e for e in c.get("/api/experiments").json()}
    assert ex["E1"]["status"] == "TRAINED" and ex["E2"]["status"] == "NOT RUN"   # never fake results


def test_predict_modalities_are_honest(env):
    _, c = env
    both = c.post("/api/predict", data={"text": "great, another meeting"}, files={"audio": ("a.wav", wav_bytes(), "audio/wav")}).json()
    assert both["model"]["name"] == "multimodal" and both["scores"]["mismatch"] is not None and "attention" in both
    t = c.post("/api/predict", data={"text": "hello there"}).json()
    assert t["model"]["name"] == "text" and t["modality"]["audio"] == "unavailable" and t["scores"]["mismatch"] is None
    assert any("text-only" in w for w in t["warnings"])
    a = c.post("/api/predict", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}).json()
    assert a["model"]["name"] == "speech" and a["modality"]["text"] == "unavailable" and a["transcript"] is None
    w = c.post("/api/predict", data={"use_whisper": "true"}, files={"audio": ("a.wav", wav_bytes(), "audio/wav")}).json()
    assert w["model"]["name"] == "multimodal" and "whisper" in w["transcript"]["source"]
    for r in (both, t, a):
        assert 0 <= r["p_sarcastic"] <= 1 and r["explanation"]["caveats"]


def test_input_validation(env):
    _, c = env
    assert c.post("/api/predict").status_code == 400
    assert c.post("/api/predict", data={"text": "x" * 1001}).status_code == 400
    assert c.post("/api/predict", files={"audio": ("a.wav", b"not audio", "audio/wav")}).status_code == 400
    assert c.post("/api/predict", files={"audio": ("a.wav", wav_bytes(35.0), "audio/wav")}).status_code == 400


def test_audio_analysis_and_history(env):
    _, c = env
    a = c.post("/api/analyze_audio", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}).json()
    assert 1.9 < a["duration_s"] < 2.1 and len(a["waveform"]) == 600 and a["features"]["mean_f0_hz"] is not None
    assert 150 < a["features"]["mean_f0_hz"] < 220          # synthetic 180 Hz tone
    h = c.get("/api/history").json()
    assert len(h) >= 4 and {"label", "model", "p_sarcastic"} <= set(h[0])


def test_missing_models_gives_503(tmp_path):
    cfg = make_fixture(tmp_path)
    c = TestClient(create_app(cfg))
    r = c.post("/api/predict", data={"text": "hi"})
    assert r.status_code == 503 and "train_final" in r.json()["detail"]


def test_index_served(env):
    _, c = env
    assert "TruePitch" in c.get("/").text


def _voice(f0=120.0, sec=2.0, sr=16000, noise=0.0):
    t = np.arange(int(sec * sr)) / sr
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 11)) * 0.1
    y = y + noise * np.random.default_rng(0).normal(size=len(t))
    return y.astype("float32")


def test_pitch_tracking_voice_like_and_noise():
    from truepitch.inference.audio import analyze_audio
    a = analyze_audio(_voice(120.0, noise=0.01))
    assert 105 < a["features"]["mean_f0_hz"] < 135 and a["features"]["voiced_fraction"] > 0.5
    assert a["pitch_warning"] is None
    n = analyze_audio(np.random.default_rng(1).normal(0, 0.1, 32000).astype("float32"))
    assert n["pitch_warning"] is not None            # pure noise must raise the reliability warning
