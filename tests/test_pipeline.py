"""Unit + end-to-end tests on SYNTHETIC features (test fixtures only; not dataset content)."""
import json

import numpy as np
import pytest
import torch

from truepitch.data.cache import save_feature
from truepitch.data.splits import make_speaker_grouped_folds, save_folds
from truepitch.evaluation.metrics import full_metrics, macro_f1
from truepitch.experiments.runner import EXPERIMENTS, mismatch_features, run_experiments
from truepitch.fusion.models import MLP, AttentionFusion
from truepitch.training.trainer import HP, Standardizer, fit_torch, predict_proba


def test_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0]); p = np.array([1, 1, 0, 0, 0, 1])
    m = full_metrics(y, p)
    assert m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 2
    assert m["fpr"] == pytest.approx(1 / 3) and m["fnr"] == pytest.approx(1 / 3)
    assert macro_f1(y, p) == pytest.approx(2 / 3)


def test_standardizer_train_only():
    tr = np.random.default_rng(0).normal(5, 2, (100, 3)); te = tr + 100
    sc = Standardizer().fit(tr)
    assert abs(sc(tr).mean()) < 1e-5 and sc(te).mean() > 10  # test not re-centred


def test_mismatch_calibration_uses_train_only():
    rng = np.random.default_rng(1)
    s_t = rng.uniform(-1, 1, 50); s_a = rng.uniform(0, 1, 50); tr = np.arange(30)
    a1, m1, d1 = mismatch_features(s_t, s_a, tr)
    s_a2 = s_a.copy(); s_a2[40:] += 5.0                       # change TEST rows only
    a2, _, _ = mismatch_features(s_t, s_a2, tr)
    assert np.allclose(a1[:30], a2[:30])                      # train-row calibration unchanged
    assert (m1 >= 0).all() and (m1 <= 2).all() and (np.abs(a1) <= 1).all()


def test_attention_shapes_and_weights():
    m = AttentionFusion(10, 12, 16, extra_dim=1)
    out = m(torch.randn(4, 10), torch.randn(4, 12), torch.randn(4, 1))
    assert out.shape == (4, 2) and torch.allclose(m.last_attn.sum(1), torch.ones(4), atol=1e-5)
    with pytest.raises(ValueError):
        m(torch.randn(4, 10), torch.randn(4, 12))


def test_mlp_learns_separable_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300); X = rng.normal(0, 1, (300, 20)); X[:, 0] += 2.5 * y
    Xt = torch.from_numpy(X.astype(np.float32)); yt = torch.from_numpy(y).long()
    torch.manual_seed(0)
    model = MLP(20, 16, 0.3, 0.0)
    fit_torch(model, (Xt[:200],), yt[:200], (Xt[200:250],), yt[200:250], {**HP, "max_epochs": 60}, 0)
    assert macro_f1(y[250:], (predict_proba(model, (Xt[250:],)) > 0.5).astype(int)) > 0.8


def make_fixture(tmp_path, n=150, n_spk=15):
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(n)]; spk = [f"SP{i % n_spk}" for i in range(n)]
    y = rng.integers(0, 2, n)
    ann = {i: {"utterance": "hi", "speaker": s, "sarcasm": bool(l), "show": ["A", "B", "C"][k % 3], "context": [], "context_speakers": []}
           for k, (i, s, l) in enumerate(zip(ids, spk, y))}
    (tmp_path / "a.json").write_text(json.dumps(ann))
    cache = tmp_path / "cache"
    save_feature(cache, "text_roberta", rng.normal(0, 1, (n, 32)) + y[:, None] * 0.7, ids)
    save_feature(cache, "egemaps", rng.normal(0, 1, (n, 88)) + y[:, None] * 0.2, ids)
    save_feature(cache, "w2v_layers", rng.normal(0, 1, (n, 13, 24)) + y[:, None, None] * 0.2, ids)
    save_feature(cache, "sentiment_text", rng.uniform(-1, 1, n), ids)
    save_feature(cache, "valence_audio", rng.uniform(0, 1, (n, 3)), ids)
    folds = make_speaker_grouped_folds(ids, spk, y.tolist(), 5, 5, 0)
    save_folds(folds, tmp_path / "out" / "splits" / "mustard_speaker_folds.json")
    return {"paths": {"mustard_annotations": str(tmp_path / "a.json"), "audio_dir": str(tmp_path / "audio"),
                      "cache_dir": str(cache), "output_dir": str(tmp_path / "out")}}


def test_end_to_end_all_experiments(tmp_path):
    cfg = make_fixture(tmp_path)
    run_experiments(cfg, list(EXPERIMENTS), n_seeds=1, max_epochs=6, device="cpu")
    for e in EXPERIMENTS:
        d = tmp_path / "out" / "experiments" / e
        for f in ("runs.csv", "preds.csv", "config.json", "history.json", "train.log"):
            assert (d / f).exists(), (e, f)
        assert len(list((d / "ckpt").iterdir())) == 5
    from truepitch.evaluation.report import build_report
    res = build_report(cfg, n_boot=50).read_text(encoding="utf-8")
    rows = {ln.split("|")[1].strip(): ln for ln in res.splitlines() if ln.startswith("| E")}
    assert "NOT RUN" in rows["E0"]                                    # E0 not run in this fixture
    assert all("NOT RUN" not in rows[e] for e in EXPERIMENTS)         # E1-E9 all trained
    assert (tmp_path / "out" / "figures" / "f1_comparison.png").exists()
    assert "bootstrap" in res.lower() and "Class balance per show" in res
    assert res.count("H1 (strongest unimodal)") == 4


def test_emotion_model_loads_with_installed_transformers(tmp_path):
    """Regression test: the model-card class used init_weights(), which breaks transformers>=5."""
    pytest.importorskip("transformers")
    from transformers import Wav2Vec2Config

    from truepitch.speech.valence import EmotionModel
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3), num_feat_extract_layers=2,
                         num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_labels=3, final_dropout=0.1)
    EmotionModel(cfg).save_pretrained(tmp_path)
    m = EmotionModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        hidden, out = m(torch.randn(1, 4000))
    assert out.shape == (1, 3)           # arousal, dominance, valence


def test_metadata_baseline_and_final_models(tmp_path):
    import csv
    import json as _json

    import numpy as _np
    import yaml as _yaml  # noqa: F401
    from truepitch.experiments.final import train_final
    from truepitch.experiments.metadata_baseline import run as meta_run

    cfg = make_fixture(tmp_path)
    ids = _json.loads((tmp_path / "cache" / "ids.json").read_text())
    rng = _np.random.default_rng(0)
    with open(tmp_path / "cache" / "audio_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "duration_s", "rms", "peak"]); w.writeheader()
        for i in ids:
            w.writerow({"path": f"x/{i}.wav", "duration_s": rng.uniform(1, 8), "rms": rng.uniform(0.01, .1), "peak": 0.5})
    assert "duration only" in meta_run(cfg)
    assert (tmp_path / "out" / "experiments" / "E0" / "preds.csv").exists()
    run_experiments(cfg, ["E1"], n_seeds=1, max_epochs=3, device="cpu")
    from truepitch.evaluation.report import build_report
    rep = build_report(cfg, n_boot=30).read_text(encoding="utf-8")
    assert "[control] Metadata only" in rep and "control (artifact baseline)" in rep
    meta = _json.loads(train_final(cfg, n_seeds=2, max_epochs=4).read_text())
    assert meta["n_train"] + meta["n_val"] == 150 and "calibration" in meta
    for name in ("text", "speech", "multimodal"):
        assert len(list((tmp_path / "out" / "models" / name).glob("seed*.pt"))) == 2
    ck = torch.load(next((tmp_path / "out" / "models" / "multimodal").glob("seed0.pt")), weights_only=False)
    assert ck["extra"] == "abs" and len(ck["mu"]) == 2
