import numpy as np
import pytest

from truepitch.data.splits import (Fold, assert_no_speaker_leakage, load_folds,
                                   make_speaker_grouped_folds, save_folds)


def synthetic(n=300, n_speakers=25, seed=0):
    """Synthetic TEST FIXTURE ONLY (not dataset content)."""
    rng = np.random.default_rng(seed)
    ids = [f"u{i}" for i in range(n)]
    speakers = [f"S{rng.integers(n_speakers)}" for _ in range(n)]
    labels = rng.integers(0, 2, n).tolist()
    return ids, speakers, labels


def test_no_leakage_and_partition():
    ids, spk, y = synthetic()
    folds = make_speaker_grouped_folds(ids, spk, y, n_folds=5, val_folds=5, seed=1)
    assert len(folds) == 5
    spk_of = dict(zip(ids, spk))
    for f in folds:
        assert_no_speaker_leakage(f, spk_of)
        assert f.train and f.val and f.test


def test_deterministic_and_seed_sensitive():
    ids, spk, y = synthetic()
    a = make_speaker_grouped_folds(ids, spk, y, seed=7)
    b = make_speaker_grouped_folds(ids, spk, y, seed=7)
    c = make_speaker_grouped_folds(ids, spk, y, seed=8)
    assert [f.test for f in a] == [f.test for f in b]
    assert [f.test for f in a] != [f.test for f in c]


def test_leakage_detector_fires():
    bad = Fold(0, train=["a", "b"], val=["c"], test=["d"])
    with pytest.raises(AssertionError):
        assert_no_speaker_leakage(bad, {"a": "X", "b": "Y", "c": "Z", "d": "X"})


def test_too_few_speakers():
    ids, spk, y = synthetic(n=50, n_speakers=3)
    with pytest.raises(ValueError):
        make_speaker_grouped_folds(ids, spk, y, n_folds=5)


def test_roundtrip(tmp_path):
    ids, spk, y = synthetic()
    folds = make_speaker_grouped_folds(ids, spk, y)
    save_folds(folds, tmp_path / "f.json")
    assert [f.test for f in load_folds(tmp_path / "f.json")] == [f.test for f in folds]


def test_dominant_speaker_never_in_validation():
    """Synthetic fixture: one speaker owns ~30% of samples (like a dominant TV character)."""
    rng = np.random.default_rng(3)
    spk = ["BIG"] * 150 + [f"S{i % 19}" for i in range(350)]
    y = [1 if rng.random() < 0.75 else 0 for _ in range(150)] + rng.integers(0, 2, 350).tolist()
    ids = [f"u{i}" for i in range(500)]
    spk_of = dict(zip(ids, spk))
    folds = make_speaker_grouped_folds(ids, spk, y, n_folds=5, val_folds=5, seed=1)
    for f in folds:
        assert_no_speaker_leakage(f, spk_of)
        assert "BIG" not in {spk_of[i] for i in f.val}
        assert len({spk_of[i] for i in f.val}) >= 2
    assert sum("BIG" in {spk_of[i] for i in f.test} for f in folds) == 1