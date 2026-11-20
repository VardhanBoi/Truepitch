import json

import pytest

from truepitch.data.mustard import MustardDataset


def test_loader_reports_missing_audio_and_does_not_invent(tmp_path):
    # Synthetic fixture in the documented sarcasm_data.json format (test only).
    ann = {"1_1": {"utterance": "Fine.", "speaker": " sheldon ", "context": [], "context_speakers": [], "sarcasm": True},
           "1_2": {"utterance": "", "speaker": "PENNY", "context": [], "context_speakers": [], "sarcasm": False}}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(ann))
    ds = MustardDataset(p, audio_dir=tmp_path / "audio")
    s = ds.load()
    rep = ds.validate(s)
    assert s[0].speaker == "SHELDON" and s[0].label == 1
    assert s[1].text is None                     # missing transcript stays missing
    assert rep.n_missing_text == 1 and rep.n_missing_audio == 2


def test_missing_annotation_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATASET_SETUP"):
        MustardDataset(tmp_path / "nope.json").load()
