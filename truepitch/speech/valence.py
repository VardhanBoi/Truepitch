"""Acoustic valence s_a from audEERING's dimensional speech-emotion model.

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (license CC BY-NC-SA 4.0,
research use only), trained on MSP-Podcast, i.e. INDEPENDENT of MUStARD sarcasm labels
(no label leakage). Outputs arousal, dominance, valence in approximately 0..1.
Class definitions follow the official model card.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.post_init()  # transformers>=5 requires post_init (model card uses init_weights)

    def forward(self, input_values):
        hidden = torch.mean(self.wav2vec2(input_values)[0], dim=1)
        return hidden, self.classifier(hidden)


def load_valence_model(device: str = "cpu"):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = EmotionModel.from_pretrained(MODEL_NAME).to(device).eval()
    return processor, model


def predict_avd(processor, model, y: np.ndarray, sr: int = 16000, device: str = "cpu") -> np.ndarray:
    """Returns [arousal, dominance, valence] for one waveform."""
    x = processor(y, sampling_rate=sr)["input_values"][0].reshape(1, -1)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))[1]
    return logits.cpu().numpy()[0]


def extract_valence(paths) -> np.ndarray:
    from .features import load_wav

    processor, model = load_valence_model()
    rows = []
    for k, p in enumerate(paths):
        rows.append(predict_avd(processor, model, load_wav(p)))
        if k % 50 == 0:
            print(f"  valence {k}/{len(paths)}", flush=True)
    return np.stack(rows)