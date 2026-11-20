"""Frozen text encoders (CPU friendly). Gold transcripts only."""
from __future__ import annotations

import numpy as np


def extract_text_embeddings(texts: list[str], model_name: str = "roberta-base",
                            batch_size: int = 16, max_length: int = 128) -> np.ndarray:
    """Mean-pooled last hidden states (attention-masked). Punctuation/case preserved."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).numpy())
        print(f"  text embeddings {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(out)


def extract_text_sentiment(texts: list[str],
                           model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                           batch_size: int = 16) -> np.ndarray:
    """s_t = P(positive) - P(negative), in [-1, 1]. Pretrained on SST-2 (movie reviews):
    domain-shifted for TV dialogue, documented as a limitation."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    id2label = {int(k): v.upper() for k, v in model.config.id2label.items()}
    pos = [k for k, v in id2label.items() if v.startswith("POS")][0]
    neg = [k for k, v in id2label.items() if v.startswith("NEG")][0]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)
        out.append((p[:, pos] - p[:, neg]).numpy())
    return np.concatenate(out)
