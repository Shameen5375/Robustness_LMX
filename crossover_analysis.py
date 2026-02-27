"""Utilities for evaluator-based crossover robustness analysis."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_EVALUATOR_MODELS = {
    "distilbert_sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta_twitter": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "bert_sst2": "textattack/bert-base-uncased-SST-2",
}


def get_device() -> str:
    """Return CUDA device string when available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_evaluators(model_dict: Mapping[str, str], device: str | None = None) -> Dict[str, dict]:
    """Load evaluator tokenizers/models into a single dict."""
    resolved_device = device or get_device()
    evaluators: Dict[str, dict] = {}
    for name, ckpt in model_dict.items():
        print(f"  Loading {name}...")
        evaluators[name] = {
            "tokenizer": AutoTokenizer.from_pretrained(ckpt),
            "model": AutoModelForSequenceClassification.from_pretrained(ckpt).to(resolved_device),
        }
    return evaluators


@torch.no_grad()
def positive_prob(text: str, tokenizer, model, device: str | None = None) -> float:
    """Return P(positive) in [0, 1], robust to label naming conventions."""
    resolved_device = device or get_device()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(resolved_device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        return float(probs[-1].item())

    labels = {i: str(label).lower() for i, label in id2label.items()}
    pos_keys = ["positive", "pos", "label_1"]

    pos_idx = None
    for i, label in labels.items():
        if any(key == label or key in label for key in pos_keys):
            pos_idx = i
            break

    if pos_idx is None:
        pos_idx = int(probs.shape[0] - 1)

    return float(probs[pos_idx].item())


def score_all_models(text: str, evaluators: Mapping[str, Mapping[str, object]], device: str | None = None) -> Dict[str, float]:
    """Compute positive probability for each evaluator model."""
    resolved_device = device or get_device()
    results: Dict[str, float] = {}
    for name, bundle in evaluators.items():
        results[name] = positive_prob(text, bundle["tokenizer"], bundle["model"], resolved_device)
    return results


def cross_model_metrics(scores_dict: Mapping[str, float], thresh: float = 0.5) -> Dict[str, float]:
    """Aggregate agreement/disagreement statistics across evaluators."""
    scores = np.array(list(scores_dict.values()), dtype=float)

    labels = (scores >= thresh).astype(int)
    majority = 1 if labels.mean() >= 0.5 else 0

    agreement = float((labels == majority).mean())
    variance = float(scores.var())
    std = float(scores.std())
    worst = float(scores.min())
    mean = float(scores.mean())

    return {
        "mean_posprob": mean,
        "worst_posprob": worst,
        "std_posprob": std,
        "var_posprob": variance,
        "agreement": agreement,
        "any_disagree": float(agreement < 1.0),
    }
