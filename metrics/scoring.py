"""Core scoring functions: sentiment strength and semantic distance."""

from __future__ import annotations

import numpy as np


def compute_sentiment_score(text: str, target_sentiment: str = "positive") -> float:
    """
    Return sentiment probability in [0, 1] from the optimization evaluator
    (Twitter-RoBERTa, loaded via models.py).
    """
    from sentiment_lmx.models import get_sentiment_analyzer

    analyzer = get_sentiment_analyzer()
    result = analyzer(text[:512])[0]
    label = result["label"].lower()
    score = float(result["score"])

    if target_sentiment == "positive":
        if "pos" in label:
            return score
        elif "neg" in label:
            return 1.0 - score
        else:  # neutral
            return 0.5
    else:  # negative
        if "neg" in label:
            return score
        elif "pos" in label:
            return 1.0 - score
        else:
            return 0.5


def compute_semantic_distance(text: str, root_embedding) -> float:
    """
    Return cosine distance in [0, 1] between text and a pre-computed
    root embedding vector.
    """
    from sentiment_lmx.models import get_embedder

    embedder = get_embedder()
    text_emb = embedder.encode([text])[0]

    norm = np.linalg.norm(text_emb) * np.linalg.norm(root_embedding) + 1e-8
    cos_sim = float(np.dot(text_emb, root_embedding) / norm)
    return float(np.clip(1.0 - cos_sim, 0.0, 1.0))
