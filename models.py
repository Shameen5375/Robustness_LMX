"""
Utilities for loading and sharing heavyweight model instances.
"""

from __future__ import annotations

from typing import Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers import pipeline

generator = None
sentiment_analysis = None
embedder = None


def load_default_models(
    generator_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    generator_device: int = 0,
    sentiment_device: int = 0,
    embed_device: str = "cuda",
) -> Tuple:
    """
    Lazily instantiate and cache the generator, sentiment classifier, and embedder.
    Repeated calls are safe and simply return the cached instances.
    """
    global generator, sentiment_analysis, embedder

    if generator is None:
        generator = pipeline(
            "text-generation",
            model=generator_model,
            device=generator_device,
        )
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

    if sentiment_analysis is None:
        sentiment_analysis = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            device=sentiment_device,
        )

    if embedder is None:
        embedder = SentenceTransformer(embed_model, device=embed_device)

    return generator, sentiment_analysis, embedder


def get_generator():
    """Return the cached generator, ensuring it is initialized."""
    load_default_models()
    return generator


def get_sentiment_analyzer():
    """Return the cached sentiment analysis pipeline."""
    load_default_models()
    return sentiment_analysis


def get_embedder():
    """Return the cached sentence-transformer embedder."""
    load_default_models()
    return embedder
