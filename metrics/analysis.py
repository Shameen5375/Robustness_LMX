"""Text quality diagnostics: repetition and vocabulary diversity."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def count_repetitions(text: str, ngram_size: int = 3) -> int:
    """
    Count how many n-grams appear more than once in a text.

    Used to detect degenerate outputs from the LMX generator (e.g.,
    "the sound of the sound of the sound of...").

    Args:
        text:       The text to analyse.
        ngram_size: Size of the n-gram window (default 3).

    Returns:
        Number of distinct n-grams that repeat at least once.
    """
    words = text.lower().split()
    if len(words) < ngram_size:
        return 0

    ngrams = [
        tuple(words[i : i + ngram_size])
        for i in range(len(words) - ngram_size + 1)
    ]
    counts = Counter(ngrams)
    return sum(1 for c in counts.values() if c > 1)


def vocab_diversity(texts: Sequence[str]) -> float:
    """
    Compute the type-token ratio (TTR) across a collection of texts.

    TTR = unique_words / total_words.  A value closer to 1.0 indicates
    higher lexical diversity; a value near 0 indicates heavy repetition
    across the corpus.

    Args:
        texts: A list of strings (e.g., all genomes in a MAP-Elites grid).

    Returns:
        TTR in [0, 1], or 0.0 if the input is empty.
    """
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())

    if not all_words:
        return 0.0

    return len(set(all_words)) / len(all_words)
