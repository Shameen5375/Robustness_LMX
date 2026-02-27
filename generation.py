"""
Stable LMX crossover utilities for small causal LMs (e.g., Pythia-350m)
Designed to reduce continuation bias and topic drift.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from sentiment_lmx.models import get_generator


# ------------------------------------------------------------
# PROMPT CONSTRUCTION
# ------------------------------------------------------------

def create_crossover_prompt(examples: Sequence[str]) -> str:
    """
    Pattern-based crossover prompt for base (non-instruct) models.
    Uses demonstration + structured boundary to prevent continuation bias.
    """

    prompt = ""

    # In-context structured example (demonstration pattern)
    prompt += "Sentence A: The weather is bad.\n"
    prompt += "Sentence B: The sky is grey.\n"
    prompt += "Combined: The weather is bad and the sky is grey, but it still feels peaceful.\n\n"

    # Real parents
    for i, sentence in enumerate(examples):
        prompt += f"Sentence {chr(65+i)}: {sentence.strip()}\n"

    # Hard boundary to break continuation of last parent
    prompt += "\n###\n"
    prompt += "Combined:"

    return prompt


# ------------------------------------------------------------
# CROSSOVER GENERATION
# ------------------------------------------------------------

def do_crossover_fast(
    pop,
    examples=3,
    temp=0.6,
    batch_size=4,
    max_new_tokens=40,
    chosen_examples=None,
):
    """
    Perform LMX crossover with controlled generation.
    Designed for short, stable offspring sentences.
    """

    generator = get_generator()

    if chosen_examples is None:
        chosen_examples = np.random.choice(pop, examples, replace=False)

    prompt = create_crossover_prompt(chosen_examples)

    model_output = generator(
        [prompt] * batch_size,
        batch_size=batch_size,
        do_sample=True,
        max_new_tokens=max_new_tokens,      # IMPORTANT: use max_new_tokens
        temperature=temp,                   # lower temp for stability
        top_p=0.9,
        repetition_penalty=1.15,            # reduce loops
        return_full_text=False,
    )

    outputs = [x[0]["generated_text"] for x in model_output]
    return outputs


# ------------------------------------------------------------
# OFFSPRING CLEANING
# ------------------------------------------------------------

def clean_text(text):
    """
    Basic cleanup to prevent repetition, multilingual drift, and spillover.
    """

    text = text.strip()

    # Remove obvious multilingual spill markers
    if any(marker in text.lower() for marker in ["italian:", "spanish:", "french:"]):
        return None

    # Remove excessive repetition (e.g., "sound of the sound of the...")
    words = text.split()
    if len(words) > 5:
        if len(set(words[:10])) < 4:
            return None

    # Keep only first sentence
    match = re.split(r'[.!?]', text)
    if match:
        text = match[0].strip() + "."

    # Length filter
    if len(text) < 15:
        return None

    return text


def process_output(output, take_offspring=1):
    """
    Extract cleaned offspring from raw model output.
    Returns up to `take_offspring` valid sentences.
    """

    candidates = []

    # Split by newline first
    lines = output.split("\n")

    for line in lines:
        cleaned = clean_text(line)
        if cleaned:
            candidates.append(cleaned)

        if len(candidates) >= take_offspring:
            break

    return list(set(candidates))
