"""Utilities for paraphrase-based robustness analysis."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sentiment_lmx.crossover_analysis import cross_model_metrics, score_all_models
from sentiment_lmx.metrics.scoring import compute_semantic_distance


PARAPHRASE_MODEL = "ramsrigouthamg/t5_paraphraser"


def get_device() -> str:
    """Return CUDA device string when available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_paraphraser(model_name: str = PARAPHRASE_MODEL, device: str | None = None):
    """Load tokenizer/model used for paraphrase generation."""
    resolved_device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


@torch.no_grad()
def generate_paraphrases(
    text: str,
    tokenizer,
    model,
    device: str,
    num_return: int = 3,
    num_beams: int = 5,
):
    """Generate unique paraphrases for a sentence."""
    input_text = "paraphrase: " + text + " </s>"
    encoding = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=256,
    ).to(device)

    outputs = model.generate(
        **encoding,
        max_length=128,
        num_beams=num_beams,
        num_return_sequences=num_return,
        temperature=1.0,
        early_stopping=True,
    )

    paraphrases = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    paraphrases = list(set([p for p in paraphrases if p != text]))
    return paraphrases[:num_return]


def paraphrase_robustness(
    text: str,
    evaluators: Mapping[str, Mapping[str, object]],
    tokenizer,
    model,
    device: str,
    num_para: int = 3,
    thresh: float = 0.5,
):
    """Measure sentiment robustness under paraphrase perturbations."""
    original_scores = score_all_models(text, evaluators, device=device)
    original_metrics = cross_model_metrics(original_scores, thresh=thresh)
    s0 = original_metrics["mean_posprob"]

    paras = generate_paraphrases(text, tokenizer, model, device, num_return=num_para)

    para_scores = []
    for paraphrase in paras:
        scores = score_all_models(paraphrase, evaluators, device=device)
        metrics = cross_model_metrics(scores, thresh=thresh)
        para_scores.append(metrics["mean_posprob"])

    para_scores = np.array(para_scores)
    if len(para_scores) == 0:
        return None

    return {
        "original_mean": s0,
        "para_mean_delta": float(np.mean(np.abs(para_scores - s0))),
        "para_worst_drop": float(s0 - np.min(para_scores)),
        "para_std": float(np.std(para_scores)),
        "para_flip_rate": float(np.mean((para_scores >= thresh) != (s0 >= thresh))),
        "paraphrases": paras,
    }


def add_paraphrase_metrics(
    df_robust: pd.DataFrame,
    evaluators,
    tokenizer,
    model,
    device: str,
    num_para: int = 3,
) -> pd.DataFrame:
    """Populate paraphrase robustness columns for each sentence."""
    result_df = df_robust.copy()
    result_df["para_mean_delta"] = np.nan
    result_df["para_worst_drop"] = np.nan
    result_df["para_std"] = np.nan
    result_df["para_flip_rate"] = np.nan

    for idx, row in tqdm(result_df.iterrows(), total=len(result_df)):
        text = row["sentence"]
        try:
            result = paraphrase_robustness(
                text,
                evaluators,
                tokenizer,
                model,
                device,
                num_para=num_para,
            )
            if result is None:
                continue

            result_df.at[idx, "para_mean_delta"] = result["para_mean_delta"]
            result_df.at[idx, "para_worst_drop"] = result["para_worst_drop"]
            result_df.at[idx, "para_std"] = result["para_std"]
            result_df.at[idx, "para_flip_rate"] = result["para_flip_rate"]
        except Exception as exc:
            print(f"Error on sentence: {text}")
            print(exc)
            continue

    return result_df


def build_paraphrase_shift_df(
    df_sentences: pd.DataFrame,
    root_sentence: str,
    evaluators,
    tokenizer,
    model,
    device: str,
    embedder,
) -> pd.DataFrame:
    """Create baseline vs paraphrased displacement table in sentiment/divergence space."""
    root_embedding = embedder.encode([root_sentence])[0]
    paired_points = []

    for _, row in df_sentences.iterrows():
        text = row["sentence"]
        s0 = row["sentiment"]
        d0 = row["divergence"]

        paras = generate_paraphrases(text, tokenizer, model, device, num_return=3)
        for paraphrase in paras:
            scores = score_all_models(paraphrase, evaluators, device=device)
            metrics = cross_model_metrics(scores)
            s1 = metrics["mean_posprob"]
            d1 = compute_semantic_distance(paraphrase, root_embedding)

            paired_points.append(
                {
                    "orig_sentiment": s0,
                    "orig_divergence": d0,
                    "para_sentiment": s1,
                    "para_divergence": d1,
                    "delta_sentiment": s1 - s0,
                    "delta_divergence": d1 - d0,
                    "displacement_2d": np.sqrt((s1 - s0) ** 2 + (d1 - d0) ** 2),
                }
            )

    return pd.DataFrame(paired_points)
