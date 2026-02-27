"""
Public API for the sentiment_lmx package.
"""

from .models import load_default_models
from .generation import create_crossover_prompt, do_crossover_fast, process_output
from .metrics.scoring import compute_semantic_distance, compute_sentiment_score
from .algorithm import MapElites2D
from .experiment import run_baseline_experiment, run_pressure_experiment, summarize_experiment

__all__ = [
    "load_default_models",
    "create_crossover_prompt",
    "do_crossover_fast",
    "process_output",
    "compute_semantic_distance",
    "compute_sentiment_score",
    "MapElites2D",
    "run_baseline_experiment",
    "run_pressure_experiment",
    "summarize_experiment",
]
