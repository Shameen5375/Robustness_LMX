"""
MAP-Elites algorithm implementation extracted from the notebook.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from sentiment_lmx.generation import do_crossover_fast, process_output
from sentiment_lmx.metrics.scoring import compute_semantic_distance, compute_sentiment_score


class MapElites2D:
    """
    MAP-Elites with 2D grid:
    - X axis: Sentiment strength (single-model or worst-case across evaluators)
    - Y axis: Semantic divergence

    When robustness_mode=True the fitness criterion switches from the
    single optimization-model score to the worst-case P(positive) across
    all three independently trained evaluators (DistilBERT-SST2,
    RoBERTa-Twitter, BERT-SST2).  Genomes that fail any evaluator are
    disfavoured, pushing the search toward evaluator-invariant solutions.
    """

    def __init__(
        self,
        root_sentence,
        target_sentiment="positive",
        grid_size=(20, 20),
        temperature=0.7,
        batch_size=8,
        robustness_mode: bool = False,
    ):
        """
        Initialize MAP-Elites 2D.

        Args:
            robustness_mode: If True, use worst-case P(positive) across all
                evaluators as the fitness criterion instead of the single
                optimization-model score.
        """
        self.root_sentence = root_sentence
        self.root_embedding = None
        self.target_sentiment = target_sentiment

        self.grid_rows, self.grid_cols = grid_size
        self.temperature = temperature
        self.batch_size = batch_size
        self.robustness_mode = robustness_mode

        self.grid = {}
        self.num_evals = 0
        self.log = []

        self._evaluators = None  # lazily loaded when robustness_mode=True

        self._ensure_root_embedding()
        self._insert(root_sentence)

        print(f"   Grid: {self.grid_rows}{self.grid_cols} = {self.grid_rows * self.grid_cols} cells")
        print(f"   Target sentiment: {target_sentiment}")
        print(f"   Temperature: {temperature}")
        print(f"   Robustness mode: {robustness_mode}")

    def _ensure_root_embedding(self):
        if self.root_embedding is None:
            from sentiment_lmx import models

            _, _, embed = models.load_default_models()
            self.root_embedding = embed.encode([self.root_sentence])[0]

    def _load_evaluators(self):
        """Lazily load the three independent evaluators for robustness mode."""
        if self._evaluators is None:
            from sentiment_lmx.crossover_analysis import (
                DEFAULT_EVALUATOR_MODELS,
                get_device,
                load_evaluators,
            )

            device = get_device()
            print("   Loading robustness evaluators...")
            self._evaluators = load_evaluators(DEFAULT_EVALUATOR_MODELS, device)
        return self._evaluators

    def _compute_fitness(self, genome: str) -> float:
        """
        Compute the fitness score used for grid insertion and replacement.

        robustness_mode=False (baseline):
            Single-model P(positive) from the optimization evaluator.

        robustness_mode=True (robustness-aware):
            Worst-case P(positive) across all three evaluators — a
            conservative lower bound that disfavours genomes that fail
            under any one classifier.
        """
        if self.robustness_mode:
            from sentiment_lmx.crossover_analysis import cross_model_metrics, score_all_models

            evaluators = self._load_evaluators()
            scores = score_all_models(genome, evaluators)
            metrics = cross_model_metrics(scores)
            return metrics["worst_posprob"]
        else:
            return compute_sentiment_score(genome, self.target_sentiment)

    def _coords_to_bin(self, sentiment, divergence):
        """Convert continuous coordinates to grid bin."""
        row = int(sentiment * self.grid_rows)
        col = int(divergence * self.grid_cols)

        row = min(self.grid_rows - 1, max(0, row))
        col = min(self.grid_cols - 1, max(0, col))

        return (row, col)

    def _insert(self, genome):
        """
        Evaluate and insert genome into grid if it improves the cell.

        The fitness used for both cell placement and replacement is
        determined by _compute_fitness(), which respects robustness_mode.
        """
        fitness = self._compute_fitness(genome)
        divergence = compute_semantic_distance(genome, self.root_embedding)

        self.num_evals += 1

        bin_key = self._coords_to_bin(fitness, divergence)

        if bin_key not in self.grid or self.grid[bin_key]["sentiment"] < fitness:
            self.grid[bin_key] = {
                "genome": genome,
                "sentiment": fitness,
                "divergence": divergence,
            }
            return True
        return False

    def _gather_population(self):
        """Get all genomes from grid."""
        return [self.grid[k]["genome"] for k in self.grid.keys()]

    def _do_crossover(self):
        """Perform one crossover operation."""
        pop = self._gather_population()

        if len(pop) < 3:
            pop = pop + [self.root_sentence] * (3 - len(pop))

        num_parents = 3
        raw_outputs = do_crossover_fast(
            pop,
            examples=num_parents,
            batch_size=self.batch_size,
            temp=self.temperature,
        )

        for raw_output in raw_outputs:
            offspring = process_output(raw_output, take_offspring=1)
            for child in offspring:
                self._insert(child.strip())

    def run(self, num_generations=50, log_every=5):
        """
        Run MAP-Elites for specified generations.
        """
        print(f"\n{'='*80}")
        print("RUNNING MAP-ELITES")
        print(f"{'='*80}\n")

        for gen in range(num_generations):
            self._do_crossover()

            if gen % log_every == 0 or gen == num_generations - 1:
                self._log_stats(gen)

        print("\n Evolution complete!")
        print(f"   Total evaluations: {self.num_evals}")
        print(f"   Final coverage: {len(self.grid)}/{self.grid_rows * self.grid_cols} cells")

    def _log_stats(self, generation):
        """Log and print statistics."""
        coverage = len(self.grid)
        total_cells = self.grid_rows * self.grid_cols
        coverage_pct = 100.0 * coverage / total_cells

        sentiments = [self.grid[k]["sentiment"] for k in self.grid.keys()]
        mean_sentiment = np.mean(sentiments)
        max_sentiment = np.max(sentiments)

        divergences = [self.grid[k]["divergence"] for k in self.grid.keys()]
        mean_divergence = np.mean(divergences)

        qd_score = np.sum(sentiments)

        log_entry = {
            "generation": generation,
            "num_evals": self.num_evals,
            "coverage": coverage,
            "coverage_pct": coverage_pct,
            "mean_sentiment": float(mean_sentiment),
            "max_sentiment": float(max_sentiment),
            "mean_divergence": float(mean_divergence),
            "qd_score": float(qd_score),
        }

        self.log.append(log_entry)

        print(
            f"Gen {generation:3d} | "
            f"Evals: {self.num_evals:5d} | "
            f"Coverage: {coverage:3d}/{total_cells:3d} ({coverage_pct:5.1f}%) | "
            f"Sentiment: {mean_sentiment:.3f} (max: {max_sentiment:.3f}) | "
            f"QD: {qd_score:.1f}"
        )

    def get_best_genomes(self, n=10):
        """Get top N genomes by sentiment score."""
        items = [(k, v["sentiment"], v["divergence"], v["genome"]) for k, v in self.grid.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    def export_results(self, filename=None):
        """Export grid and log to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mapelites_results_{timestamp}.json"

        grid_export = {}
        for (row, col), data in self.grid.items():
            key = f"{row},{col}"
            grid_export[key] = {
                "genome": data["genome"],
                "sentiment": float(data["sentiment"]),
                "divergence": float(data["divergence"]),
            }

        results = {
            "config": {
                "root_sentence": self.root_sentence,
                "target_sentiment": self.target_sentiment,
                "grid_size": (self.grid_rows, self.grid_cols),
                "temperature": self.temperature,
                "batch_size": self.batch_size,
                "robustness_mode": self.robustness_mode,
            },
            "grid": grid_export,
            "log": self.log,
            "num_evals": self.num_evals,
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f" Results exported to {filename}")
        return filename
