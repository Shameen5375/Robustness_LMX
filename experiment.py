"""
Experiment orchestration helpers for running MAP-Elites variants.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from sentiment_lmx.algorithm import MapElites2D
from sentiment_lmx.metrics.atlas import visualize_atlas
from sentiment_lmx.metrics.evolution import plot_evolution_curve


def run_baseline_experiment(
    root_sentence,
    target_sentiment="positive",
    num_generations=50,
    temperature=0.7,
    grid_size=(20, 20),
    batch_size=8,
    output_dir="./results",
):
    """
    Run complete baseline experiment.
    """
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*80}")
    print("BASELINE EXPERIMENT")
    print(f"{'='*80}")
    print(f"Root: {root_sentence}")
    print(f"Target sentiment: {target_sentiment}")
    print(f"Temperature: {temperature}")
    print(f"Generations: {num_generations}")
    print(f"Grid: {grid_size[0]}{grid_size[1]}")
    print(f"{'='*80}\n")

    me = MapElites2D(
        root_sentence=root_sentence,
        target_sentiment=target_sentiment,
        grid_size=grid_size,
        temperature=temperature,
        batch_size=batch_size,
    )

    me.run(num_generations=num_generations)

    results_file = f"{output_dir}/results_{timestamp}.json"
    me.export_results(results_file)

    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    atlas_file = f"{output_dir}/atlas_{timestamp}.png"
    visualize_atlas(me, save_path=atlas_file)

    curve_file = f"{output_dir}/evolution_{timestamp}.png"
    plot_evolution_curve(me, save_path=curve_file)

    print(f"\n{'='*80}")
    print("TOP 10 EVOLVED SENTENCES")
    print(f"{'='*80}\n")

    best = me.get_best_genomes(n=10)
    for i, (coords, sentiment, divergence, genome) in enumerate(best, 1):
        print(f"{i:2d}. [{sentiment:.3f}, {divergence:.3f}] {genome}")

    print(f"\n BASELINE EXPERIMENT COMPLETE!")
    print(f"   Results: {results_file}")
    print(f"   Atlas: {atlas_file}")
    print(f"   Evolution: {curve_file}")

    return me


def _print_config(label, root_sentence, temperature, generations, grid_size, batch_size):
    print("\n" + "=" * 80)
    print(f"  {label.upper()} PRESSURE EXPERIMENT (temp={temperature})")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Root: '{root_sentence}'")
    print(f"  Temperature: {temperature}")
    print(f"  Generations: {generations}")
    print(f"  Grid: {grid_size[0]}{grid_size[1]}")
    print(f"  Batch size: {batch_size}")
    print("=" * 80 + "\n")


def run_pressure_experiment(
    label,
    root_sentence,
    target_sentiment="positive",
    temperature=0.7,
    num_generations=50,
    grid_size=(20, 20),
    batch_size=8,
    log_every=5,
    results_dir="./results",
    save_prefix=None,
):
    """Shared routine used by both low and high pressure experiments."""
    _print_config(label, root_sentence, temperature, num_generations, grid_size, batch_size)

    exp = MapElites2D(
        root_sentence=root_sentence,
        target_sentiment=target_sentiment,
        grid_size=grid_size,
        temperature=temperature,
        batch_size=batch_size,
    )

    print(" Starting evolution...")
    exp.run(num_generations=num_generations, log_every=log_every)

    Path(results_dir).mkdir(exist_ok=True, parents=True)
    prefix = save_prefix or f"{label.lower()}_pressure"
    json_path = Path(results_dir) / f"{prefix}.json"
    print("\n Saving results...")
    exp.export_results(str(json_path))

    print("\n Creating visualizations...")
    visualize_atlas(exp, save_path=str(Path(results_dir) / f"atlas_{prefix}.png"))
    plot_evolution_curve(exp, save_path=str(Path(results_dir) / f"evolution_{prefix}.png"))

    print("\n" + "=" * 80)
    print(f" TOP 10 EVOLVED SENTENCES ({label.upper()} PRESSURE)")
    print("=" * 80 + "\n")

    best = exp.get_best_genomes(n=10)
    for i, (coords, sentiment, divergence, genome) in enumerate(best, 1):
        print(f"{i:2d}. [S:{sentiment:.3f}, D:{divergence:.3f}] {genome}")

    return exp


def summarize_experiment(exp, label):
    """Print summary statistics for a finished experiment."""
    print("\n" + "=" * 80)
    print(f" {label.upper()} PRESSURE EXPERIMENT RESULTS")
    print("=" * 80)

    print(f"\n Evolution Complete!")
    print(f"   Total evaluations: {exp.num_evals}")
    print(f"   Coverage: {len(exp.grid)}/400 ({100*len(exp.grid)/400:.1f}%)")

    sentiments = [exp.grid[k]["sentiment"] for k in exp.grid.keys()]
    divergences = [exp.grid[k]["divergence"] for k in exp.grid.keys()]

    print(f"\n Sentiment Stats:")
    print(f"   Mean: {np.mean(sentiments):.3f}")
    print(f"   Max:  {np.max(sentiments):.3f}")
    print(f"   Min:  {np.min(sentiments):.3f}")
    print(f"   Std:  {np.std(sentiments):.3f}")

    print(f"\n  Divergence Stats:")
    print(f"   Mean: {np.mean(divergences):.3f}")
    print(f"   Max:  {np.max(divergences):.3f}")
    print(f"   Min:  {np.min(divergences):.3f}")

    print("\n" + "=" * 80)
    print(" TOP 10 EVOLVED SENTENCES")
    print("=" * 80 + "\n")

    best = exp.get_best_genomes(n=10)
    for i, (coords, sentiment, divergence, genome) in enumerate(best, 1):
        print(f"{i:2d}. [S:{sentiment:.3f}, D:{divergence:.3f}]")
        print(f"    \"{genome}\"")
        print()

    print("=" * 80)
    print(" ROOT SENTENCE (for comparison)")
    print("=" * 80)
    print(f"   \"{exp.root_sentence}\"")
    print(f"   Sentiment: {0.391:.3f}")
    print(f"   Divergence: 0.000")
    print("\n" + "=" * 80)
