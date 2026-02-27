"""Evolution curve visualization for MAP-Elites runs."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_evolution_curve(me, save_path: str | None = None) -> None:
    """
    Plot coverage and QD score over generations from a finished MAP-Elites run.

    Top panel:  grid coverage (%) vs generation
    Bottom panel: QD score (sum of fitness values) vs generation

    Args:
        me:        A MapElites2D instance after running evolution.
        save_path: If provided, save the figure to this path.
    """
    if not me.log:
        print("   No log data to plot.")
        return

    generations = [entry["generation"] for entry in me.log]
    coverage_pct = [entry["coverage_pct"] for entry in me.log]
    qd_scores = [entry["qd_score"] for entry in me.log]
    mean_sentiments = [entry["mean_sentiment"] for entry in me.log]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # Coverage
    axes[0].plot(generations, coverage_pct, color="steelblue", linewidth=2)
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Evolution Curves")

    # QD score
    axes[1].plot(generations, qd_scores, color="darkorange", linewidth=2)
    axes[1].set_ylabel("QD Score")
    axes[1].grid(True, alpha=0.3)

    # Mean fitness
    axes[2].plot(generations, mean_sentiments, color="seagreen", linewidth=2)
    axes[2].set_ylabel("Mean Fitness")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    mode_label = "robustness" if me.robustness_mode else "baseline"
    fig.suptitle(f"MAP-Elites Evolution [{mode_label}]", fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Evolution curve saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
