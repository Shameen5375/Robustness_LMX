"""Visualization of the MAP-Elites behavioral atlas."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def visualize_atlas(me, save_path: str | None = None) -> None:
    """
    Render the MAP-Elites grid as a 2D heatmap.

    X-axis: semantic divergence (grid columns)
    Y-axis: sentiment / fitness (grid rows)
    Color:  fitness value stored in each occupied cell

    Args:
        me:        A MapElites2D instance after running evolution.
        save_path: If provided, save the figure to this path instead of
                   displaying it.
    """
    grid_data = np.full((me.grid_rows, me.grid_cols), np.nan)

    for (row, col), data in me.grid.items():
        grid_data[row, col] = data["sentiment"]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        grid_data,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=[0, 1, 0, 1],
    )

    plt.colorbar(im, ax=ax, label="Fitness (sentiment / worst-case P(pos))")

    coverage = len(me.grid)
    total = me.grid_rows * me.grid_cols
    mode_label = "robustness" if me.robustness_mode else "baseline"

    ax.set_xlabel("Semantic Divergence")
    ax.set_ylabel("Sentiment Strength")
    ax.set_title(
        f"MAP-Elites Atlas [{mode_label}] — "
        f"coverage {coverage}/{total} ({100*coverage/total:.1f}%)"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Atlas saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
