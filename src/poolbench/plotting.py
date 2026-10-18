"""
Convergence plots.
"""
# imports ---------------------------------------------------------------------
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

__all__ = ["plot_convergence"]

# convergence -----------------------------------------------------------------
def plot_convergence(
    results: pd.DataFrame,
    y_col: str = "best_so_far",
    true_value: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Median and IQR over seeds for each optimiser.

    y_col      : "best_so_far" or "simple_regret"
    true_value : draws a dashed line at the optimum
    """
    df = results.copy()
    for col in ("model", "batch_size"):
        if col not in df.columns:
            df[col] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    for (optimizer, model, batch_size), group in df.groupby(["optimizer", "model", "batch_size"], dropna=False):
        q = group.groupby("step")[y_col].quantile([0.25, 0.5, 0.75]).unstack()
        label = optimizer
        if not pd.isna(model):
            label += f" ({model})"
        if not pd.isna(batch_size) and df["batch_size"].nunique() > 1:
            label += f" [bs={int(batch_size)}]"
        line, = ax.plot(q.index, q[0.5], linewidth=2, label=label)
        ax.fill_between(q.index, q[0.25], q[0.75], color=line.get_color(), alpha=0.15, linewidth=0)
    if true_value is not None:
        ax.axhline(true_value, color="black", linestyle="--", linewidth=1, label="true optimum")

    ax.set_xlabel("queries")
    ax.set_ylabel(y_col)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="gray", alpha=0.15))
    labels.append("IQR (25th-75th pct)")
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    return fig
