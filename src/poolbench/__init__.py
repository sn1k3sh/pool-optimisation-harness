"""
poolbench - benchmark optimisers on a fixed pool of candidates.

    from poolbench import Pool, RandomOptimizer, run
    pool = Pool.from_dataframe(df, feature_cols=["x1", "x2"], target_cols="y")
    results = run(pool, RandomOptimizer(), budget=50, seeds=10)
"""

from .harness import run, run_single, summarize
from .optimizers import BOOptimizer, OptimizationState, Optimizer, RandomOptimizer
from .plotting import plot_convergence
from .pool import Pool, hypervolume, pareto_efficient_mask

__version__ = "0.1.0"

__all__ = [
    "Pool",
    "Optimizer",
    "OptimizationState",
    "RandomOptimizer",
    "BOOptimizer",
    "run",
    "run_single",
    "summarize",
    "plot_convergence",
    "hypervolume",
    "pareto_efficient_mask",
]
