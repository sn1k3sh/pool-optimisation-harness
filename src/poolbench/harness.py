"""
Benchmark loop. Picks initial points then asks the optimiser for batches
    until the budget runs out.

Results are returned as a dataframe with one row per step.
"""
# imports ---------------------------------------------------------------------
import concurrent.futures
import warnings
from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm.auto import tqdm

# local imports ---------------------------------------------------------------
from .optimizers import OptimizationState, Optimizer
from .pool import Pool

__all__ = ["init_idx", "run_single", "run", "summarize"]

warnings.filterwarnings("ignore", message="The balance properties of Sobol")

# initial points --------------------------------------------------------------
def init_idx(pool: Pool, n_init: int, method: str, rng: np.random.Generator) -> list:
    """
    Initial pool indices. "random", or "sobol"/"lhc" which sample the feature
    space and take the nearest unused row.
    """
    if method == "random":
        return rng.choice(pool.n, size=n_init, replace=False).tolist()
    samplers = {"sobol": qmc.Sobol, "lhc": qmc.LatinHypercube}
    if method not in samplers:
        raise ValueError(f"unknown init method '{method}', choose from {['random', *samplers]}")

    # drop constant columns, qmc.scale fails on them
    lo, hi = pool.X.min(axis=0), pool.X.max(axis=0)
    varying = hi > lo
    if not varying.any():
        return rng.choice(pool.n, size=n_init, replace=False).tolist()
    X = pool.X[:, varying]
    targets = qmc.scale(samplers[method](d=X.shape[1], seed=rng).random(n_init), lo[varying], hi[varying])

    mu, sd = X.mean(axis=0), X.std(axis=0)
    X_z, targets_z = (X - mu) / sd, (targets - mu) / sd
    available = np.ones(pool.n, dtype=bool)
    idx = []
    for t in targets_z:
        dists = np.linalg.norm(X_z - t, axis=1)
        dists[~available] = np.inf
        chosen = int(np.argmin(dists))
        idx.append(chosen)
        available[chosen] = False
    return idx

# single run ------------------------------------------------------------------
def run_single(
    pool: Pool,
    optimizer: Optimizer,
    budget: int,
    batch_size: int = 1,
    n_init: int = 5,
    init_method: str = "sobol",
    seed: int = 0,
    tol: float = 0.0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    One optimiser, one seed.

    budget : total queries including n_init
    tol    : regret within tol counts as finding the optimum
    """
    rng = np.random.default_rng(seed)
    optimizer.reset()

    first = init_idx(pool, min(n_init, budget, pool.n), init_method, rng)
    state = OptimizationState(
        pool=pool, queried_idx=list(first), observed_y=[pool.y[i].copy() for i in first], rng=rng,
    )
    notes = [None] * len(first)
    budget = min(budget, pool.n)

    pbar = tqdm(total=budget, initial=len(first), desc=f"{optimizer.name} seed {seed}",
                leave=False, disable=not show_progress)
    while len(state.queried_idx) < budget:
        this_batch = min(batch_size, budget - len(state.queried_idx))
        batch = [int(i) for i in optimizer.ask(state, batch_size=this_batch)]
        _check_batch(batch, this_batch, state)
        for i in batch:
            state.queried_idx.append(i)
            state.observed_y.append(pool.y[i].copy())
        notes.extend([state.note] * this_batch)
        pbar.update(this_batch)
    pbar.close()

    best = pool.best_so_far_curve(state.observed_y)
    regret = pool.true_best - best
    hits = np.nonzero(regret <= tol)[0]
    queries_to_best = int(hits[0]) + 1 if hits.size else None

    y_obs = state.y_observed
    df = pd.DataFrame({
        "optimizer": optimizer.name,
        "model": optimizer.model_name,
        "batch_size": batch_size,
        "seed": seed,
        "step": np.arange(1, len(best) + 1),
        "idx": state.queried_idx,
        "id": [pool.ids[i] for i in state.queried_idx],
        "best_so_far": best,
        "simple_regret": regret,
        "queries_to_best": queries_to_best,
        "note": notes,
    })
    for j, name in enumerate(pool.objective_names):
        df[name] = y_obs[:, j]
    return df

def _check_batch(batch: list, expected: int, state: OptimizationState):
    assert len(batch) == expected, f"ask() returned {len(batch)} indices, expected {expected}"
    assert len(set(batch)) == expected, "ask() returned duplicate indices"
    assert set(batch) <= set(state.unobserved_idx.tolist()), "ask() returned an already-queried or invalid index"

# benchmark -------------------------------------------------------------------
def run(
    pool: Pool,
    optimizers: Union[Optimizer, Sequence[Optimizer]],
    budget: int,
    batch_size: int = 1,
    n_init: int = 5,
    init_method: str = "sobol",
    seeds: Union[int, Sequence[int]] = 10,
    tol: float = 0.0,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    Run each optimiser over all seeds.

    seeds     : int n for range(n), or a list
    n_workers : run seeds in parallel threads, useful for LLM optimisers
    """
    if isinstance(optimizers, Optimizer):
        optimizers = [optimizers]
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    kwargs = dict(budget=budget, batch_size=batch_size, n_init=n_init, init_method=init_method, tol=tol)

    frames = []
    for optimizer in optimizers:
        def _run(seed):
            return run_single(pool, optimizer, seed=seed, show_progress=False, **kwargs)

        desc = f"{optimizer.name} seeds"
        if n_workers == 1:
            frames += [_run(seed) for seed in tqdm(seeds, desc=desc)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                frames += list(tqdm(executor.map(_run, seeds), total=len(seeds), desc=f"{desc} ({n_workers} workers)"))

    return pd.concat(frames).sort_values(["optimizer", "seed", "step"]).reset_index(drop=True)

# summary ---------------------------------------------------------------------
def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Final regret, hit rate and mean queries to the optimum per optimiser."""
    final = results.sort_values("step").groupby(["optimizer", "seed"]).tail(1)
    return (
        final.groupby("optimizer")
        .agg(final_regret_median=("simple_regret", "median"),
             final_regret_q25=("simple_regret", lambda s: s.quantile(0.25)),
             final_regret_q75=("simple_regret", lambda s: s.quantile(0.75)),
             found_rate=("queries_to_best", lambda s: s.notna().mean()),
             mean_queries_to_best=("queries_to_best", "mean"))
        .reset_index()
    )
