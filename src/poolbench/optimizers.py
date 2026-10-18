"""
Optimisers used by the harness, with random and BO baselines.

Each optimiser is given the current state and returns which pool rows to query
    next. Thus, adding a new optimiser only requires subclassing Optimizer and
    writing an ask() method.
"""
# imports ---------------------------------------------------------------------
import dataclasses
import warnings
from typing import Optional

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.utils.warnings import NumericalWarning

# local imports ---------------------------------------------------------------
from .pool import Pool, hypervolume

__all__ = ["OptimizationState", "Optimizer", "RandomOptimizer", "BOOptimizer"]

warnings.filterwarnings("ignore", category=NumericalWarning)   # jitter warnings during GP fit

# state -----------------------------------------------------------------------
@dataclasses.dataclass
class OptimizationState:
    """What the optimiser can see during one seed."""

    pool: Pool
    queried_idx: list
    observed_y: list
    rng: np.random.Generator
    note: Optional[str] = None     # optional text saved with each step

    @property
    def unobserved_idx(self) -> np.ndarray:
        mask = np.ones(self.pool.n, dtype=bool)
        mask[self.queried_idx] = False
        return np.nonzero(mask)[0]

    @property
    def X_observed(self) -> np.ndarray:
        return self.pool.X[self.queried_idx]

    @property
    def y_observed(self) -> np.ndarray:
        return np.array(self.observed_y).reshape(-1, self.pool.n_objectives)

    @property
    def best_so_far(self) -> float:
        if not self.observed_y:
            return -np.inf
        if not self.pool.is_multi_objective:
            return float(np.max(self.y_observed[:, 0]))
        return hypervolume(self.pool.normalize_y(self.y_observed), self.pool.reference_point)

# optimiser -------------------------------------------------------------------
class Optimizer:
    name: str = "base"
    model_name: Optional[str] = None

    def ask(self, state: OptimizationState, batch_size: int) -> list:
        """Return batch_size unobserved pool indices."""
        raise NotImplementedError

    def reset(self):
        pass

# random ----------------------------------------------------------------------
class RandomOptimizer(Optimizer):
    name = "random"

    def ask(self, state: OptimizationState, batch_size: int) -> list:
        return state.rng.choice(state.unobserved_idx, size=batch_size, replace=False).tolist()

# bo --------------------------------------------------------------------------
def _fit_gp(X_train: torch.Tensor, Y_train: torch.Tensor, bounds: torch.Tensor):

    model = SingleTaskGP(
        X_train, Y_train,
        input_transform=Normalize(d=X_train.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=Y_train.shape[-1]),
    ).to(X_train)

    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

    return model

def _match_to_idx(candidates: torch.Tensor, choices: torch.Tensor, unobserved_idx: np.ndarray) -> list:
    """Convert the rows returned by optimize_acqf_discrete back to pool indices."""

    remaining = list(range(choices.shape[0]))
    chosen = []

    for row in candidates:
        match = next(i for i in remaining if torch.equal(choices[i], row))
        remaining.remove(match)
        chosen.append(int(unobserved_idx[match]))

    return chosen

class BOOptimizer(Optimizer):
    """qLogNEI for single objective, qLogNEHVI for multi-objective"""

    name = "bo"

    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _tensor(self, a) -> torch.Tensor:
        return torch.tensor(np.asarray(a), dtype=torch.double, device=self.device)

    def ask(self, state: OptimizationState, batch_size: int) -> list:
        pool = state.pool
        X_train = self._tensor(state.X_observed)
        y = state.y_observed
        Y_train = self._tensor(pool.normalize_y(y) if pool.is_multi_objective else y)
        bounds = self._tensor(np.stack([pool.X.min(axis=0), pool.X.max(axis=0)]))

        model = _fit_gp(X_train, Y_train, bounds)
        if pool.is_multi_objective:
            # ref point from observed points only
            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=Y_train.min(dim=0).values, X_baseline=X_train, prune_baseline=True,
            )
        else:
            acqf = qLogNoisyExpectedImprovement(model=model, X_baseline=X_train, prune_baseline=True)

        unobserved_idx = state.unobserved_idx
        choices = self._tensor(pool.X[unobserved_idx])
        candidates, _ = optimize_acqf_discrete(acq_function=acqf, q=batch_size, choices=choices)
        return _match_to_idx(candidates, choices, unobserved_idx)
