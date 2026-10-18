"""
Pool (candidate features and respective objective values) to optimise over.

All objectives are maximised. Multi-objective pools are scored by hypervolume.
"""
# imports ---------------------------------------------------------------------
import dataclasses
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["Pool", "pareto_efficient_mask", "hypervolume", "running_hypervolume"]

# pareto ----------------------------------------------------------------------
def pareto_efficient_mask(y: np.ndarray) -> np.ndarray:
    """True for rows of y that no other row dominates."""
    n = y.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominates_i = np.all(y >= y[i], axis=1) & np.any(y > y[i], axis=1)
        if np.any(dominates_i):
            efficient[i] = False
    return efficient

# hypervolume -----------------------------------------------------------------
def hypervolume(y: np.ndarray, reference_point: np.ndarray) -> float:
    """Hypervolume of y above reference_point (dimension sweep)."""
    if y.shape[0] == 0:
        return 0.0
    points = y[pareto_efficient_mask(y)]
    return _hv_recursive(points, reference_point)

def _hv_recursive(points: np.ndarray, ref: np.ndarray) -> float:
    n, d = points.shape
    if n == 0:
        return 0.0
    if d == 1:
        return float(np.max(points[:, 0]) - ref[0])

    points = points[np.argsort(-points[:, -1])]
    hv = 0.0
    for i in range(n):
        next_level = points[i + 1, -1] if i + 1 < n else ref[-1]
        height = points[i, -1] - next_level
        if height <= 0:
            continue
        hv += _hv_recursive(points[: i + 1, :-1], ref[:-1]) * height
    return hv

def running_hypervolume(y: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    return np.array([hypervolume(y[: i + 1], reference_point) for i in range(y.shape[0])])

# pool ------------------------------------------------------------------------
@dataclasses.dataclass
class Pool:
    """Candidate features X and objectives y."""

    X: np.ndarray                  # (n, n_features)
    y: np.ndarray                  # (n, n_objectives)
    ids: Optional[list] = None
    feature_names: Optional[list] = None
    objective_names: Optional[list] = None
    feature_descriptions: Optional[dict] = None
    objective_description: Optional[str] = None

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        y = np.asarray(self.y, dtype=float)
        self.y = y.reshape(-1, 1) if y.ndim == 1 else y
        if self.ids is None:
            self.ids = [f"row_{i}" for i in range(self.X.shape[0])]
        assert self.X.shape[0] == self.y.shape[0] == len(self.ids), "X, y and ids must have the same length"
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
        if self.objective_names is None:
            self.objective_names = [f"objective_{i}" for i in range(self.y.shape[1])]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_cols: Union[str, Sequence[str]],
        id_col: Optional[str] = None,
        **metadata,
    ) -> "Pool":
        """
        Build a pool from a dataframe. Rows with NaNs are dropped.

        feature_cols : numeric columns (categorical columns should be one-hot encoded first)
        target_cols  : one column or a list for multi-objective
        id_col       : optional label column
        """
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        clean = df.dropna(subset=list(feature_cols) + list(target_cols)).reset_index(drop=True)
        ids = clean[id_col].astype(str).tolist() if id_col is not None else None
        return cls(
            X=clean[list(feature_cols)].to_numpy(dtype=float),
            y=clean[list(target_cols)].to_numpy(dtype=float),
            ids=ids, feature_names=list(feature_cols), objective_names=list(target_cols),
            **metadata,
        )

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def n_objectives(self) -> int:
        return self.y.shape[1]

    @property
    def is_multi_objective(self) -> bool:
        return self.n_objectives > 1

    # single objective --------------------------------------------------------
    @property
    def true_max(self) -> float:
        assert not self.is_multi_objective, "true_max is single-objective only, use true_hypervolume"
        return float(np.max(self.y[:, 0]))

    @property
    def argmax(self) -> int:
        assert not self.is_multi_objective, "argmax is single-objective only, use pareto_front_idx"
        return int(np.argmax(self.y[:, 0]))

    # multi objective ---------------------------------------------------------
    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Scale each objective to [0, 1] using the pool's min and max."""
        y_min, y_max = self.y.min(axis=0), self.y.max(axis=0)
        return (np.asarray(y, dtype=float) - y_min) / np.maximum(y_max - y_min, 1e-12)

    @property
    def reference_point(self) -> np.ndarray:
        return np.zeros(self.n_objectives)

    @property
    def pareto_front_idx(self) -> np.ndarray:
        return np.nonzero(pareto_efficient_mask(self.y))[0]

    @property
    def true_hypervolume(self) -> float:
        return hypervolume(self.normalize_y(self.y[self.pareto_front_idx]), self.reference_point)

    # scoring -----------------------------------------------------------------
    def best_so_far_curve(self, observed_y: np.ndarray) -> np.ndarray:
        """Running max, or running hypervolume if multi-objective."""
        y_obs = np.asarray(observed_y, dtype=float).reshape(-1, self.n_objectives)
        if self.is_multi_objective:
            return running_hypervolume(self.normalize_y(y_obs), self.reference_point)
        return np.maximum.accumulate(y_obs[:, 0])

    @property
    def true_best(self) -> float:
        return self.true_hypervolume if self.is_multi_objective else self.true_max
