"""
Tests for pool ground truth, pareto mask and hypervolume.
"""
# imports ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from poolbench import Pool, hypervolume, pareto_efficient_mask

# ground truth ----------------------------------------------------------------
def test_from_dataframe_drops_nans_and_keeps_names():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [0, 1, 1, 0], "t": [1, np.nan, 3, 4], "name": list("wxyz")})
    pool = Pool.from_dataframe(df, ["a", "b"], "t", id_col="name")
    assert pool.n == 2
    assert pool.ids == ["w", "z"]
    assert pool.feature_names == ["a", "b"]
    assert pool.objective_names == ["t"]

# pareto / hypervolume --------------------------------------------------------
def test_pareto_mask():
    y = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.4, 0.4], [1, 0]])
    assert pareto_efficient_mask(y).tolist() == [True, True, True, False, True]

def test_hypervolume_2d_by_hand():
    y = np.array([[1.0, 0.2], [0.2, 1.0], [0.6, 0.6]])
    expected = 1.0 * 0.2 + 0.6 * (0.6 - 0.2) + 0.2 * (1.0 - 0.6)
    assert hypervolume(y, np.zeros(2)) == pytest.approx(expected)

def test_true_hypervolume_is_normalized():
    pool = Pool(X=np.zeros((3, 1)), y=[[0, 100], [10, 0], [5, 50]])
    # normalised to [0,1], [1,0], [0.5,0.5]
    assert pool.true_hypervolume == pytest.approx(0.25)
    assert sorted(pool.pareto_front_idx.tolist()) == [0, 1, 2]
