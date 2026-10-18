"""
Tests for the benchmark loop.
"""
# imports ---------------------------------------------------------------------
import numpy as np
import pytest

from poolbench import BOOptimizer, Optimizer, Pool, RandomOptimizer, run, run_single, summarize

# fixtures --------------------------------------------------------------------
@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(60, 3))
    return Pool(X=X, y=-np.sum((X - 0.5) ** 2, axis=1))

@pytest.fixture
def mo_pool():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(60, 2))
    return Pool(X=X, y=np.column_stack([X[:, 0], 1 - X[:, 0] + 0.1 * X[:, 1]]))

# harness loop ----------------------------------------------------------------
@pytest.mark.parametrize("init_method", ["random", "sobol", "lhc"])
def test_run_single_no_requeries(pool, init_method):
    df = run_single(pool, RandomOptimizer(), budget=20, batch_size=3, n_init=5,
                    init_method=init_method, show_progress=False)
    assert len(df) == 20
    assert df["idx"].is_unique
    assert (np.diff(df["best_so_far"]) >= 0).all()
    assert np.allclose(df["simple_regret"], pool.true_max - df["best_so_far"])

def test_budget_larger_than_pool_exhausts_pool(pool):
    df = run_single(pool, RandomOptimizer(), budget=500, batch_size=7, show_progress=False)
    assert len(df) == pool.n
    assert df["simple_regret"].iloc[-1] == 0
    assert df["queries_to_best"].iloc[0] >= 1

def test_concurrent_matches_serial(pool):
    serial = run(pool, RandomOptimizer(), budget=15, seeds=4)
    threaded = run(pool, RandomOptimizer(), budget=15, seeds=4, n_workers=2)
    assert serial["idx"].tolist() == threaded["idx"].tolist()

def test_multi_objective_curves(mo_pool):
    df = run(mo_pool, RandomOptimizer(), budget=mo_pool.n, seeds=1)
    assert df["simple_regret"].iloc[-1] == pytest.approx(0)
    assert {"objective_0", "objective_1"} <= set(df.columns)
    assert set(summarize(df)["optimizer"]) == {"random"}

def test_bad_optimizer_is_rejected(pool):
    class Repeats(Optimizer):
        def ask(self, state, batch_size):
            return [state.queried_idx[0]] * batch_size

    with pytest.raises(AssertionError):
        run_single(pool, Repeats(), budget=10, show_progress=False)

# bo baseline -----------------------------------------------------------------
def test_bo_runs(pool, mo_pool):
    for p in (pool, mo_pool):
        df = run_single(p, BOOptimizer(device="cpu"), budget=10, batch_size=2, n_init=4, show_progress=False)
        assert df["idx"].is_unique and len(df) == 10
