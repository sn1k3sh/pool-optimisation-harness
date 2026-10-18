"""
Example custom optimiser. The LLM is shown the observed points and a random
    sample of candidates and picks the next batch.

Uses a local Ollama model:

    ollama pull llama3.1
    python examples/llm_optimizer.py
"""
# imports ---------------------------------------------------------------------
import re

import numpy as np
import requests

from poolbench import Optimizer, Pool, RandomOptimizer, plot_convergence, run, summarize

# optimiser -------------------------------------------------------------------
class LLMOptimizer(Optimizer):
    name = "llm"

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434",
                 n_candidates: int = 50):
        self.model = model
        self.host = host
        self.model_name = model
        self.n_candidates = n_candidates

    def _chat(self, prompt):
        r = requests.post(f"{self.host}/api/chat", json={
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7},
        }, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def ask(self, state, batch_size):
        unobserved = state.unobserved_idx
        shown = state.rng.choice(unobserved, size=min(self.n_candidates, len(unobserved)), replace=False)

        reply = self._chat(self._prompt(state, shown, batch_size))

        # take every number inside brackets, models write [3, 17] or [3], [17]
        # fill with random picks if the reply is short or invalid
        picked = []
        for group in re.findall(r"\[([\d,\s]*)\]", reply):
            for i in map(int, re.findall(r"\d+", group)):
                if i in shown and i not in picked:
                    picked.append(i)
        picked = picked[:batch_size]
        rest = [i for i in unobserved if i not in picked]
        picked += state.rng.choice(rest, size=batch_size - len(picked), replace=False).tolist()
        state.note = reply
        return picked

    def _prompt(self, state, shown, batch_size):
        pool = state.pool
        row = lambda i: ", ".join(f"{n}={v:.3g}" for n, v in zip(pool.feature_names, pool.X[i]))
        observed = "\n".join(
            f"{row(i)} -> " + ", ".join(f"{n}={v:.4g}" for n, v in zip(pool.objective_names, y))
            for i, y in zip(state.queried_idx, state.observed_y)
        )
        candidates = "\n".join(f"[{i}] {row(i)}" for i in shown)
        return (
            f"We are maximizing {', '.join(pool.objective_names)}.\n\n"
            f"Observed so far:\n{observed}\n\n"
            f"Candidates:\n{candidates}\n\n"
            f"Pick the {batch_size} candidates most likely to improve the objective. "
            f"Answer with a JSON list of their bracketed ids only, e.g. [3, 17]."
        )

# test pool -------------------------------------------------------------------
def branin_pool(n_grid: int = 40) -> Pool:
    """Branin on a grid, negated so higher = better."""
    x1, x2 = np.meshgrid(np.linspace(-5, 10, n_grid), np.linspace(0, 15, n_grid))
    x1, x2 = x1.ravel(), x2.ravel()
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    f = (x2 - b * x1 ** 2 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10
    return Pool(X=np.column_stack([x1, x2]), y=-f, feature_names=["x1", "x2"], objective_names=["neg_branin"])

# main ------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pool = branin_pool()
    results = run(pool, [RandomOptimizer(), LLMOptimizer()], budget=30, batch_size=3, seeds=3, n_workers=3)
    print(summarize(results))
    plot_convergence(results, true_value=pool.true_max)
    plt.show()
