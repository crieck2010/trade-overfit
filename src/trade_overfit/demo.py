"""Seeded synthetic data: a true edge vs N noise strategies.

The money demo: with N=1000 noise strategies and one year of data, the best
in-sample Sharpe looks spectacular (~3) — and the DSR kills it, because under
the null of no skill the expected best-of-1000 is also ~3.
"""

from __future__ import annotations

import random

from .dsr import deflated_sharpe_ratio, expected_sharpe_under_null
from .metrics import sharpe_ratio


def demo_true_edge(n: int = 1260, seed: int = 7, mu: float = 0.0008,
                   sigma: float = 0.008) -> list[float]:
    """A genuinely skilled strategy: daily N(mu, sigma).

    Defaults give annualized Sharpe ~1.6 over 5 years with max drawdown
    inside the desk's 15% gate — the "honest winner" the desk should PASS.
    """
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def demo_noise_strategies(n_strategies: int = 1000, n: int = 252,
                          seed: int = 7, sigma: float = 0.01) -> list[list[float]]:
    """N zero-mean noise strategies — pure luck, no edge whatsoever."""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, sigma) for _ in range(n)] for _ in range(n_strategies)]


def demo_selection_bias(n_strategies: int = 1000, n: int = 252,
                        seed: int = 7, sigma: float = 0.01) -> dict:
    """Run the selection-bias demo end to end.

    Returns the trial Sharpes, the in-sample winner, its (inflated) Sharpe,
    the expected best-of-N under the null, and the winner's DSR.
    """
    strategies = demo_noise_strategies(n_strategies, n, seed, sigma)
    trial_sharpes = [sharpe_ratio(s) or 0.0 for s in strategies]
    best_idx = max(range(n_strategies), key=lambda i: trial_sharpes[i])
    best_sr = trial_sharpes[best_idx]
    sr0 = expected_sharpe_under_null(trial_sharpes)
    dsr = deflated_sharpe_ratio(best_sr, n, trial_sharpes)
    return {
        "n_strategies": n_strategies,
        "n_obs": n,
        "seed": seed,
        "trial_sharpes": trial_sharpes,
        "best_idx": best_idx,
        "best_in_sample_sharpe": best_sr,
        "expected_sharpe_under_null": sr0,
        "best_dsr": dsr,
        # "Killed" = fails the desk's standard DSR gate (0.95). The winner's
        # DSR hovers near 0.5 — a coin flip on whether any true edge exists —
        # which is exactly the point: a 3+ in-sample Sharpe that the desk
        # cannot distinguish from luck does not ship.
        "killed": dsr is not None and dsr < 0.95,
    }
