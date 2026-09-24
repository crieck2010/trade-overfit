import pytest

from trade_overfit.demo import (
    demo_noise_strategies,
    demo_selection_bias,
    demo_true_edge,
)
from trade_overfit.metrics import sharpe_ratio


def test_true_edge_is_deterministic():
    assert demo_true_edge(seed=7) == demo_true_edge(seed=7)
    assert demo_true_edge(seed=7) != demo_true_edge(seed=8)


def test_true_edge_has_real_sharpe():
    sr = sharpe_ratio(demo_true_edge(seed=7))
    assert 1.0 < sr < 2.0  # seeded series: genuine edge, deterministic value


def test_noise_count_and_shape():
    strats = demo_noise_strategies(n_strategies=50, n=100, seed=7)
    assert len(strats) == 50
    assert all(len(s) == 100 for s in strats)


def test_best_noise_looks_great_in_sample():
    demo = demo_selection_bias(seed=7)
    assert demo["best_in_sample_sharpe"] > 2.5  # spectacular — and fake


def test_dsr_kills_the_noise_winner():
    demo = demo_selection_bias(seed=7)
    assert demo["best_dsr"] is not None
    # A 3+ in-sample Sharpe reduced to a coin flip: nowhere near the 0.95 gate.
    assert demo["best_dsr"] < 0.95
    assert demo["killed"] is True


def test_expected_null_close_to_best():
    # under the null, the expected best-of-1000 ≈ the observed best
    demo = demo_selection_bias(seed=7)
    assert demo["expected_sharpe_under_null"] == pytest.approx(
        demo["best_in_sample_sharpe"], abs=0.6
    )


def test_true_edge_survives_dsr():
    from trade_overfit.dsr import dsr_from_returns
    d = dsr_from_returns(demo_true_edge(seed=7))
    assert d["dsr"] > 0.95
