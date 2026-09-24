import pytest

from trade_overfit.dsr import (
    deflated_sharpe_ratio,
    dsr_from_returns,
    expected_sharpe_under_null,
    psr,
)


def test_psr_is_probability():
    p = psr(1.5, 0.0, 252)
    assert 0.0 <= p <= 1.0
    assert p > 0.99  # strong Sharpe, long history -> near-certain


def test_psr_needs_two_observations():
    assert psr(1.5, 0.0, 1) is None
    assert psr(1.5, 0.0, 0) is None


def test_psr_negative_sharpe_is_tiny():
    assert psr(-1.0, 0.0, 252) < 0.01


def test_expected_null_single_trial_is_zero():
    assert expected_sharpe_under_null([1.5]) == 0.0
    assert expected_sharpe_under_null([]) == 0.0


def test_expected_null_zero_variance_is_zero():
    assert expected_sharpe_under_null([1.0, 1.0, 1.0]) == 0.0


def test_dsr_n1_equals_psr_zero():
    # With a single trial there is no selection bias: DSR == PSR(0).
    sr, n, skew, ke = 1.2, 500, 0.1, 0.5
    assert deflated_sharpe_ratio(sr, n, [sr], skew, ke) == pytest.approx(
        psr(sr, 0.0, n, skew, ke)
    )


def test_dsr_falls_as_n_rises():
    # wide trial distribution: more trials -> harsher deflation of the same hat
    trials5 = [-1.0, -0.5, 0.0, 0.5, 1.0]
    d1 = deflated_sharpe_ratio(1.0, 60, [1.0])
    d5 = deflated_sharpe_ratio(1.0, 60, trials5)
    assert d5 < d1


def test_dsr_rises_with_sample_size():
    trials = [1.0]
    d_short = deflated_sharpe_ratio(1.0, 60, trials)
    d_long = deflated_sharpe_ratio(1.0, 2520, trials)
    assert d_long > d_short


def test_dsr_short_history_is_none():
    assert deflated_sharpe_ratio(1.5, 1, [1.5]) is None


def test_dsr_from_returns_wrapper():
    rets = [0.001 + (0.001 if i % 2 else -0.001) for i in range(500)]
    d = dsr_from_returns(rets, [1.0])
    assert d["n_trials"] == 1
    assert d["n_obs"] == 500
    assert 0.0 <= d["dsr"] <= 1.0
    # default: single trial -> null expectation is zero
    assert d["expected_sharpe_under_null"] == 0.0


def test_dsr_from_returns_defaults_to_n1():
    rets = [0.002] * 300 + [-0.001] * 200
    d = dsr_from_returns(rets)
    assert d["n_trials"] == 1
