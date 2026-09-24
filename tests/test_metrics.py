import math

import pytest

from trade_overfit.metrics import (
    annualized_return,
    calmar_ratio,
    cost_haircut,
    excess_returns,
    max_drawdown,
    per_period_rf,
    performance_summary,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)


def test_per_period_rf_conversion():
    # (1 + r)^(1/252) - 1 for 5% annual
    assert per_period_rf(0.05) == pytest.approx((1.05) ** (1 / 252) - 1)


def test_excess_returns_subtracts_rf():
    xs = excess_returns([0.01, 0.02], rf_annual=0.0)
    assert xs == [0.01, 0.02]
    xs = excess_returns([0.01], rf_annual=0.05)
    assert xs[0] == pytest.approx(0.01 - per_period_rf(0.05))


def test_sharpe_known_value():
    # constant excess return: mean/std*sqrt(252)
    rets = [0.001] * 100 + [0.002] * 100
    sr = sharpe_ratio(rets)
    import statistics
    expected = statistics.mean(rets) / statistics.stdev(rets) * math.sqrt(252)
    assert sr == pytest.approx(expected)


def test_sharpe_undefined_cases():
    assert sharpe_ratio([]) is None
    assert sharpe_ratio([0.01]) is None
    assert sharpe_ratio([0.01] * 50) is None  # zero volatility


def test_sortino_only_downside_counts():
    # upside-only series: downside deviation tiny -> large sortino
    rets = [0.01] * 100
    assert sortino_ratio(rets) is None or sortino_ratio(rets) > 0
    # symmetric noise: sortino defined
    rets = [0.01, -0.01] * 50
    s = sortino_ratio(rets)
    assert s is not None and math.isfinite(s)


def test_max_drawdown_values():
    dd = max_drawdown([0.1, -0.5, 0.1])  # 1.1 -> 0.55 -> 0.605
    assert dd["max_drawdown"] == pytest.approx(-0.5, abs=1e-9)
    assert dd["duration_bars"] >= 1
    dd2 = max_drawdown([0.01] * 20)
    assert dd2["max_drawdown"] == pytest.approx(0.0)


def test_total_and_annualized_return():
    assert total_return([0.1, 0.1]) == pytest.approx(0.21)
    assert total_return([]) is None
    assert annualized_return([0.1, 0.1]) == pytest.approx(1.21 ** (252 / 2) - 1)


def test_calmar_ratio():
    rets = [0.01] * 252  # no drawdown -> None
    assert calmar_ratio(rets) is None
    rets = [0.02] * 100 + [-0.01] * 50
    c = calmar_ratio(rets)
    assert c is not None and c > 0


def test_profit_factor_and_win_rate():
    assert profit_factor([0.02, 0.02, -0.01]) == pytest.approx(4.0)
    assert profit_factor([0.01, 0.02]) == math.inf  # no losers
    assert profit_factor([]) is None
    assert win_rate([0.01, -0.01, 0.02]) == pytest.approx(2 / 3)
    assert win_rate([]) is None


def test_cost_haircut_reduces_returns():
    rets = [0.001, 0.002] * 126
    net = cost_haircut(rets, cost_bps_per_trade=5.0, trades_per_year=252.0)
    drag = (5.0 / 1e4) * 252.0 / 252
    assert net[0] == pytest.approx(0.001 - drag)
    assert sharpe_ratio(net) < sharpe_ratio(rets)


def test_performance_summary_keys():
    s = performance_summary([0.01, -0.005, 0.02] * 100)
    for key in ("sharpe", "sortino", "max_drawdown", "calmar", "profit_factor",
                "win_rate", "skewness", "excess_kurtosis", "annualized_return",
                "total_return", "n_bars", "volatility_ann",
                "max_drawdown_duration_bars"):
        assert key in s, key
    assert s["n_bars"] == 300
