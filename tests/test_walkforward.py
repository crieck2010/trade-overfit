import pytest

from trade_overfit.demo import demo_noise_strategies, demo_true_edge
from trade_overfit.walkforward import walk_forward


def _edge(seed=7):
    return demo_true_edge(n=1260, seed=seed)


def test_window_counts_rolling():
    wf = walk_forward(_edge(), train_len=252, test_len=63)
    # (1260 - 252) // 63 = 16 windows
    assert wf["summary"]["n_windows"] == 16
    assert not wf["summary"]["anchored"]


def test_window_counts_anchored():
    wf = walk_forward(_edge(), train_len=252, test_len=63, anchored=True)
    assert wf["summary"]["n_windows"] == 16
    first = wf["windows"][0]
    assert first["train_start"] == 0 and first["test_start"] == 252
    second = wf["windows"][1]
    assert second["train_start"] == 0 and second["train_end"] == 315  # grew


def test_step_defaults_to_test_len():
    wf = walk_forward(_edge(), train_len=252, test_len=63)
    assert wf["summary"]["step"] == 63


def test_too_short_series_gives_no_windows():
    wf = walk_forward([0.01] * 100, train_len=252, test_len=63)
    assert wf["summary"]["n_windows"] == 0
    assert wf["summary"]["median_oos_sharpe"] is None


def test_true_edge_positive_oos():
    wf = walk_forward(_edge(), train_len=252, test_len=63)
    s = wf["summary"]
    assert s["median_oos_sharpe"] > 0.5
    assert s["oos_hit_rate"] > 0.7


def test_noise_degrades():
    noise = demo_noise_strategies(1, 1260, seed=3)[0]
    wf = walk_forward(noise, train_len=252, test_len=63)
    s = wf["summary"]
    # pure noise: median OOS Sharpe hovers near zero
    assert abs(s["median_oos_sharpe"]) < 1.0
    assert set(wf["windows"][0]) == {
        "train_start", "train_end", "test_start", "test_end",
        "is_sharpe", "oos_sharpe", "oos_return", "oos_max_drawdown",
    }


def test_summary_has_degradation():
    wf = walk_forward(_edge(), train_len=252, test_len=63)
    d = wf["summary"]["degradation"]
    assert d is not None  # mean IS minus mean OOS
