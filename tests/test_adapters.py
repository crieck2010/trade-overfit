import pytest

from trade_overfit.adapters import (
    gate_research_idea,
    promotion_payload,
    returns_from_backtest,
)


def test_plain_list_passes_through():
    assert returns_from_backtest([0.01, -0.02]) == [0.01, -0.02]


def test_dict_with_returns():
    assert returns_from_backtest({"returns": [0.01]}) == [0.01]


def test_dict_with_equity_curve():
    out = returns_from_backtest({"equity_curve": [100.0, 110.0, 99.0]})
    assert out == pytest.approx([0.10, -0.10])


def test_dict_with_neither_raises_value_error():
    with pytest.raises(ValueError):
        returns_from_backtest({"sharpe": 1.5})


def test_missing_sibling_raises_helpful_error():
    # trade_backtest is not installed in this env; a non-plain object
    # triggers the lazy import path.
    with pytest.raises(RuntimeError, match="trade-backtest"):
        returns_from_backtest(object())


def test_gate_research_idea_plain_data():
    from trade_overfit.demo import demo_true_edge
    idea = {"id": "idea-1", "returns": demo_true_edge(seed=7)}
    verdict = gate_research_idea(idea)
    assert verdict["idea_id"] == "idea-1"
    assert verdict["verdict"] in ("PASS", "FAIL")


def test_promotion_payload_only_on_pass():
    assert promotion_payload({"verdict": "PASS", "evidence": {
        "dsr": 0.99, "median_oos_sharpe": 1.4, "max_drawdown": -0.05},
        "n_gates_passed": 5, "n_gates": 5})["source"] == "trade-overfit"
    assert promotion_payload({"verdict": "FAIL"}) is None
