import pytest

from trade_overfit.gates import (
    DEFAULT_GATES,
    PRESETS,
    Gate,
    evaluate_gates,
    preset_gates,
    validate,
)


def _good_evidence():
    return {
        "dsr": 0.98,
        "median_oos_sharpe": 1.4,
        "max_drawdown": -0.08,
        "worst_regime_sharpe": 0.4,
        "excess_return_vs_benchmark_after_costs": 0.05,
    }


def test_all_gates_pass_on_good_evidence():
    results = evaluate_gates(_good_evidence())
    assert all(g["passed"] for g in results)
    assert len(results) == len(DEFAULT_GATES)


def test_one_failure_fails():
    ev = _good_evidence()
    ev["dsr"] = 0.5
    results = evaluate_gates(ev)
    failed = [g for g in results if not g["passed"]]
    assert len(failed) == 1 and failed[0]["name"] == "deflated_sharpe"


def test_missing_evidence_fails():
    # missing data never passes — suite convention
    results = evaluate_gates({})
    assert not any(g["passed"] for g in results)
    assert all(g["value"] is None for g in results)


def test_none_evidence_fails():
    ev = _good_evidence()
    ev["median_oos_sharpe"] = None
    results = evaluate_gates(ev)
    assert [g for g in results if g["name"] == "oos_sharpe"][0]["passed"] is False


def test_custom_threshold_flips_result():
    ev = _good_evidence()
    gates = [Gate("dsr_gate", "dsr", "gte", 0.99, "stricter")]
    assert evaluate_gates(ev, gates)[0]["passed"] is False
    gates = [Gate("dsr_gate", "dsr", "gte", 0.90, "looser")]
    assert evaluate_gates(ev, gates)[0]["passed"] is True


def test_bad_op_rejected():
    with pytest.raises(ValueError):
        Gate("x", "dsr", "approx", 0.9)


def test_presets_exist_and_differ():
    assert set(PRESETS) == {"standard", "strict", "lenient"}
    strict = {g.name: g.threshold for g in preset_gates("strict")}
    std = {g.name: g.threshold for g in preset_gates("standard")}
    assert strict["deflated_sharpe"] > std["deflated_sharpe"]
    with pytest.raises(ValueError):
        preset_gates("nope")


def test_validate_true_edge_passes():
    from trade_overfit.demo import demo_true_edge
    verdict = validate(demo_true_edge(seed=7))
    assert verdict["verdict"] == "PASS"
    assert verdict["n_gates_passed"] == verdict["n_gates"]
    assert verdict["evidence"]["dsr"] > 0.95


def test_validate_noise_winner_fails():
    from trade_overfit.demo import demo_noise_strategies, demo_selection_bias
    demo = demo_selection_bias(seed=7)
    winner = demo_noise_strategies(
        demo["n_strategies"], demo["n_obs"], 7)[demo["best_idx"]]
    verdict = validate(winner, trial_sharpes=demo["trial_sharpes"])
    assert verdict["verdict"] == "FAIL"
    dsr_gate = [g for g in verdict["gates"] if g["name"] == "deflated_sharpe"][0]
    assert dsr_gate["passed"] is False


def test_validate_is_json_serializable():
    import json
    from trade_overfit.demo import demo_true_edge
    verdict = validate(demo_true_edge(n=400, seed=7))
    json.dumps(verdict)  # must not raise


def test_validate_with_benchmark_and_costs():
    from trade_overfit.demo import demo_true_edge
    rets = demo_true_edge(seed=7)
    bench = [0.0002] * len(rets)
    verdict = validate(rets, benchmark=bench, cost_bps_per_trade=2.0)
    assert verdict["evidence"]["excess_return_vs_benchmark_after_costs"] is not None
    assert verdict["config"]["cost_bps_per_trade"] == 2.0
