"""The money demo: 1000 noise strategies vs one true edge.

Run:  PYTHONPATH=src python examples/overfit_example.py
"""

from trade_overfit import (
    demo_noise_strategies,
    demo_selection_bias,
    demo_true_edge,
    promotion_payload,
    validate,
)

print("=== Part 1: selection bias under multiple testing ===")
demo = demo_selection_bias(n_strategies=1000, n=252, seed=7)
print(f"tried {demo['n_strategies']} zero-edge strategies on {demo['n_obs']} bars")
print(f"best in-sample Sharpe : {demo['best_in_sample_sharpe']:.2f}  <- looks amazing")
print(f"expected best (null)  : {demo['expected_sharpe_under_null']:.2f}  <- luck explains it")
print(f"DSR of the winner     : {demo['best_dsr']:.2f}  <- far short of the 0.95 gate")
print(f"verdict on the winner : {'KILLED' if demo['killed'] else 'SURVIVED'}")


def _v(x):
    return f"{x:.4f}" if x is not None else "-"


print("\n=== Part 2: the winner through the full desk ===")
winner = demo_noise_strategies(1000, 252, 7)[demo["best_idx"]]
verdict = validate(winner, trial_sharpes=demo["trial_sharpes"])
print(f"desk verdict: {verdict['verdict']} "
      f"({verdict['n_gates_passed']}/{verdict['n_gates']} gates)")
for g in verdict["gates"]:
    mark = "PASS" if g["passed"] else "FAIL"
    print(f"  [{mark}] {g['name']:<22} value={_v(g['value'])}  "
          f"(needs {g['op']} {g['threshold']})")

print("\n=== Part 3: a true edge survives ===")
edge = demo_true_edge(seed=7)
verdict = validate(edge)
print(f"desk verdict: {verdict['verdict']} "
      f"({verdict['n_gates_passed']}/{verdict['n_gates']} gates)")
print(f"DSR={verdict['evidence']['dsr']:.3f}, "
      f"median OOS Sharpe={verdict['evidence']['median_oos_sharpe']:.2f}, "
      f"maxDD={verdict['evidence']['max_drawdown']:.2%}")
payload = promotion_payload(verdict)
print("promotion payload for trade-paper:", "created" if payload else "blocked (FAIL)")

print("\nTakeaway: a 3+ Sharpe that is really the best of 1000 coin flips gets")
print("killed; a genuine Sharpe ~1.6 survives. That is the overfitting desk.")
