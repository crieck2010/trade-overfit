# trade-overfit

The overfitting desk for the [trade-suite](https://github.com/crieck2010/trade-suite) — a strategy-validation engine that kills fake alpha before anything ships. Pure-Python, stdlib-only, no NumPy, no pandas.

What it does:

- **Deflated Sharpe Ratio (DSR)** — Bailey & de Prado (2014): the probability your backtest Sharpe is real *after correcting for how many strategies you tried*
- **Walk-forward analysis** — rolling or anchored train/test windows, out-of-sample Sharpe per window, train-vs-OOS degradation
- **Regime splits** — does the edge survive calm *and* stress markets (trailing-vol regimes or your own labels)
- **Pass/fail gates** — DSR ≥ 0.95, OOS Sharpe > 1.0, max drawdown < 15%, worst-regime Sharpe > 0, beats benchmark net of costs; every threshold configurable, every gate carries its evidence
- **Seeded selection-bias demo** — 1000 zero-edge strategies: the best in-sample Sharpe looks spectacular (~3.2) and the DSR kills it (~0.6)
- **Suite adapters (all lazy)** — returns from trade-backtest, idea gating for trade-agents, PASS-only promotion payloads for trade-paper

Research/backtesting/paper-trading only — never live trading, never personalized investment advice.

## Install

```bash
pip install git+https://github.com/crieck2010/trade-overfit.git
```

Requires Python 3.10+. No third-party dependencies.

## Quick start

```bash
# the money demo: best of 1000 noise strategies through the full desk
trade-overfit validate --demo noise-winner

# a genuine edge through the full desk
trade-overfit validate --demo edge

# your own returns (one column, header optional)
trade-overfit validate --csv my_strategy.csv --benchmark spy.csv --cost-bps 2

# just the deflated Sharpe ratio, JSON out
trade-overfit dsr --demo noise-winner --format json

# walk-forward and gate detail
trade-overfit walk-forward --demo edge --train-len 252 --test-len 63
trade-overfit gates --demo edge --preset strict
trade-overfit presets
```

```python
from trade_overfit import demo_selection_bias, demo_true_edge, validate

# the desk needs the trial Sharpes, not just the winner's
demo = demo_selection_bias(n_strategies=1000, seed=7)
verdict = validate(winner_returns, trial_sharpes=demo["trial_sharpes"])
print(verdict["verdict"], verdict["evidence"]["dsr"])   # FAIL 0.61

verdict = validate(demo_true_edge(seed=7))
print(verdict["verdict"], verdict["evidence"]["dsr"])   # PASS 1.0
```

## CLI

```
trade-overfit validate      --demo edge|noise-winner | --csv FILE [--benchmark FILE]
                            [--cost-bps N] [--trades-per-year N] [--preset standard|strict|lenient]
trade-overfit dsr           --demo ... | --csv FILE
trade-overfit walk-forward  --demo ... | --csv FILE [--train-len 252] [--test-len 63] [--anchored]
trade-overfit gates         --demo ... | --csv FILE [--preset NAME]
trade-overfit metrics       --demo ... | --csv FILE
trade-overfit presets
trade-overfit license | trade-overfit update-check
```

Common options: `--seed 7`, `--rf 0.02`, `--format table|json|csv`.

## The maths

**What you learn.** Whether a backtest result is *real edge* or *the luckiest of N tries*. The desk computes the full performance picture (Sharpe, Sortino, max drawdown, Calmar, profit factor, skew/kurtosis), then attacks it three ways: the Deflated Sharpe Ratio corrects the in-sample Sharpe for selection bias across every strategy you tried; walk-forward analysis measures how much Sharpe evaporates on unseen data; regime splits check the edge survives both calm and stress markets. Five gates turn all of that into a PASS/FAIL verdict with per-gate evidence.

**Why it matters.** Trying 1000 strategies and shipping the best is not research — it is buying the maximum of 1000 coin flips. With one year of daily data the expected best Sharpe of 1000 *zero-edge* strategies is ≈ 3.2, a number most researchers would frame and hang on a wall. The DSR is the antidote: it asks "what Sharpe would the luckiest unskilled strategy have achieved?" and demands yours beat *that*. Walk-forward and regime splits attack the other two classic lies: parameters tuned to one history, and edges that only work in one market mood. If a strategy cannot survive the desk, it does not ship.

**The maths.**

- *Annualized Sharpe:* `SR = mean(r − rf_p) / stdev(r − rf_p) · √252`, with `rf_p = (1 + rf_annual)^(1/252) − 1`. Undefined (never invented) when T < 2 or volatility is zero.
- *Probabilistic Sharpe Ratio:* `PSR(SR*) = Φ( (SR̂ − SR*)·√(T−1) / √(1 − skew·SR̂ + (kurt_excess/4)·SR̂²) )` — the probability the true Sharpe exceeds `SR*`, with `Φ` via `math.erf`.
- *Deflated Sharpe Ratio (Bailey & de Prado 2014):* `DSR = PSR(SR₀)`, where `SR₀ = √V̂ · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))`, `γ = 0.5772156649` (Euler–Mascheroni), `V̂` = variance of the N trial Sharpes, `Φ⁻¹` = Acklam's approximation. `SR₀` is the expected Sharpe of the luckiest of N unskilled strategies — the best of N normals grows like `√(2·ln N)`. With N = 1 there is no selection bias, so `SR₀ = 0` and DSR = PSR(0).
- *Walk-forward:* rolling (or anchored/expanding) train windows of `train_len` bars stepped by `test_len`; per test window the OOS Sharpe, OOS return, OOS max drawdown. **Degradation** = mean in-sample Sharpe − mean OOS Sharpe: positive means the edge decays out of sample.
- *Regime splits:* trailing 63-day annualized realized vol vs its sample median → `stress`/`calm` labels (or your own label vector). The stability verdict is the **worst-regime Sharpe**.
- *Gates (defaults, all configurable):* DSR ≥ 0.95 · median OOS Sharpe > 1.0 · max drawdown > −15% · worst-regime Sharpe > 0 · annualized excess return vs benchmark after the cost haircut > 0 (skipped, not failed, when no benchmark is given). Missing evidence never passes.

**Honest limitations.** The DSR assumes the N trial Sharpes are roughly normal — heavy-tailed or correlated trials distort the correction, and it can only correct for the trials you *report* (the file-drawer problem: untried-and-discarded strategies are invisible to it). Walk-forward still snoops if you re-tune on the same history — one clean pass, then freeze the parameters. Regime labels are backward-looking descriptions, not predictions. The cost haircut is a flat per-bar drag, not a fill simulator. The gate thresholds (0.95, 1.0, 15%) are defensible conventions, not laws of nature — a PASS is evidence, not a promise. The demo data is Gaussian and stationary; real returns are neither.

## Interop

| Sibling | Adapter |
|---|---|
| trade-backtest | `returns_from_backtest` — equity curve / returns / result object → returns list |
| trade-agents | `gate_research_idea` — run a research idea through the desk before promotion |
| trade-paper | `promotion_payload` — only PASS verdicts become promotion payloads (plain data) |
| trade-suite | consumes `validate()` verdicts in the research pipeline |

All sibling imports are lazy — trade-overfit imports and tests clean with nothing else installed.

## Docs

- `docs/ARCHITECTURE.md` — module map and design decisions
- `docs/METHODOLOGY.md` — formulas, estimators, and honest limitations
- `examples/overfit_example.py` — true edge vs 1000 noise strategies, end to end

## License

MIT. See `LICENSE`.
