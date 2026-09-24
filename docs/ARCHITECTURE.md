# ARCHITECTURE — trade-overfit

The overfitting desk: a strategy-validation engine that kills fake alpha before
anything ships. Pure Python, stdlib-only, zero UI-framework imports.

## Module map

```
src/trade_overfit/
├── __init__.py      # public API: validate, DSR, metrics, walk-forward, regimes, gates, demo, adapters
├── _stats.py        # internal: normal CDF (erf), Acklam inverse-normal CDF, moments
├── metrics.py       # performance stats from a returns series (Sharpe, Sortino, maxDD, Calmar, ...)
├── dsr.py           # Probabilistic Sharpe Ratio + Deflated Sharpe Ratio (Bailey & de Prado 2014)
├── walkforward.py   # rolling/anchored train-test windows, OOS Sharpe, degradation
├── regimes.py       # vol-regime splits (calm/stress) + user labels, per-regime stats
├── gates.py         # Gate dataclass, DEFAULT_GATES, PRESETS, evaluate_gates, validate()
├── demo.py          # seeded synthetic data: true edge vs N noise strategies
├── adapters.py      # lazy sibling adapters (trade-backtest, trade-agents, trade-paper)
├── licensing.py     # license-key + update-check hooks (suite convention, stubbed)
└── cli.py           # validate | dsr | walk-forward | gates | metrics | presets | license | update-check
```

## Design decisions

- **Plain data everywhere.** Inputs are `list[float]` returns; outputs are dicts
  and dataclasses. No pandas, no NumPy, no custom array types. Every result is
  JSON-serializable so the CLI, the dashboards, and sibling repos can consume it
  without importing anything beyond this package.
- **The desk is a pipeline, not a model.** `validate()` runs five stages in a
  fixed order — cost haircut → metrics → DSR → walk-forward → regime splits →
  gates — and returns one verdict dict. Each stage is independently callable and
  testable; the pipeline just wires them.
- **Missing data never passes.** A gate whose evidence is `None` (e.g. DSR on a
  1-bar series) FAILS. The one deliberate exception: the benchmark gate is
  *skipped* (not failed) when no benchmark series is supplied, because "no
  benchmark" is a configuration choice, not missing data.
- **DSR needs the trial distribution.** The deflated Sharpe ratio corrects for
  *selection bias across N trials*. A single Sharpe number cannot be deflated —
  you must pass the in-sample Sharpes of every strategy tried
  (`trial_sharpes`). With one trial the DSR correctly collapses to PSR(0).
  The CLI's `--trials N` flag without trial Sharpes only warns; it cannot
  manufacture the correction.
- **Lazy siblings.** `adapters.py` imports `trade_backtest` / `trade_agents`
  inside functions and raises a helpful error naming the pip package when one
  is absent. `promotion_payload()` shapes PASS verdicts for trade-paper as
  plain dicts — trade-paper never imports this package.
- **Deterministic demos.** All synthetic data comes from `random.Random(seed)`.
  The selection-bias demo (1000 noise strategies, 1 year) is the canonical
  illustration: best in-sample Sharpe ≈ 3.2, DSR ≈ 0.6 — a coin flip the desk
  refuses to ship.

## Scaling

- `metrics` is O(T); `walk_forward` is O(windows · T); regime labeling is O(T).
- Per-window walk-forward work is embarrassingly parallel — it can move to a
  process pool later without changing any public signature.
- The DSR itself is O(N) in the number of trials; the expensive part is
  *generating* N trial Sharpes, which belongs to the research desk
  (trade-agents), not here.

## Interop

| Sibling        | Adapter |
|----------------|---------|
| trade-backtest | `returns_from_backtest` — equity curve / returns / result object → returns |
| trade-agents   | `gate_research_idea` — validate a research idea before promotion |
| trade-paper    | `promotion_payload` — only PASS verdicts become promotion payloads |
| trade-suite    | consumes `validate()` verdicts in the research pipeline |

## Conventions honored

Suite-standard: engine/UI split (there is no UI), stdlib-only, semantic
versioning + CHANGELOG, stubbed license/update hooks, README "The maths"
section (what you learn / why it matters / the maths / honest limitations),
fresh-clone verification before release.
