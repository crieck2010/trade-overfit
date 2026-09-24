# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-24

### Fixed
- `demo_selection_bias()` now uses the winning strategy's own skew/kurtosis in
  the DSR (per Bailey & de Prado), so the demo DSR and the desk-gate DSR agree
  exactly (0.83 — still killed by the 0.95 gate).
- `examples/overfit_example.py` no longer crashes on gates with missing
  evidence (e.g. walk-forward on short series); prints `-` instead.

## [0.1.0] - 2026-09-24

### Added
- `metrics`: Sharpe, Sortino, max drawdown + duration, Calmar, profit factor,
  win rate, skewness, excess kurtosis, cost haircut, `performance_summary`.
- `dsr`: Probabilistic Sharpe Ratio and Deflated Sharpe Ratio
  (Bailey & de Prado 2014); normal CDF via `math.erf`, inverse-normal via
  Acklam's approximation; N=1 collapses to PSR(0); graceful edge cases.
- `walkforward`: rolling/anchored train-test windows, OOS Sharpe per window,
  hit-rate, median OOS Sharpe, train-vs-OOS degradation.
- `regimes`: vol-regime (trailing 63d realized vol vs median) calm/stress
  labels plus user-supplied labels; per-regime stats; worst-regime Sharpe.
- `gates`: configurable `Gate` dataclass, documented default set
  (DSR >= 0.95, OOS Sharpe > 1.0, maxDD < 15%, worst-regime Sharpe > 0,
  beats benchmark net of costs), `standard`/`strict`/`lenient` presets,
  `validate()` pipeline returning a PASS/FAIL verdict with per-gate evidence.
- `demo`: seeded true-edge series and the 1000-noise-strategy selection-bias
  demo (best in-sample Sharpe ~3.2, DSR ~0.6 — killed by the desk).
- `adapters`: lazy sibling adapters for trade-backtest, trade-agents,
  trade-paper (PASS-only promotion payloads).
- CLI: `validate`, `dsr`, `walk-forward`, `gates`, `metrics`, `presets`,
  `license`, `update-check` with table/JSON/CSV output.
- Docs: README with "The maths", `docs/ARCHITECTURE.md`,
  `docs/METHODOLOGY.md`, `examples/overfit_example.py`.
- 82 tests, all passing.
