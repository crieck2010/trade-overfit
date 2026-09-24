# METHODOLOGY — trade-overfit

Formulas, estimators, and honest limitations. All symbols: `T` = number of
return observations, returns are simple per-period (daily by convention).

## Performance metrics

- **Per-period risk-free:** `rf_p = (1 + rf_annual)^(1/252) − 1`.
- **Annualized Sharpe:** `SR = mean(r − rf_p) / stdev(r − rf_p) · √252`
  (sample stdev, n−1). Undefined (returns `None`) when T < 2 or volatility is zero.
- **Annualized Sortino:** `mean(r − rf_p) / downside_dev · √252`, where
  `downside_dev = √(mean(min(0, r − rf_p)²))`.
- **Max drawdown:** `min_t (W_t / max_{s≤t} W_s − 1)` on the wealth index
  `W_t = Π(1 + r_i)`; always ≤ 0. Duration = bars from peak to trough.
- **Calmar:** annualized geometric return / |max drawdown|.
- **Profit factor:** `Σ gains / Σ |losses|`; +∞ with no losing bars.
- **Skewness / kurtosis:** bias-corrected sample estimators (Fisher).
- **Cost haircut:** flat per-bar drag `(cost_bps/1e4) · trades_per_year / 252`.
  Deliberately turnover-agnostic — a conservative haircut, not a fill model.

## Probabilistic Sharpe Ratio (PSR)

`PSR(SR*) = Φ( (SR̂ − SR*)·√(T−1) / √(1 − skew·SR̂ + (kurt_excess/4)·SR̂²) )`

the estimated probability the true Sharpe exceeds `SR*` (Bailey & de Prado,
2014; the denominator is Lo's (2002) asymptotic variance of the Sharpe
estimator). `Φ` via `math.erf`.

## Deflated Sharpe Ratio (DSR) — the centerpiece

`DSR = PSR(SR₀)` — the PSR evaluated at the benchmark Sharpe expected under
the null of no skill after trying N strategies:

`SR₀ = √V̂ · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))`

- `γ = 0.5772156649` (Euler–Mascheroni),
- `V̂` = sample variance of the N in-sample trial Sharpes,
- `Φ⁻¹` = Acklam's rational approximation (~1e-9 accuracy).

Intuition: the best of N standard normals grows like `√(2·ln N)`; SR₀ is the
expected Sharpe of the luckiest of N unskilled strategies. A backtest Sharpe
below SR₀ is indistinguishable from the best of N coin flips. `N = 1` gives
`SR₀ = 0` (no selection bias to correct), so DSR = PSR(0).

Worked example (seeded demo, `demo_selection_bias()`): N = 1000 noise
strategies, T = 252. Best in-sample Sharpe ≈ 3.31 — spectacular. Expected
best-of-1000 under the null ≈ 3.29, so the winner's own skew/kurtosis give
DSR ≈ 0.83: not remotely near the 0.95 the desk demands → FAIL.
That is selection bias made visible.

## Walk-forward analysis

Rolling (default) or anchored (expanding) train windows of `train_len` bars,
stepping by `step` (default `test_len`) with non-overlapping test windows of
`test_len` bars. Per window: in-sample Sharpe, OOS Sharpe, OOS return, OOS
max drawdown. Summary: median/mean OOS Sharpe, OOS hit-rate (fraction of
windows with OOS Sharpe > 0), and **degradation** = mean IS Sharpe − mean OOS
Sharpe (positive = the edge decays out of sample).

## Regime splits

Vol-regime labeling: trailing 63-day annualized realized vol vs its sample
median → `stress` (above) / `calm` (below); warmup bars label `calm`.
User-supplied label vectors are also accepted (length must match). Per-regime
Sharpe / total return / max drawdown; the stability verdict is the
**worst-regime Sharpe** — an edge that only works in one regime is not an edge.

## Gates

Five default gates (all thresholds configurable; presets `standard`,
`strict`, `lenient`):

| Gate | Rule | Rationale |
|---|---|---|
| `deflated_sharpe` | DSR ≥ 0.95 | True Sharpe positive with 95% confidence after trial correction |
| `oos_sharpe` | median OOS Sharpe > 1.0 | Edge survives unseen data |
| `max_drawdown` | maxDD > −15% | Survivable, not just profitable |
| `regime_stability` | worst-regime Sharpe > 0 | Works in calm *and* stress |
| `beats_benchmark_net` | excess ann. return vs benchmark after costs > 0 | Beats buy-and-hold net |

Verdict PASS iff every applicable gate passes. The benchmark gate is skipped
(not failed) when no benchmark is supplied.

## Honest limitations

- **DSR assumes the N trial Sharpes are approximately normal** (it uses their
  variance and normal quantiles). Heavy-tailed or highly correlated trial
  Sharpes distort SR₀; the correction is then optimistic or pessimistic in
  ways the single number hides.
- **The DSR is only as honest as N.** It corrects for the trials you *report*.
  Strategies tried and discarded without recording their Sharpes are invisible
  to it — the classic file-drawer problem. Garbage in, gospel out.
- **Walk-forward still snoops if you re-tune.** Re-running walk-forward after
  tweaking parameters on the same history is just in-sample testing with extra
  steps. One clean pass, then freeze the parameters.
- **Regime labels are backward-looking.** The vol-median split is descriptive,
  not predictive; "stress" today may not resemble "stress" tomorrow.
- **Gates are conventions, not guarantees.** DSR ≥ 0.95, OOS Sharpe > 1.0,
  drawdown < 15% — these are reasonable, defensible, and ultimately arbitrary.
  Tune them to your mandate; a PASS is evidence, not a promise.
- **Costs are a haircut, not a simulator.** The flat per-bar drag ignores
  market impact, spread variation, and borrow costs.
- Synthetic demo data is Gaussian and stationary by construction; real returns
  are neither.
