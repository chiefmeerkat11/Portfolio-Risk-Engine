# Changelog

## [v1.2.1] - 2026-11-19
### Fixed & Polished
- **Strict Rebalancing Simulator Filtering:** Restricted simulated trims strictly to holdings exceeding the configured `risk_weight_ratio_high` threshold rather than trimming neutral holdings.
- **Unified Volatility & Loss Sign Conventions:** Standardized log-return covariance volatility across all summary tables and applied consistent `-` loss signs across KPI cards, narrative takeaways, and VaR/CVaR tables.
- **Grammar & Currency Risk Tracking:** Added dynamic singular/plural formatting in advisory messages and added post-rebalance `Non-EUR Currency Risk Share` tracking to the Rebalancing Impact table.

## [v1.2.0] - 2026-11-19
### Added
- **Portfolio-Agnostic JSON Schema & Validator (`src/config_loader.py`):** Added support for any user-supplied portfolio (`shares`, `amount`, or `weight` sizing; arbitrary currencies, sectors, and regions) via `--config` CLI flag, with friendly schema validation and automatic privacy routing (`outputs/public/` vs. gitignored `outputs/private/`).
- **Rule-Based Advisory & Rebalancing Engine (`src/advisory.py`):** Added an 8-check traffic-light governance scorecard scaled to portfolio size plus a risk-only Rebalancing Impact Simulator.
- **"Answer First, Evidence Second" Report Redesign:** Restructured the HTML/PDF report with a "Read This First" executive summary, hover tooltips on financial terms, explicit `[Parametric]` / `[Student-t]` model tags on every headline figure, and dynamic proxy/inception disclosure for pre-2009 crisis replays.
- **Expanded 4-Portfolio Pytest Suite (`tests/test_risk_engine.py`):** Added end-to-end robustness tests covering a 5-stock all-EUR portfolio, a 3-stock all-USD portfolio, a 40-stock mixed portfolio, a `<500`-day short-history portfolio, and automated text-vs-table number consistency checks.

## [v1.1.0] - 2026-11-14
### Added
- Added EUR Benchmark Comparison (`IWDA.AS`), Diversification Ratio, Effective Number of Bets ($N_{\text{eff}}$), Sector/Currency attribution tables, 10-day governance horizon, and 1-year drift sensitivity analysis.

## [v1.0.0] - 2026-11-11
- Initial modular Python risk engine (`data_pipeline.py`, `risk_models.py`, `attribution.py`, `validation.py`) with EUR/USD spot FX conversion, Student-$t$ Cholesky Monte Carlo, Euler risk attribution, and Kupiec POF backtesting.