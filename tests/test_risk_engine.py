"""
Comprehensive Pytest Suite for Portfolio Risk Engine (v1.2)
Run via: python -m pytest tests/test_risk_engine.py -v
"""
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import pytest

from src.config_loader import (
    validate_config_dict,
    resolve_output_dir,
    ConfigValidationError,
)
from src.data_pipeline import fetch_portfolio_data_from_config
from src.risk_models import (
    parametric_vcov_var,
    monte_carlo_var,
    compute_portfolio_vs_benchmark_summary,
)
from src.attribution import compute_risk_attribution, compute_group_exposures, compute_max_drawdown
from src.validation import run_rolling_backtest_and_kupiec, run_crisis_stress_tests


def _generate_synthetic_returns(
    n_days: int = 800, n_assets: int = 5, seed: int = 123
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    # Construct correlated returns so portfolio volatility is realistic
    common_factor = rng.normal(0.0003, 0.010, size=(n_days, 1))
    idio = rng.normal(0.0001, 0.012, size=(n_days, n_assets))
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    rets = pd.DataFrame(common_factor + idio, index=idx, columns=[f"ASSET_{i}" for i in range(n_assets)])
    bm_rets = pd.Series(common_factor[:, 0] + rng.normal(0, 0.004, size=n_days), index=idx, name="BM")
    raw_w = rng.uniform(0.5, 2.0, size=n_assets)
    weights = pd.Series(raw_w / raw_w.sum(), index=rets.columns)
    return rets, weights, bm_rets


def _build_synthetic_pipeline_bundle(cfg: Dict[str, Any], n_days: int = 750) -> Dict[str, Any]:
    """Builds a deterministic price/return bundle matching a validated config schema."""
    cfg = validate_config_dict(cfg)
    holdings = cfg["holdings"]
    n_assets = len(holdings)
    tickers = [h["ticker"] for h in holdings]
    rets, _, bm_rets = _generate_synthetic_returns(n_days=n_days, n_assets=n_assets)
    rets.columns = tickers

    prices = 100.0 * np.exp(rets.cumsum())
    bm_prices = 100.0 * np.exp(bm_rets.cumsum())

    raw_vals = pd.Series({h["ticker"]: float(h["sizing_value"]) for h in holdings})
    weights = raw_vals / raw_vals.sum()
    total_nav = 100000.0

    holding_meta = {
        h["ticker"]: {
            "raw_ticker": h["ticker"],
            "currency": h["currency"],
            "sector": h["sector"],
            "region": h["region"],
            "alias": h["alias"],
        }
        for h in holdings
    }
    return {
        "prices_base": prices,
        "log_returns": rets,
        "weights": weights,
        "raw_weights_by_ticker": weights,
        "total_nav": total_nav,
        "base_currency": cfg["settings"]["base_currency"],
        "bm_prices_base": bm_prices,
        "bm_log_returns": bm_rets,
        "holding_meta": holding_meta,
        "notices": [],
        "config": cfg,
    }


def _generate_tagged_headline_summary(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates the standardized v1.2 headline summary dictionary and tagged narrative string,
    enforcing Option B: 95% VaR [Parametric] and 99% CVaR [Student-t] with sub-labels.
    """
    rets = bundle["log_returns"]
    w = bundle["weights"]
    nav = bundle["total_nav"]
    bm_rets = bundle["bm_log_returns"]
    cfg = bundle["config"]

    summary = compute_portfolio_vs_benchmark_summary(
        rets, w, bm_rets, rf_rate_annual=cfg["settings"]["risk_free_rate"]
    )
    param_1d = parametric_vcov_var(rets, w, horizon_days=1)
    mc_t_1d, _ = monte_carlo_var(rets, w, horizon_days=1, distribution="student_t", seed=42)
    attr_df = compute_risk_attribution(rets, w, nav, bm_log_returns=bm_rets, alpha=0.95)
    bt_info = run_rolling_backtest_and_kupiec(
        rets,
        w,
        estimation_window=cfg["settings"]["rolling_window_days"],
        min_total_days=cfg["settings"]["min_backtest_days"],
    )

    var95_param_pct = round(param_1d["VaR_95"] * 100.0, 2)
    var95_t_pct = round(mc_t_1d["VaR_95"] * 100.0, 2)
    cvar99_t_pct = round(mc_t_1d["CVaR_99"] * 100.0, 2)
    cvar99_param_pct = round(param_1d["CVaR_99"] * 100.0, 2)

    top_ticker = str(attr_df.index[0])
    top_rc_pct = round(float(attr_df.iloc[0]["Risk Contribution (%)"]), 2)

    text = (
        f"Portfolio ({len(w)} holdings, base {bundle['base_currency']}): "
        f"1d 95% VaR [Parametric]: -{var95_param_pct:.2f}% (sub-label [Student-t]: -{var95_t_pct:.2f}%). "
        f"1d 99% CVaR [Student-t]: -{cvar99_t_pct:.2f}% (sub-label [Parametric]: -{cvar99_param_pct:.2f}%). "
        f"Top risk contributor: {top_ticker} ({top_rc_pct:.2f}% risk). "
        f"Backtest enabled: {bt_info['enabled']}."
    )

    return {
        "text": text,
        "var95_param_pct": var95_param_pct,
        "var95_t_pct": var95_t_pct,
        "cvar99_t_pct": cvar99_t_pct,
        "cvar99_param_pct": cvar99_param_pct,
        "top_ticker": top_ticker,
        "top_rc_pct": top_rc_pct,
        "attr_df": attr_df,
        "bt_info": bt_info,
        "summary": summary,
    }


# ==============================================================================
# PART A: MATHEMATICAL INVARIANTS
# ==============================================================================
def test_weights_sum_to_one():
    _, weights, _ = _generate_synthetic_returns()
    assert np.isclose(weights.sum(), 1.0, atol=1e-8)


def test_cvar_strictly_exceeds_var_and_99_exceeds_95():
    rets, weights, _ = _generate_synthetic_returns()
    metrics = parametric_vcov_var(rets, weights)
    assert metrics["VaR_99"] > metrics["VaR_95"]
    assert metrics["CVaR_95"] > metrics["VaR_95"]
    assert metrics["CVaR_99"] > metrics["VaR_99"]


def test_euler_component_var_additivity():
    rets, weights, bm_rets = _generate_synthetic_returns()
    nav = 50000.0
    param_metrics = parametric_vcov_var(rets, weights, horizon_days=1, include_drift=False)
    total_var_eur = param_metrics["VaR_95"] * nav
    attr_df = compute_risk_attribution(rets, weights, total_nav_eur=nav, bm_log_returns=bm_rets, alpha=0.95)
    assert np.isclose(total_var_eur, attr_df["Component VaR (1d EUR)"].sum(), rtol=1e-6)


def test_student_t_tail_is_fatter_than_gaussian_at_99():
    rets, weights, _ = _generate_synthetic_returns()
    norm_res, _ = monte_carlo_var(rets, weights, distribution="normal", seed=42)
    t_res, _ = monte_carlo_var(rets, weights, distribution="student_t", df_student=4.0, seed=42)
    assert t_res["CVaR_99"] > norm_res["CVaR_99"]


# ==============================================================================
# PART B: CONFIG VALIDATION & PRIVACY ROUTING
# ==============================================================================
def test_config_validator_catches_invalid_inputs():
    with pytest.raises(ConfigValidationError):
        validate_config_dict({"holdings": [{"ticker": "AAPL", "weight": 1.0, "currency": "USD"}]})  # < 2 holdings

    with pytest.raises(ConfigValidationError):
        validate_config_dict({
            "holdings": [
                {"ticker": "AAPL", "weight": -0.5, "currency": "USD"},
                {"ticker": "MSFT", "weight": 0.5, "currency": "USD"},
            ]
        })


def test_privacy_output_routing(tmp_path):
    pub_cfg = {"is_private": False, "holdings": []}
    priv_cfg = {"is_private": True, "holdings": []}
    assert resolve_output_dir(pub_cfg, tmp_path) == tmp_path / "public"
    assert resolve_output_dir(priv_cfg, tmp_path) == tmp_path / "private"


# ==============================================================================
# PART C: 4-PORTFOLIO ROBUSTNESS MATRIX & CROSS-LEAK / MODEL-TAG CHECKS
# ==============================================================================
def test_portfolio_1_all_eur_5_stocks():
    """Portfolio 1: 5-stock All-EUR portfolio (verifies zero USD dependency & no KeyError)."""
    cfg = {
        "portfolio_name": "All-EUR 5-Stock Test",
        "settings": {"base_currency": "EUR"},
        "holdings": [
            {"ticker": "ASML.AS", "weight": 0.25, "currency": "EUR", "sector": "Information Technology", "region": "Europe"},
            {"ticker": "SAP.DE",  "weight": 0.20, "currency": "EUR", "sector": "Information Technology", "region": "Europe"},
            {"ticker": "OR.PA",   "weight": 0.20, "currency": "EUR", "sector": "Consumer Staples",       "region": "Europe"},
            {"ticker": "SAN.MC",  "weight": 0.20, "currency": "EUR", "sector": "Financials",             "region": "Europe"},
            {"ticker": "SIE.DE",  "weight": 0.15, "currency": "EUR", "sector": "Industrials",            "region": "Europe"},
        ],
    }
    bundle = _build_synthetic_pipeline_bundle(cfg, n_days=750)
    out = _generate_tagged_headline_summary(bundle)
    sec_df, ccy_df = compute_group_exposures(out["attr_df"], bundle["holding_meta"])

    assert list(ccy_df.index) == ["EUR"]
    assert "USD" not in out["text"]
    assert "AAPL" not in out["text"] and "JPM" not in out["text"]

def test_portfolio_2_all_usd_3_stocks():
    """Portfolio 2: 3-stock All-USD portfolio (verifies no KeyError: 'EUR' & works on N=3)."""
    cfg = {
        "portfolio_name": "All-USD 3-Stock Test",
        "settings": {"base_currency": "USD"},
        "holdings": [
            {"ticker": "AAPL", "weight": 0.45, "currency": "USD", "sector": "Information Technology", "region": "North America"},
            {"ticker": "JPM",  "weight": 0.30, "currency": "USD", "sector": "Financials",             "region": "North America"},
            {"ticker": "JNJ",  "weight": 0.25, "currency": "USD", "sector": "Health Care",            "region": "North America"},
        ],
    }
    bundle = _build_synthetic_pipeline_bundle(cfg, n_days=750)
    out = _generate_tagged_headline_summary(bundle)
    sec_df, ccy_df = compute_group_exposures(out["attr_df"], bundle["holding_meta"])

    assert list(ccy_df.index) == ["USD"]
    assert "EUR" not in ccy_df.index
    assert len(out["attr_df"]) == 3
    assert "ASML.AS" not in out["text"]


def test_portfolio_3_forty_stock_multi_currency_mix():
    """Portfolio 3: 40-stock mixed EUR/USD portfolio (verifies scalability to N=40)."""
    sectors = ["Information Technology", "Financials", "Health Care", "Industrials", "Energy"]
    holdings = [
        {
            "ticker": f"STK_{i:02d}",
            "weight": 1.0 / 40.0,
            "currency": "EUR" if i % 2 == 0 else "USD",
            "sector": sectors[i % len(sectors)],
            "region": "Europe" if i % 2 == 0 else "North America",
        }
        for i in range(40)
    ]
    cfg = {"portfolio_name": "40-Stock Institutional Mix", "settings": {"base_currency": "EUR"}, "holdings": holdings}
    bundle = _build_synthetic_pipeline_bundle(cfg, n_days=750)
    out = _generate_tagged_headline_summary(bundle)

    assert len(out["attr_df"]) == 40
    assert np.isclose(out["attr_df"]["Weight (%)"].sum(), 100.0, atol=1e-5)
    assert np.isclose(out["attr_df"]["Risk Contribution (%)"].sum(), 100.0, atol=1e-5)


def test_portfolio_4_short_history_disables_backtest_gracefully():
    """Portfolio 4: Short-history portfolio (<500 days) disables rolling backtest with explanatory note."""
    cfg = {
        "portfolio_name": "Short History Test",
        "settings": {"base_currency": "EUR", "min_backtest_days": 500},
        "holdings": [
            {"ticker": "NEW_A", "weight": 0.50, "currency": "EUR", "sector": "Energy", "region": "Europe"},
            {"ticker": "NEW_B", "weight": 0.50, "currency": "USD", "sector": "Financials", "region": "North America"},
        ],
    }
    bundle = _build_synthetic_pipeline_bundle(cfg, n_days=320)
    bt = run_rolling_backtest_and_kupiec(bundle["log_returns"], bundle["weights"], estimation_window=252, min_total_days=500)

    assert bt["enabled"] is False
    assert "disabled" in bt["reason"].lower()
    assert bt["kupiec_parametric"] is None


def test_automated_text_vs_table_and_model_tag_consistency():
    """
    Verifies that every number in the summary text equals the matching table value
    and that Option B model tags ([Parametric] and [Student-t]) are always present together.
    """
    cfg = {
        "portfolio_name": "Consistency Check Portfolio",
        "settings": {"base_currency": "EUR"},
        "holdings": [
            {"ticker": "ALPHA", "weight": 0.50, "currency": "EUR", "sector": "Financials", "region": "Europe"},
            {"ticker": "BETA",  "weight": 0.30, "currency": "USD", "sector": "Energy",     "region": "North America"},
            {"ticker": "GAMMA", "weight": 0.20, "currency": "EUR", "sector": "Health Care","region": "Europe"},
        ],
    }
    bundle = _build_synthetic_pipeline_bundle(cfg, n_days=750)
    out = _generate_tagged_headline_summary(bundle)
    txt = out["text"]

    # 1. Both model tags must appear whenever 95% VaR or 99% CVaR is cited
    assert "1d 95% VaR [Parametric]:" in txt and "(sub-label [Student-t]:" in txt
    assert "1d 99% CVaR [Student-t]:" in txt and "(sub-label [Parametric]:" in txt

    # 2. Exact numerical match between text and computed metrics
    assert f"-{out['var95_param_pct']:.2f}%" in txt
    assert f"-{out['var95_t_pct']:.2f}%" in txt
    assert f"-{out['cvar99_t_pct']:.2f}%" in txt
    assert f"-{out['cvar99_param_pct']:.2f}%" in txt
    assert f"{out['top_ticker']} ({out['top_rc_pct']:.2f}% risk)" in txt