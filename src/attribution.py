"""
Module 3: Euler Risk Attribution, Sector/Currency Concentration & Benchmark Drawdown (v1.1)
Replaces redundant internal Portfolio Beta column with External Benchmark Beta (vs MSCI World EUR)
and classifies positions by their Risk-to-Weight ratio.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def compute_risk_attribution(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    total_nav_eur: float,
    bm_log_returns: pd.Series | None = None,
    alpha: float = 0.95
) -> pd.DataFrame:
    """
    Computes Euler risk decomposition (Marginal VaR, Component VaR, % Risk Contribution,
    Risk/Weight Ratio) plus external Market Beta against MSCI World (EUR).
    """
    w = weights.values
    cov_matrix = log_returns.cov().values
    port_vol = float(np.sqrt(w.T @ cov_matrix @ w))
    z_alpha = float(stats.norm.ppf(alpha))
    total_port_var = z_alpha * port_vol

    cov_with_port = cov_matrix @ w
    marginal_var = z_alpha * (cov_with_port / port_vol)
    component_var_pct = w * marginal_var
    component_var_eur = component_var_pct * total_nav_eur
    pct_contribution = (component_var_pct / total_port_var) * 100.0
    risk_weight_ratio = pct_contribution / (w * 100.0)

    # External Benchmark Beta (Cov(r_i, r_BM) / Var(r_BM))
    if bm_log_returns is not None:
        bm_var = float(np.var(bm_log_returns.values, ddof=1))
        bm_betas = [
            float(np.cov(log_returns[col].values, bm_log_returns.values, ddof=1)[0, 1] / bm_var)
            for col in weights.index
        ]
    else:
        bm_betas = list(risk_weight_ratio)

    roles = []
    for rwr in risk_weight_ratio:
        if rwr > 1.20:
            roles.append("Risk Amplifier (>1.20x)")
        elif rwr < 0.80:
            roles.append("Diversifier (<0.80x)")
        else:
            roles.append("Core / Neutral (0.80-1.20x)")

    df_attr = pd.DataFrame({
        "Weight (%)": w * 100.0,
        "Position EUR": w * total_nav_eur,
        "Ann. Volatility (%)": np.sqrt(np.diag(cov_matrix) * 252.0) * 100.0,
        "Benchmark Beta": bm_betas,
        "Marginal VaR (1d %)": marginal_var * 100.0,
        "Component VaR (1d EUR)": component_var_eur,
        "Risk Contribution (%)": pct_contribution,
        "Risk / Weight Ratio": risk_weight_ratio,
        "Risk Role": roles,
    }, index=weights.index)

    return df_attr.sort_values(by="Risk Contribution (%)", ascending=False)


def compute_group_exposures(
    attr_df: pd.DataFrame,
    portfolio: Dict[str, Dict[str, object]],
    anonymize: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates Capital Weight (%) vs. Risk Contribution (%) by Sector and by Currency Exposure.
    Automatically detects dynamic base currency column names (e.g. 'Position (EUR)' or 'Position (USD)').
    """
    sector_map = {
        (str(v["alias"]) if anonymize else k): str(v.get("sector", "Unclassified"))
        for k, v in portfolio.items()
    }
    ccy_map = {
        (str(v["alias"]) if anonymize else k): str(v.get("currency", "EUR"))
        for k, v in portfolio.items()
    }

    pos_col = [c for c in attr_df.columns if c.startswith("Position")][0]
    cvar_col = [c for c in attr_df.columns if c.startswith("Component VaR")][0]

    temp = attr_df.copy()
    temp["Sector"] = [sector_map.get(idx, "Other") for idx in temp.index]
    temp["Currency Bucket"] = [ccy_map.get(idx, "Other") for idx in temp.index]

    agg_cols = [pos_col, "Weight (%)", cvar_col, "Risk Contribution (%)"]

    sector_df = (
        temp.groupby("Sector")[agg_cols]
        .sum()
        .sort_values(by="Risk Contribution (%)", ascending=False)
    )
    sector_df["Risk / Weight Ratio"] = sector_df["Risk Contribution (%)"] / sector_df["Weight (%)"]

    ccy_df = (
        temp.groupby("Currency Bucket")[agg_cols]
        .sum()
        .sort_values(by="Risk Contribution (%)", ascending=False)
    )
    ccy_df["Risk / Weight Ratio"] = ccy_df["Risk Contribution (%)"] / ccy_df["Weight (%)"]

    return sector_df, ccy_df


def compute_max_drawdown(
    prices_eur: pd.DataFrame,
    weights: pd.Series,
    bm_prices_eur: pd.Series | None = None
) -> Dict[str, Any]:
    """
    Reconstructs the historical constant-weight portfolio wealth index and calculates
    Maximum Drawdown alongside the MSCI World (EUR) Benchmark Drawdown.
    """
    simple_rets = prices_eur.pct_change().dropna()
    port_rets = simple_rets @ weights
    wealth_index = (1.0 + port_rets).cumprod()
    running_peak = wealth_index.cummax()
    drawdown_series = (wealth_index - running_peak) / running_peak

    max_dd = float(drawdown_series.min())
    trough_date = drawdown_series.idxmin()
    peak_date = wealth_index.loc[:trough_date].idxmax()

    result: Dict[str, Any] = {
        "max_drawdown_pct": max_dd * 100.0,
        "peak_date": str(peak_date.date()),
        "trough_date": str(trough_date.date()),
        "drawdown_series": drawdown_series,
        "wealth_index": wealth_index,
    }

    if bm_prices_eur is not None:
        bm_rets = bm_prices_eur.pct_change().loc[simple_rets.index].fillna(0.0)
        bm_wealth = (1.0 + bm_rets).cumprod()
        bm_peak = bm_wealth.cummax()
        bm_dd_series = (bm_wealth - bm_peak) / bm_peak
        bm_max_dd = float(bm_dd_series.min())
        bm_trough = bm_dd_series.idxmin()
        bm_peak_dt = bm_wealth.loc[:bm_trough].idxmax()
        result["bm_drawdown_series"] = bm_dd_series
        result["bm_max_drawdown_pct"] = bm_max_dd * 100.0
        result["bm_peak_date"] = str(bm_peak_dt.date())
        result["bm_trough_date"] = str(bm_trough.date())

    return result