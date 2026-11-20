"""
Module 1: Portfolio-Agnostic Multi-Currency Data Pipeline (v1.2)
Supports arbitrary base currencies, shares/amount/weight sizing, automatic FX pair
resolution, short-history detection, and backward compatibility with v1.1 calls.
"""
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import yfinance as yf

from src.config_loader import load_config_file, validate_config_dict

# Load public example config as default fallback dictionary for backward compatibility
PORTFOLIO_HOLDINGS: Dict[str, Dict[str, object]] = {
    "ASML.AS": {"shares": 3,   "currency": "EUR", "sector": "Information Technology", "region": "Europe",        "alias": "EU_SEMI_1"},
    "AAPL":    {"shares": 10,  "currency": "USD", "sector": "Information Technology", "region": "North America", "alias": "US_TECH_1"},
    "SAP.DE":  {"shares": 8,   "currency": "EUR", "sector": "Information Technology", "region": "Europe",        "alias": "EU_SOFT_1"},
    "JPM":     {"shares": 6,   "currency": "USD", "sector": "Financials",             "region": "North America", "alias": "US_BANK_1"},
    "SAN.MC":  {"shares": 200, "currency": "EUR", "sector": "Financials",             "region": "Europe",        "alias": "EU_BANK_1"},
    "OR.PA":   {"shares": 3,   "currency": "EUR", "sector": "Consumer Staples",       "region": "Europe",        "alias": "EU_CONS_1"},
    "JNJ":     {"shares": 6,   "currency": "USD", "sector": "Health Care",            "region": "North America", "alias": "US_HLTH_1"},
    "SIE.DE":  {"shares": 5,   "currency": "EUR", "sector": "Industrials",            "region": "Europe",        "alias": "EU_IND_1"},
    "XOM":     {"shares": 7,   "currency": "USD", "sector": "Energy",                 "region": "North America", "alias": "US_NRG_1"},
    "PG":      {"shares": 5,   "currency": "USD", "sector": "Consumer Staples",       "region": "North America", "alias": "US_STPL_1"},
}


def _convert_series_to_base_ccy(
    series: pd.Series,
    asset_ccy: str,
    base_ccy: str,
    cleaned_df: pd.DataFrame,
) -> pd.Series:
    """Converts a local-currency price series into base_ccy using Yahoo Finance spot FX."""
    if asset_ccy == base_ccy:
        return series.copy()
    direct_pair = f"{base_ccy}{asset_ccy}=X"   # e.g. EURUSD=X (units of asset_ccy per 1 base_ccy)
    inverse_pair = f"{asset_ccy}{base_ccy}=X"  # e.g. USDEUR=X (units of base_ccy per 1 asset_ccy)
    if direct_pair in cleaned_df.columns and cleaned_df[direct_pair].notna().sum() > 10:
        return series / cleaned_df[direct_pair]
    if inverse_pair in cleaned_df.columns and cleaned_df[inverse_pair].notna().sum() > 10:
        return series * cleaned_df[inverse_pair]
    raise ValueError(f"Could not resolve FX conversion from {asset_ccy} to {base_ccy}.")


def fetch_portfolio_data_from_config(
    cfg: Dict[str, Any],
    anonymize: bool = False,
) -> Dict[str, Any]:
    """
    Primary v1.2 data ingestion pipeline driven by a validated config dictionary.
    Handles arbitrary currencies, delisted/missing ticker checks, weight normalization,
    and logs any ticker that truncates the historical window.
    """
    cfg = validate_config_dict(cfg) if "sizing_mode" not in cfg["holdings"][0] else cfg
    settings = cfg["settings"]
    base_ccy = settings["base_currency"]
    start_date = settings["start_date"]
    end_date = settings["end_date"]
    max_ffill = int(settings["max_ffill_days"])
    bm_ticker = settings["benchmark_ticker"]
    bm_ccy = settings.get("benchmark_currency", base_ccy)

    holdings_list = cfg["holdings"]
    tickers = [h["ticker"] for h in holdings_list]

    # Identify all required FX pairs relative to base_ccy
    needed_ccys = {h["currency"] for h in holdings_list} | {bm_ccy}
    fx_tickers: List[str] = []
    for ccy in sorted(needed_ccys):
        if ccy != base_ccy:
            fx_tickers.append(f"{base_ccy}{ccy}=X")

    dl_list = list(dict.fromkeys(tickers + [bm_ticker] + fx_tickers))
    raw_df = yf.download(dl_list, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw_df, pd.Series):
        raw_df = raw_df.to_frame(name=dl_list[0])

    notices: List[str] = []
    valid_holdings: List[Dict[str, Any]] = []
    first_valid_dates: Dict[str, pd.Timestamp] = {}

    for h in holdings_list:
        t = h["ticker"]
        if t not in raw_df.columns or raw_df[t].dropna().shape[0] < 10:
            notices.append(f"Excluded ticker '{t}': insufficient or missing price history on Yahoo Finance.")
        else:
            valid_holdings.append(h)
            first_valid_dates[t] = raw_df[t].first_valid_index()

    if len(valid_holdings) < 2:
        raise ValueError("Fewer than 2 valid holdings remained after checking market data availability.")

    valid_tickers = [h["ticker"] for h in valid_holdings]

    # Check if a recently listed stock truncates the common history by > 60 trading days
    earliest_start = min(first_valid_dates.values())
    latest_start_ticker = max(first_valid_dates, key=first_valid_dates.get)
    latest_start_date = first_valid_dates[latest_start_ticker]
    if (latest_start_date - earliest_start).days > 90:
        notices.append(
            f"Common sample window starts on {latest_start_date.date()} because holding "
            f"'{latest_start_ticker}' has no price history prior to that date."
        )

    keep_cols = list(dict.fromkeys(valid_tickers + [bm_ticker] + fx_tickers))
    cleaned_data = raw_df[keep_cols].ffill(limit=max_ffill).dropna(how="any")

    if len(cleaned_data) < 30:
        raise ValueError(f"Only {len(cleaned_data)} overlapping trading days found across holdings.")

    prices_base = pd.DataFrame(index=cleaned_data.index)
    for h in valid_holdings:
        t = h["ticker"]
        prices_base[t] = _convert_series_to_base_ccy(cleaned_data[t], h["currency"], base_ccy, cleaned_data)

    bm_prices_base = _convert_series_to_base_ccy(cleaned_data[bm_ticker], bm_ccy, base_ccy, cleaned_data)

    # Calculate market values & weights according to sizing_mode ('shares', 'amount', or 'weight')
    sizing_mode = valid_holdings[0]["sizing_mode"]
    latest_prices = prices_base.iloc[-1]

    if sizing_mode == "shares":
        shares_s = pd.Series({h["ticker"]: float(h["sizing_value"]) for h in valid_holdings})
        market_values = shares_s * latest_prices
        total_nav = float(market_values.sum())
        weights = market_values / total_nav
    elif sizing_mode == "amount":
        market_values = pd.Series({h["ticker"]: float(h["sizing_value"]) for h in valid_holdings})
        total_nav = float(market_values.sum())
        weights = market_values / total_nav
    else:  # 'weight'
        raw_w = pd.Series({h["ticker"]: float(h["sizing_value"]) for h in valid_holdings})
        w_sum = float(raw_w.sum())
        if not np.isclose(w_sum, 1.0, atol=1e-3) and not np.isclose(w_sum, 100.0, atol=1e-1):
            notices.append(f"Input weights summed to {w_sum:.4f} and were normalized to 100.0%.")
        weights = raw_w / w_sum
        total_nav = 100000.0  # Standardized €/$100,000 notional base for weight-defined portfolios

    # Build metadata lookup indexed by final ticker (or alias when anonymized)
    holding_meta: Dict[str, Dict[str, Any]] = {}
    for h in valid_holdings:
        key = h["alias"] if anonymize else h["ticker"]
        holding_meta[key] = {
            "raw_ticker": h["ticker"],
            "currency": h["currency"],
            "sector": h["sector"],
            "region": h["region"],
            "alias": h["alias"],
        }

    raw_weights_by_ticker = weights.copy()

    if anonymize:
        alias_map = {h["ticker"]: h["alias"] for h in valid_holdings}
        prices_base = prices_base.rename(columns=alias_map)
        weights = weights.rename(index=alias_map)
        total_nav = 100000.0

    log_returns = np.log(prices_base / prices_base.shift(1)).dropna(how="any")
    bm_log_returns = np.log(bm_prices_base / bm_prices_base.shift(1)).loc[log_returns.index]

    return {
        "prices_base": prices_base,
        "log_returns": log_returns,
        "weights": weights,
        "raw_weights_by_ticker": raw_weights_by_ticker,
        "total_nav": total_nav,
        "base_currency": base_ccy,
        "bm_prices_base": bm_prices_base,
        "bm_log_returns": bm_log_returns,
        "holding_meta": holding_meta,
        "notices": notices,
        "config": cfg,
    }


def fetch_and_clean_eur_prices(
    portfolio: Dict[str, Dict[str, object]],
    benchmark_ticker: str = "IWDA.AS",
    start_date: str = "2021-01-01",
    end_date: str | None = None,
    max_ffill_days: int = 5,
    anonymize: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, float, pd.Series, pd.Series]:
    """Backward-compatible wrapper converting legacy dict format into v1.2 config pipeline."""
    holdings = []
    for t, meta in portfolio.items():
        holdings.append({
            "ticker": t,
            "shares": float(meta.get("shares", 10)),
            "currency": str(meta.get("currency", "EUR")),
            "sector": str(meta.get("sector", "Unclassified")),
            "region": str(meta.get("region", "Global")),
            "alias": str(meta.get("alias", t)),
        })
    temp_cfg = validate_config_dict({
        "portfolio_name": "Legacy Dict Portfolio",
        "settings": {
            "base_currency": "EUR",
            "start_date": start_date,
            "end_date": end_date,
            "max_ffill_days": max_ffill_days,
            "benchmark_ticker": benchmark_ticker,
            "benchmark_currency": "EUR",
        },
        "holdings": holdings,
    })
    res = fetch_portfolio_data_from_config(temp_cfg, anonymize=anonymize)
    return (
        res["prices_base"],
        res["log_returns"],
        res["weights"],
        res["total_nav"],
        res["bm_prices_base"],
        res["bm_log_returns"],
    )