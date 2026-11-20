"""
Module 4: Portfolio-Agnostic Backtesting & Dynamic Crisis Stress Testing (v1.2)
Automatically handles short-history portfolios (<500 days), detects missing pre-IPO
crisis data per ticker, and dynamically resolves and labels benchmark crisis proxies.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


def run_rolling_backtest_and_kupiec(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    estimation_window: int = 252,
    min_total_days: int = 500,
    alpha: float = 0.95,
) -> Dict[str, Any]:
    """
    Runs a strict out-of-sample rolling 1-day VaR backtest and Kupiec (1995) POF test.
    If total observations < min_total_days (default 500), disables backtesting gracefully
    with an explanatory message instead of raising an error.
    """
    simple_rets = (np.exp(log_returns) - 1.0) @ weights
    n_total = len(simple_rets)

    if n_total < max(min_total_days, estimation_window + 30):
        return {
            "enabled": False,
            "total_sample_days": n_total,
            "min_required_days": min_total_days,
            "reason": (
                f"Rolling {estimation_window}-day out-of-sample backtest disabled: common portfolio history "
                f"contains {n_total} trading days, which is below the {min_total_days}-day minimum required "
                f"for statistically meaningful Kupiec Proportion-of-Failures testing."
            ),
            "dates": pd.Index([]),
            "realized_returns": np.array([]),
            "hist_var_series": np.array([]),
            "param_var_series": np.array([]),
            "kupiec_historical": None,
            "kupiec_parametric": None,
        }

    realized_returns = simple_rets.iloc[estimation_window:].values
    dates = simple_rets.index[estimation_window:]
    n_obs = len(realized_returns)

    z_alpha = stats.norm.ppf(alpha)
    hist_var_forecasts = np.zeros(n_obs)
    param_var_forecasts = np.zeros(n_obs)

    for i in range(n_obs):
        window_slice = simple_rets.iloc[i : i + estimation_window].values
        hist_var_forecasts[i] = -np.percentile(window_slice, (1.0 - alpha) * 100.0)
        param_var_forecasts[i] = z_alpha * np.std(window_slice, ddof=1)

    def _kupiec_pof_test(var_series: np.ndarray) -> Dict[str, float | bool]:
        exceedances = realized_returns < (-var_series)
        x = int(np.sum(exceedances))
        p_expected = 1.0 - alpha
        p_hat = x / n_obs

        if x == 0:
            lr_pof = -2.0 * n_obs * np.log(1.0 - p_expected)
        elif x == n_obs:
            lr_pof = -2.0 * n_obs * np.log(p_expected)
        else:
            ll_null = (n_obs - x) * np.log(1.0 - p_expected) + x * np.log(p_expected)
            ll_alt = (n_obs - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
            lr_pof = float(-2.0 * (ll_null - ll_alt))

        p_value = float(1.0 - stats.chi2.cdf(lr_pof, df=1))
        return {
            "total_days": n_obs,
            "expected_breaches": round(n_obs * p_expected, 1),
            "actual_breaches": x,
            "expected_rate_pct": p_expected * 100.0,
            "actual_rate_pct": p_hat * 100.0,
            "lr_pof_stat": lr_pof,
            "p_value": p_value,
            "reject_null_95": bool(p_value < 0.05),
        }

    return {
        "enabled": True,
        "total_sample_days": n_total,
        "min_required_days": min_total_days,
        "reason": "",
        "dates": dates,
        "realized_returns": realized_returns,
        "hist_var_series": hist_var_forecasts,
        "param_var_series": param_var_forecasts,
        "kupiec_historical": _kupiec_pof_test(hist_var_forecasts),
        "kupiec_parametric": _kupiec_pof_test(param_var_forecasts),
    }


def run_crisis_stress_tests(
    weights: pd.Series,
    total_nav_eur: float,
    log_returns: pd.DataFrame,
    raw_weights_by_ticker: pd.Series | None = None,
    holding_meta: Dict[str, Dict[str, Any]] | None = None,
    config: Dict[str, Any] | None = None,
    top_risk_sector: str | None = None,
    anonymize: bool = False,
) -> pd.DataFrame:
    """
    Replays 3 historical crises + configurable hypothetical factor scenarios.
    - Auto-detects missing pre-IPO history per ticker during each crisis window and routes
      to sector ETF proxies configured in 'sector_proxy_map'.
    - Detects whether the primary benchmark has data in each crisis window; if not, uses
      'benchmark_crisis_proxy' (e.g., SPY adjusted close in base currency) and dynamically
      prints the benchmark's first available date read from the data.
    """
    w_tickers = raw_weights_by_ticker if raw_weights_by_ticker is not None else weights
    settings = (config or {}).get("settings", {})
    base_ccy = settings.get("base_currency", "EUR")
    bm_ticker = settings.get("benchmark_ticker", "IWDA.AS")
    bm_ccy = settings.get("benchmark_currency", base_ccy)
    bm_proxy = settings.get("benchmark_crisis_proxy", "SPY")
    bm_proxy_ccy = settings.get("benchmark_crisis_proxy_currency", "USD")

    default_proxy_map = {
        "Information Technology": {"ticker": "XLK", "currency": "USD"},
        "Communication Services": {"ticker": "XLC", "currency": "USD"},
        "Health Care": {"ticker": "XLV", "currency": "USD"},
        "Financials": {"ticker": "XLF", "currency": "USD"},
        "Consumer Discretionary": {"ticker": "EXV5.DE", "currency": "EUR"},
        "Consumer Staples": {"ticker": "XLP", "currency": "USD"},
        "Industrials": {"ticker": "XLI", "currency": "USD"},
        "Energy": {"ticker": "XLE", "currency": "USD"},
    }
    sector_proxy_map = {**default_proxy_map, **((config or {}).get("sector_proxy_map") or {})}

    # Map each raw ticker to its currency & sector
    ticker_info: Dict[str, Dict[str, str]] = {}
    if holding_meta:
        for alias_or_t, m in holding_meta.items():
            raw_t = m.get("raw_ticker", alias_or_t)
            ticker_info[raw_t] = {
                "currency": m.get("currency", "USD"),
                "sector": m.get("sector", "Information Technology"),
                "display": m.get("alias", raw_t) if anonymize else raw_t,
            }
    else:
        for t in w_tickers.index:
            is_eur = t.endswith((".BR", ".PA", ".MI", ".MC", ".AS", ".DE"))
            ticker_info[t] = {
                "currency": "EUR" if is_eur else "USD",
                "sector": "Information Technology",
                "display": "Position_Worst" if anonymize else t,
            }

    # Read actual earliest available date of primary benchmark from Yahoo Finance
    bm_hist_check = yf.download(bm_ticker, start="2005-01-01", progress=False)["Close"]
    if isinstance(bm_hist_check, pd.DataFrame):
        bm_hist_check = bm_hist_check.iloc[:, 0]
    bm_first_date_str = str(bm_hist_check.dropna().index[0].date()) if len(bm_hist_check.dropna()) > 0 else "post-crisis"

    crisis_windows = {
        "1. 2008 Global Financial Crisis (Sep 2008 - Mar 2009 Trough)": ("2008-09-15", "2009-03-09"),
        "2. COVID-19 Liquidity Crash (Feb 19 - Mar 23, 2020)":          ("2020-02-19", "2020-03-23"),
        "3. 2022 Inflation & Rate Tightening (Jan - Oct 2022)":         ("2022-01-03", "2022-10-14"),
    }

    all_tickers = list(w_tickers.index)
    proxy_tickers = list({v["ticker"] for v in sector_proxy_map.values()})
    fx_needed = {info["currency"] for info in ticker_info.values()} | {bm_ccy, bm_proxy_ccy} | {
        v["currency"] for v in sector_proxy_map.values()
    }
    fx_pairs = [f"{base_ccy}{c}=X" for c in fx_needed if c != base_ccy]

    def _fx_return(ccy: str, raw_df: pd.DataFrame) -> float:
        if ccy == base_ccy:
            return 0.0
        pair = f"{base_ccy}{ccy}=X"
        if pair in raw_df.columns and raw_df[pair].dropna().shape[0] >= 2:
            s = raw_df[pair].dropna()
            return float((s.iloc[-1] / s.iloc[0]) - 1.0)
        return 0.0

    rows: List[Dict[str, Any]] = []
    for name, (start, end) in crisis_windows.items():
        dl_list = list(dict.fromkeys(all_tickers + proxy_tickers + [bm_ticker, bm_proxy] + fx_pairs))
        raw = yf.download(dl_list, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=dl_list[0])
        raw = raw.ffill().bfill()

        asset_cum_returns = {}
        actual_weight_covered = 0.0
        n_proxied = 0

        for t in all_tickers:
            info = ticker_info[t]
            has_actual = (t in raw.columns) and (raw[t].dropna().shape[0] >= 5) and (raw[t].iloc[0] > 0)
            if has_actual:
                series = raw[t].dropna()
                asset_ccy = info["currency"]
                actual_weight_covered += float(w_tickers[t])
            else:
                n_proxied += 1
                p_cfg = sector_proxy_map.get(info["sector"], {"ticker": "SPY", "currency": "USD"})
                p_sym = p_cfg["ticker"] if p_cfg["ticker"] in raw.columns and raw[p_cfg["ticker"]].dropna().shape[0] >= 2 else bm_proxy
                series = raw[p_sym].dropna()
                asset_ccy = p_cfg["currency"]

            local_ret = float((series.iloc[-1] / series.iloc[0]) - 1.0)
            fx_chg = _fx_return(asset_ccy, raw)
            base_ret = local_ret if asset_ccy == base_ccy else ((1.0 + local_ret) / (1.0 + fx_chg)) - 1.0
            asset_cum_returns[t] = float(base_ret)

        # Benchmark crisis return: auto-detect if primary benchmark has data in this window
        bm_has_data = (
            (bm_ticker in raw.columns)
            and (raw[bm_ticker].dropna().shape[0] >= 5)
            and (str(raw[bm_ticker].dropna().index[0].date()) <= start[:7] + "-31")
            and (bm_first_date_str <= start)
        )
        if bm_has_data:
            bm_s = raw[bm_ticker].dropna()
            bm_loc = float((bm_s.iloc[-1] / bm_s.iloc[0]) - 1.0)
            bm_fx = _fx_return(bm_ccy, raw)
            bm_ret = bm_loc if bm_ccy == base_ccy else ((1.0 + bm_loc) / (1.0 + bm_fx)) - 1.0
            bm_label = f"{bm_ret * 100.0:.2f}% ({bm_ticker})"
        else:
            p_s = raw[bm_proxy].dropna()
            p_loc = float((p_s.iloc[-1] / p_s.iloc[0]) - 1.0)
            p_fx = _fx_return(bm_proxy_ccy, raw)
            bm_ret = p_loc if bm_proxy_ccy == base_ccy else ((1.0 + p_loc) / (1.0 + p_fx)) - 1.0
            bm_label = (
                f"{bm_ret * 100.0:.2f}% (via US-only proxy {bm_proxy} in {base_ccy}; "
                f"{bm_ticker} data begins {bm_first_date_str})"
            )

        ret_series = pd.Series(asset_cum_returns)
        port_crisis_ret = float((ret_series * w_tickers).sum())
        worst_t = str(ret_series.idxmin())
        worst_disp = ticker_info[worst_t]["display"]
        proxy_pct = (1.0 - actual_weight_covered) * 100.0

        rows.append({
            "Scenario": name,
            "Portfolio Return (%)": round(port_crisis_ret * 100.0, 2),
            "Benchmark Return": bm_label,
            f"Estimated P&L ({base_ccy})": round(port_crisis_ret * total_nav_eur, 2),
            "Data Coverage": f"{actual_weight_covered*100:.1f}% Actual ({len(all_tickers)-n_proxied}/{len(all_tickers)}) | {proxy_pct:.1f}% Proxy ({n_proxied}/{len(all_tickers)})",
            "Worst Performer": f"{worst_disp} ({ret_series.min()*100:.1f}%)",
        })

    # Configurable Hypothetical Scenarios (defaults to shocking top-risk sector + non-base FX)
    scenarios_cfg = (config or {}).get("stress_scenarios") or [
        {
            "name": "auto_top_sector",
            "target_sector": "AUTO_TOP_RISK_SECTOR",
            "sector_shock": -0.25,
            "other_equity_shock": -0.08,
            "non_base_fx_shock": -0.08,
        }
    ]

    # Determine dominant sector dynamically if not provided
    if not top_risk_sector:
        sec_weights: Dict[str, float] = {}
        for t, w_val in w_tickers.items():
            sec = ticker_info[t]["sector"]
            sec_weights[sec] = sec_weights.get(sec, 0.0) + float(w_val)
        top_risk_sector = max(sec_weights, key=sec_weights.get)

    for idx, sc in enumerate(scenarios_cfg, start=len(rows) + 1):
        target_sec = top_risk_sector if sc.get("target_sector") == "AUTO_TOP_RISK_SECTOR" else sc.get("target_sector", top_risk_sector)
        sec_shock = float(sc.get("sector_shock", -0.25))
        oth_shock = float(sc.get("other_equity_shock", -0.08))
        fx_shock_cfg = float(sc.get("non_base_fx_shock", -0.08))

        hyp_returns = {}
        has_non_base = False
        for t in all_tickers:
            info = ticker_info[t]
            is_foreign = info["currency"] != base_ccy
            if is_foreign:
                has_non_base = True
            fx_s = fx_shock_cfg if is_foreign else 0.0
            eq_s = sec_shock if info["sector"] == target_sec else oth_shock
            hyp_returns[t] = (1.0 + eq_s) * (1.0 + fx_s) - 1.0

        hyp_series = pd.Series(hyp_returns)
        hyp_port_ret = float((hyp_series * w_tickers).sum())
        fx_clause = f" + Non-{base_ccy} FX ({fx_shock_cfg*100:+.0f}%)" if has_non_base else f" (Single-Currency {base_ccy})"
        sc_title = f"{idx}. Hypothetical: {target_sec} Sector Shock ({sec_shock*100:+.0f}%){fx_clause}"

        bm_hyp_ret = (0.70 * ((1.0 - 0.15) * (1.0 + (fx_shock_cfg if has_non_base else 0.0)) - 1.0)) + (0.30 * (-0.08))
        worst_hyp_t = str(hyp_series.idxmin())

        rows.append({
            "Scenario": sc_title,
            "Portfolio Return (%)": round(hyp_port_ret * 100.0, 2),
            "Benchmark Return": f"{bm_hyp_ret * 100.0:.2f}% (Factor-approximated)",
            f"Estimated P&L ({base_ccy})": round(hyp_port_ret * total_nav_eur, 2),
            "Data Coverage": "100.0% Factor Shock",
            "Worst Performer": f"{ticker_info[worst_hyp_t]['display']} ({hyp_series.min()*100:.1f}%)",
        })

    return pd.DataFrame(rows)