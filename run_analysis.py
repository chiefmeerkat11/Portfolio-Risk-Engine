"""
Master Execution Script: Portfolio-Agnostic Multi-Currency Risk & Attribution Engine (v1.2)
Usage:
  Public demo portfolio:   python run_analysis.py
  Custom config file:      python run_analysis.py --config config/portfolio_private.json
  Anonymized public mode:  python run_analysis.py --config config/portfolio_private.json --anonymize
"""
import argparse
import base64
import webbrowser
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config_loader import load_config_file, resolve_output_dir
from src.data_pipeline import fetch_portfolio_data_from_config
from src.risk_models import (
    historical_simulation_var,
    parametric_vcov_var,
    monte_carlo_var,
    compute_portfolio_vs_benchmark_summary,
    simulate_horizon_and_drift_sensitivity,
)
from src.attribution import (
    compute_risk_attribution,
    compute_group_exposures,
    compute_max_drawdown,
)
from src.validation import run_rolling_backtest_and_kupiec, run_crisis_stress_tests
from src.advisory import evaluate_advisory_guidelines


def _abbr(term: str, definition: str) -> str:
    """Wraps a financial term in an HTML <abbr> tooltip on first use for non-specialists."""
    return f'<abbr title="{definition}" style="text-decoration: underline dotted; cursor: help;">{term}</abbr>'


def build_and_open_browser_report(
    bundle: Dict[str, Any],
    summary_stats: Dict[str, Any],
    bm_comp_table: pd.DataFrame,
    model_comp_df: pd.DataFrame,
    horizon_drift_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    ccy_df: pd.DataFrame,
    attr_df: pd.DataFrame,
    dd_info: Dict[str, Any],
    bt_info: Dict[str, Any],
    stress_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    rebal_meta: Dict[str, Any],
    fitted_df: float,
    anonymize: bool,
    chart_path: Path,
    output_html: Path,
) -> None:
    """Generates a 100% dynamically populated, print-friendly HTML report and opens it."""
    cfg = bundle["config"]
    settings = cfg["settings"]
    base_ccy = bundle["base_currency"]
    bm_ticker = settings["benchmark_ticker"]
    rf_rate = float(settings["risk_free_rate"])
    eq_erp = float(settings["equilibrium_erp"])
    total_nav = bundle["total_nav"]
    log_returns = bundle["log_returns"]
    n_holdings = len(attr_df)
    start_dt = str(log_returns.index[0].date())
    end_dt = str(log_returns.index[-1].date())
    start_yr = log_returns.index[0].year
    end_yr = log_returns.index[-1].year

    img_b64 = base64.b64encode(chart_path.read_bytes()).decode("utf-8")

    # Extract exact table numbers so narrative text matches tables 100%
    def _parse_num(val: object) -> float:
        return float(str(val).replace("%", "").replace(base_ccy, "").replace(",", "").replace("-", "").strip())

    # Extract exact table numbers so narrative text matches tables 100%
    t_row_name = [idx for idx in model_comp_df.index if "Student-t" in idx][0]
    p_var95_1d_pct = _parse_num(model_comp_df.loc["Parametric (Variance-Covariance)", "1d 95% VaR (%)"])
    p_var95_1d_ccy = _parse_num(model_comp_df.loc["Parametric (Variance-Covariance)", f"1d 95% VaR ({base_ccy})"])
    t_var95_1d_pct = _parse_num(model_comp_df.loc[t_row_name, "1d 95% VaR (%)"])

    t_cvar99_1d_pct = _parse_num(model_comp_df.loc[t_row_name, "1d 99% CVaR (%)"])
    t_cvar99_1d_ccy = _parse_num(model_comp_df.loc[t_row_name, f"1d 99% CVaR ({base_ccy})"])
    n_cvar99_1d_pct = _parse_num(model_comp_df.loc["Monte Carlo (Gaussian)", "1d 99% CVaR (%)"])
    n_cvar99_1d_ccy = _parse_num(model_comp_df.loc["Monte Carlo (Gaussian)", f"1d 99% CVaR ({base_ccy})"])
    tail_prem_pct = ((t_cvar99_1d_pct / n_cvar99_1d_pct) - 1.0) * 100.0

    # 10-Day metrics (Option B: 95% VaR = Parametric, 99% CVaR = Student-t, both sub-labeled)
    p_var95_10d_pct = summary_stats["port_param_10d"]["VaR_95"] * 100.0
    p_var95_10d_ccy = summary_stats["port_param_10d"]["VaR_95"] * total_nav
    t_var95_10d_pct = summary_stats["port_t_10d"]["VaR_95"] * 100.0

    t_cvar99_10d_pct = summary_stats["port_t_10d"]["CVaR_99"] * 100.0
    t_cvar99_10d_ccy = summary_stats["port_t_10d"]["CVaR_99"] * total_nav
    p_cvar99_10d_pct = summary_stats["port_param_10d"]["CVaR_99"] * 100.0

    # Dynamic top/bottom contributors & sectors
    top_k_disp = min(3, n_holdings)
    top_risk_str = ", ".join([
        f"<code>{idx}</code> ({row['Risk Contribution (%)']:.1f}% risk vs. {row['Weight (%)']:.1f}% weight)"
        for idx, row in attr_df.head(top_k_disp).iterrows()
    ])
    low_div_df = attr_df.sort_values(by="Risk / Weight Ratio", ascending=True).head(top_k_disp)
    low_div_str = ", ".join([
        f"<code>{idx}</code> ({row['Risk / Weight Ratio']:.2f}x ratio)"
        for idx, row in low_div_df.iterrows()
    ])

    top_sector = str(sector_df.index[0])
    top_sector_rc = float(sector_df.iloc[0]["Risk Contribution (%)"])
    top_sector_w = float(sector_df.iloc[0]["Weight (%)"])
    low_sector = str(sector_df.index[-1])
    low_sector_rc = float(sector_df.iloc[-1]["Risk Contribution (%)"])

    if len(sector_df) == 1:
        sector_takeaway = f"100.0% of portfolio capital and risk resides in a single sector ({top_sector})."
    else:
        sector_takeaway = (
            f"The largest sector risk driver is <strong>{top_sector}</strong> ({top_sector_rc:.1f}% of risk vs. "
            f"{top_sector_w:.1f}% weight), while <strong>{low_sector}</strong> contributes the smallest share ({low_sector_rc:.1f}%)."
        )

    base_ccy_w = float(ccy_df["Weight (%)"].get(base_ccy, 0.0))
    base_ccy_rc = float(ccy_df["Risk Contribution (%)"].get(base_ccy, 0.0))
    if len(ccy_df) == 1 and base_ccy in ccy_df.index:
        ccy_takeaway = f"100.0% of portfolio capital and risk is denominated in the base currency ({base_ccy}); there is no foreign exchange exposure."
    else:
        ccy_takeaway = (
            f"Foreign-currency holdings represent <strong>{100.0 - base_ccy_w:.1f}%</strong> of capital weight and "
            f"generate <strong>{100.0 - base_ccy_rc:.1f}%</strong> of total portfolio risk, whereas {base_ccy}-denominated "
            f"holdings represent <strong>{base_ccy_w:.1f}%</strong> of capital and <strong>{base_ccy_rc:.1f}%</strong> of risk."
        )

    enb_cap = summary_stats["enb_capital"]
    rc_decimal = attr_df["Risk Contribution (%)"].values / 100.0
    enb_risk = float(1.0 / np.sum(rc_decimal ** 2))
    div_ratio = summary_stats["diversification_ratio"]
    port_vol = summary_stats["port_ann_vol_pct"]
    bm_vol = summary_stats["bm_ann_vol_pct"]
    vol_comp_word = "higher than" if port_vol >= bm_vol else "lower than"

    # Backtest KPI & narrative
    if bt_info["enabled"]:
        kp = bt_info["kupiec_parametric"]
        kh = bt_info["kupiec_historical"]
        bt_kpi_val = f"p = {kp['p_value']:.3f}"
        bt_kpi_sub = f"{kp['actual_breaches']} breaches vs {kp['expected_breaches']:.1f} exp [Parametric]"
        bt_takeaway_bullet = (
            f"<strong>Out-of-Sample Backtest Calibration:</strong> Over <strong>{kp['total_days']}</strong> out-of-sample "
            f"trading days, 1-day 95% VaR [Parametric] recorded <strong>{kp['actual_breaches']}</strong> breaches "
            f"({kp['actual_rate_pct']:.2f}%, Kupiec <strong>p = {kp['p_value']:.3f}</strong>) and Historical VaR recorded "
            f"<strong>{kh['actual_breaches']}</strong> breaches ({kh['actual_rate_pct']:.2f}%, <strong>p = {kh['p_value']:.3f}</strong>)."
        )
        kupiec_df = pd.DataFrame([
            {
                "Backtested Model (1-Day 95% VaR)": "Historical Simulation",
                "Out-of-Sample Days": kh["total_days"],
                "Expected Breaches (5.0%)": f"{kh['expected_breaches']:.1f}",
                "Actual Breaches": kh["actual_breaches"],
                "Actual Breach Rate (%)": f"{kh['actual_rate_pct']:.2f}%",
                "Kupiec LR Stat": f"{kh['lr_pof_stat']:.3f}",
                "p-value": f"{kh['p_value']:.3f}",
                "Status (Threshold: p >= 0.05)": "PASS (Unconditional Coverage)" if not kh["reject_null_95"] else "REJECT (p < 0.05)",
            },
            {
                "Backtested Model (1-Day 95% VaR)": "Parametric (Variance-Covariance)",
                "Out-of-Sample Days": kp["total_days"],
                "Expected Breaches (5.0%)": f"{kp['expected_breaches']:.1f}",
                "Actual Breaches": kp["actual_breaches"],
                "Actual Breach Rate (%)": f"{kp['actual_rate_pct']:.2f}%",
                "Kupiec LR Stat": f"{kp['lr_pof_stat']:.3f}",
                "p-value": f"{kp['p_value']:.3f}",
                "Status (Threshold: p >= 0.05)": "PASS (Unconditional Coverage)" if not kp["reject_null_95"] else "REJECT (p < 0.05)",
            },
        ])
        backtest_html_block = kupiec_df.to_html(index=False, border=0)
    else:
        bt_kpi_val = "Disabled"
        bt_kpi_sub = f"{bt_info['total_sample_days']}d < {bt_info['min_required_days']}d min"
        bt_takeaway_bullet = f"<strong>Out-of-Sample Backtest Status:</strong> {bt_info['reason']}"
        backtest_html_block = f'<div class="guide-box" style="border-left-color:#b45309; background:#fffbeb;"><strong>Notice:</strong> {bt_info["reason"]}</div>'

    # Dynamic stress figures
    s_2008_ret = float(stress_df.iloc[0]["Portfolio Return (%)"])
    s_2008_ccy = float(stress_df.iloc[0][f"Estimated P&L ({base_ccy})"])
    s_2020_ret = float(stress_df.iloc[1]["Portfolio Return (%)"])
    s_2020_ccy = float(stress_df.iloc[1][f"Estimated P&L ({base_ccy})"])
    s_2022_ret = float(stress_df.iloc[2]["Portfolio Return (%)"])
    s_2022_ccy = float(stress_df.iloc[2][f"Estimated P&L ({base_ccy})"])
    s_hyp_name = str(stress_df.iloc[3]["Scenario"])
    s_hyp_ret = float(stress_df.iloc[3]["Portfolio Return (%)"])
    s_hyp_ccy = float(stress_df.iloc[3][f"Estimated P&L ({base_ccy})"])

    notices_html = ""
    if bundle.get("notices"):
        items = "".join([f"<li>{n}</li>" for n in bundle["notices"]])
        notices_html = f'<div class="notice-banner"><strong>Data & Configuration Notices:</strong><ul>{items}</ul></div>'

    mode_badge = "ANONYMIZED PUBLIC VIEW (NAV SCALED TO 100,000)" if anonymize else ("PRIVATE PORTFOLIO REPORT" if cfg["is_private"] else "PUBLIC ILLUSTRATIVE DEMO")
    n_flags = rebal_meta["n_triggered"]
    advisory_headline = (
        f"{n_flags} of 8 governance checks flagged for committee discussion."
        if n_flags > 0
        else "All 8 governance checks passed without triggering review flags."
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{cfg['portfolio_name']} - Risk & Attribution Report</title>
    <style>
        :root {{
            --ink: #0f172a;
            --muted: #475569;
            --border: #cbd5e1;
            --bg: #f1f5f9;
            --card: #ffffff;
            --accent: #1e40af;
            --danger: #b91c1c;
            --ok: #15803d;
        }}
        body {{ font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; margin: 0; padding: 24px; background: var(--bg); color: var(--ink); line-height: 1.52; }}
        .page {{ max-width: 1360px; margin: auto; background: var(--card); padding: 36px 44px; border-radius: 8px; box-shadow: 0 4px 20px rgba(15,23,42,0.08); }}
        .header-bar {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid var(--ink); padding-bottom: 12px; margin-bottom: 10px; }}
        h1 {{ margin: 0; font-size: 23px; font-weight: 800; color: var(--ink); letter-spacing: -0.3px; }}
        .badge-mode {{ font-size: 11px; font-weight: 700; text-transform: uppercase; background: #e2e8f0; color: var(--ink); padding: 4px 10px; border-radius: 4px; }}
        .meta-line {{ font-size: 13px; color: var(--muted); margin-bottom: 14px; }}
        .disclaimer-bar {{ background: #f8fafc; border: 1px solid var(--border); padding: 8px 14px; font-size: 12px; color: var(--muted); border-radius: 4px; margin-bottom: 18px; }}
        .notice-banner {{ background: #fffbeb; border: 1px solid #fcd34d; border-left: 4px solid #d97706; padding: 10px 14px; font-size: 12.5px; border-radius: 4px; margin-bottom: 18px; }}
        .notice-banner ul {{ margin: 4px 0 0 0; padding-left: 18px; }}
        h2 {{ color: var(--ink); font-size: 17px; font-weight: 700; margin-top: 34px; margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.4px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 16px 0 22px 0; }}
        .kpi {{ background: #f8fafc; border: 1px solid var(--border); border-top: 4px solid var(--accent); padding: 12px 14px; border-radius: 6px; }}
        .kpi.danger {{ border-top-color: var(--danger); }}
        .kpi.ok {{ border-top-color: var(--ok); }}
        .kpi-label {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); font-weight: 700; }}
        .kpi-value {{ font-size: 19px; font-weight: 800; color: var(--ink); margin-top: 4px; }}
        .kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 3px; }}
        .read-first-box {{ background: #f8fafc; border: 1px solid var(--border); border-left: 5px solid var(--accent); padding: 16px 20px; border-radius: 6px; margin-bottom: 24px; font-size: 13.5px; }}
        .read-first-box h3 {{ margin: 0 0 8px 0; font-size: 14px; text-transform: uppercase; color: var(--accent); letter-spacing: 0.4px; }}
        .read-first-box ul {{ margin: 6px 0 0 0; padding-left: 20px; }}
        .read-first-box li {{ margin-bottom: 6px; }}
        .guide-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #475569; padding: 10px 14px; margin: 8px 0 14px 0; font-size: 13px; border-radius: 0 4px 4px 0; }}
        .guide-box p {{ margin: 3px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 18px; font-size: 12.5px; }}
        th {{ background: #1e293b; color: #ffffff; padding: 8px 10px; text-align: right; font-weight: 600; }}
        th:first-child, td:first-child {{ text-align: left; font-weight: 600; }}
        td {{ padding: 7px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
        tr:nth-child(even) {{ background: #f8fafc; }}
        .advisory-table td:nth-child(2), .advisory-table td:nth-child(4), .advisory-table td:nth-child(5),
        .advisory-table th:nth-child(2), .advisory-table th:nth-child(4), .advisory-table th:nth-child(5) {{ text-align: left; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .chart-box {{ text-align: center; margin: 14px 0 24px 0; }}
        .chart-box img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border); }}
        .lim-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; font-size: 12.5px; }}
        .lim-card {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px 14px; border-radius: 6px; }}
        @media print {{
            body {{ background: #ffffff; padding: 0; }}
            .page {{ box-shadow: none; padding: 10px; max-width: 100%; }}
            table, .kpi-grid, .read-first-box, .guide-box, .lim-card {{ page-break-inside: avoid; }}
            h2 {{ page-break-after: avoid; }}
        }}
    </style>
</head>
<body>
<div class="page">
    <div class="header-bar">
        <h1>{cfg['portfolio_name']} — Risk & Attribution Report</h1>
        <span class="badge-mode">{mode_badge}</span>
    </div>
    <div class="meta-line">
        <strong>Portfolio Descriptor:</strong> {n_holdings} Equity Positions | Base Currency: {base_ccy} | Window: {start_dt} to {end_dt} ({len(log_returns)} trading days) | Benchmark: <code>{bm_ticker}</code> ({base_ccy})
    </div>
    <div class="disclaimer-bar">
        <strong>Governance & Educational Notice:</strong> {cfg['illustrative_disclaimer']} All advisory flags and rebalancing simulations are rule-based outputs derived from historical return covariance and do not constitute personalized investment advice or buy/sell instructions.
    </div>
    {notices_html}

    <div class="kpi-grid">
        <div class="kpi">
            <div class="kpi-label">Portfolio NAV ({base_ccy})</div>
            <div class="kpi-value">{base_ccy} {total_nav:,.2f}</div>
            <div class="kpi-sub">Effective Bets: {enb_cap:.1f} cap / {enb_risk:.1f} risk (of {n_holdings})</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">10-Day 95% VaR [Parametric]</div>
            <div class="kpi-value">-{p_var95_10d_pct:.2f}%</div>
            <div class="kpi-sub">-{base_ccy} {p_var95_10d_ccy:,.0f} | [Student-t]: -{t_var95_10d_pct:.2f}%</div>
     </div>
     <div class="kpi danger">
         <div class="kpi-label">1-Day 99% CVaR [Student-t]</div>
         <div class="kpi-value">-{t_cvar99_1d_pct:.2f}%</div>
         <div class="kpi-sub">-{base_ccy} {t_cvar99_1d_ccy:,.2f} | [Parametric]: -{n_cvar99_1d_pct:.2f}% (-{base_ccy} {n_cvar99_1d_ccy:,.2f})</div>
     </div>
        <div class="kpi danger">
            <div class="kpi-label">Max Historical Drawdown</div>
            <div class="kpi-value">{dd_info['max_drawdown_pct']:.2f}%</div>
            <div class="kpi-sub">Benchmark ({bm_ticker}) Max DD: {dd_info['bm_max_drawdown_pct']:.2f}%</div>
        </div>
        <div class="kpi ok">
            <div class="kpi-label">Kupiec Backtest [Parametric]</div>
            <div class="kpi-value">{bt_kpi_val}</div>
            <div class="kpi-sub">{bt_kpi_sub}</div>
        </div>
    </div>

    <h2>1. Read This First: Executive Overview & Key Takeaways</h2>
    <div class="read-first-box">
        <p style="margin-top:0;"><strong>What this report is & how to read it:</strong> This diagnostic measures how much money the portfolio could lose during normal market swings versus severe crashes, identifies which holdings drive that risk, and tests how the portfolio behaves during historical crises. <strong>Section 2</strong> summarizes actionable governance takeaways; <strong>Sections 3–8</strong> provide the underlying statistical evidence. Hover over any dotted term for a plain-English definition.</p>
        <h3>Five Key Takeaways</h3>
        <ul>
            <li><strong>1. Overall Risk vs. Global Benchmark:</strong> The portfolio exhibits annualized volatility of <strong>{port_vol:.2f}%</strong> ({vol_comp_word} the <code>{bm_ticker}</code> benchmark at <strong>{bm_vol:.2f}%</strong>; Benchmark Beta = <strong>{summary_stats['port_beta_bm']:.2f}</strong>). Over a 2-week (10-trading-day) governance window, 95% {_abbr('Value at Risk (VaR)', 'The maximum loss not exceeded 95% of the time over the stated horizon.')} <strong>[Parametric]</strong> is <strong>-{p_var95_10d_pct:.2f}% ({base_ccy} {p_var95_10d_ccy:,.0f})</strong> (<strong>[Student-t]</strong>: -{t_var95_10d_pct:.2f}%).</li>
            <li><strong>2. Fat Tails Increase Extreme Crash Estimates:</strong> Fitting empirical daily returns yields a heavy-tailed {_abbr('Student-t distribution', 'A bell-shaped distribution with fatter tails that assigns higher probability to extreme market crashes.')} with <strong>df = {fitted_df:.1f}</strong>. While a Gaussian bell curve estimates 1-day 99% {_abbr('Expected Shortfall (CVaR)', 'The average loss on the worst 1% of days when the VaR threshold is breached.')} <strong>[Gaussian MC]</strong> at <strong>-{n_cvar99_1d_pct:.2f}% ({base_ccy} {n_cvar99_1d_ccy:,.2f})</strong>, the fat-tailed model <strong>[Student-t MC]</strong> estimates <strong>-{t_cvar99_1d_pct:.2f}% ({base_ccy} {t_cvar99_1d_ccy:,.2f})</strong>—a <strong>{tail_prem_pct:+.1f}%</strong> difference in tail severity.</li>
            <li><strong>3. Risk Concentration vs. Diversification:</strong> Capital across the {n_holdings} positions equals <strong>{enb_cap:.1f}</strong> effective capital bets, while cross-asset correlations compress {_abbr('Effective Risk Bets', '1 divided by the sum of squared risk contributions; measures how many independent bets drive portfolio risk.')} to <strong>{enb_risk:.1f}</strong>. The top risk contributors are {top_risk_str}, whereas the strongest portfolio stabilizers are {low_div_str}.</li>
            <li><strong>4. Historical Crisis Resilience &amp; Factor Vulnerability:</strong> Replaying the 2008–09 GFC produces a <strong>{s_2008_ret:.2f}% (-{base_ccy} {abs(s_2008_ccy):,.2f})</strong> drawdown, and the 2020 COVID crash produces <strong>{s_2020_ret:.2f}% (-{base_ccy} {abs(s_2020_ccy):,.2f})</strong>. In the custom stress test (<em>{s_hyp_name}</em>), the portfolio experiences an estimated <strong>{s_hyp_ret:.2f}% (-{base_ccy} {abs(s_hyp_ccy):,.2f})</strong> loss.</li>            
            <li><strong>5. Governance & Backtest Summary:</strong> {advisory_headline} {bt_takeaway_bullet}</li>
        </ul>
    </div>

    <h2>2. What the Numbers Suggest: Rule-Based Risk Governance & Rebalancing Impact</h2>
    <div class="guide-box">
        <p><strong>What this shows:</strong> An automated governance scorecard evaluating 8 risk-concentration and calibration rules, followed by a risk-only simulation showing the effect of trimming the top risk amplifiers into the top diversifiers.</p>
        <p><strong>How to read it:</strong> Default thresholds are illustrative portfolio-management rules of thumb (not regulatory or universal constants) scaled to portfolio size. Flags are derived from historical data on current holdings and serve as discussion prompts rather than buy/sell instructions.</p>
        <p><strong>What to take away:</strong> {advisory_headline} In the illustrative rebalance below, shifting <strong>{rebal_meta['moved_total_pct']:.2f}%</strong> of total portfolio weight from <em>{rebal_meta['trimmed_str']}</em> into <em>{rebal_meta['added_str']}</em> shows how marginal weight adjustments affect portfolio-wide tail risk.</p>
    </div>
    <table class="advisory-table">
        <thead>
            <tr>{''.join([f'<th>{col}</th>' for col in flags_df.columns])}</tr>
        </thead>
        <tbody>
            {''.join(['<tr>' + ''.join([f'<td>{val}</td>' for val in row]) + '</tr>' for row in flags_df.values])}
        </tbody>
    </table>

    <div class="guide-box" style="margin-top:16px;">
        <p><strong>Rebalancing Impact Simulator (Risk-Only Sensitivity):</strong> Recomputes portfolio volatility, VaR, and CVaR before and after reallocating <strong>{rebal_meta['moved_total_pct']:.2f}%</strong> weight (<strong>Trimmed:</strong> {rebal_meta['trimmed_str']} &rarr; <strong>Added:</strong> {rebal_meta['added_str']}). <em>Caveat: Evaluates historical risk only, ignores expected returns, valuation, and transaction costs, and assumes historical covariance remains unchanged.</em></p>
    </div>
    {rebal_meta['rebal_table'].to_html(index=False, border=0)}

    <h2>3. Benchmark-Relative Performance & Risk Reference Ranges (Portfolio vs. {bm_ticker})</h2>
    <div class="guide-box">
        <p><strong>What this shows:</strong> Side-by-side comparison of the portfolio against the configured {base_ccy} benchmark (<code>{bm_ticker}</code>) over the identical {summary_stats['n_years']:.1f}-year sample window ({start_yr}–{end_yr}).</p>
        <p><strong>How to read it:</strong> Because there is no universal "healthy" volatility or VaR number for equities, risk metrics are judged relative to the passive benchmark, while structural metrics (Diversification Ratio, Effective Bets) use mathematical baselines.</p>
        <p><strong>What to take away:</strong> Diversification reduces weighted-average standalone stock volatility from <strong>{summary_stats['weighted_avg_vol_pct']:.2f}%</strong> to <strong>{port_vol:.2f}%</strong> (Diversification Ratio = <strong>{div_ratio:.2f}x</strong>), while total volatility is {vol_comp_word} <code>{bm_ticker}</code> (<strong>{bm_vol:.2f}%</strong>).</p>
    </div>
    {bm_comp_table.to_html(index=False, border=0)}

    <h2>4. Annotated 6-Panel Visual Diagnostic Dashboard</h2>
    <div class="guide-box">
        <p><strong>What this shows:</strong> Six diagnostic views covering (A) 1-day Gaussian vs. Student-t tail density, (B) Capital Weight vs. Euler Risk Contribution, (C) 252-day rolling out-of-sample VaR backtest, (D) Portfolio vs. Benchmark underwater drawdown, (E) 1-year equilibrium Monte Carlo fan chart, and (F) 1-year drift sensitivity overlay.</p>
        <p><strong>How to read it:</strong> Callout boxes on each panel highlight the primary empirical finding directly from the plotted data.</p>
        <p><strong>What to take away:</strong> Panels A and B highlight extreme tail divergence beyond the 99th percentile and position-level risk concentration, while Panels C and D illustrate historical drawdown dynamics relative to <code>{bm_ticker}</code>.</p>
    </div>
    <div class="chart-box">
        <img src="data:image/png;base64,{img_b64}" alt="6-Panel Portfolio Risk Dashboard"/>
    </div>

    <h2>5. Daily Tail Risk Comparison Across Four Distributional Models (1-Day Horizon, Zero Drift)</h2>
    <div class="guide-box">
        <p><strong>What this shows:</strong> 1-day 95% and 99% VaR and Expected Shortfall (CVaR) in percentage of NAV and {base_ccy} across four models (every figure explicitly tagged with its underlying model).</p>
        <p><strong>How to read it:</strong> Compare Gaussian models (rows 2–3) against empirical Historical Simulation (row 1) and fat-tailed Student-t Monte Carlo (row 4).</p>
        <p><strong>What to take away:</strong> At the 95% threshold, 1-day VaR [Student-t] (<strong>{t_var95_1d_pct:.2f}%</strong>) is close to VaR [Parametric] (<strong>{p_var95_1d_pct:.2f}%</strong>) because a variance-matched leptokurtic distribution narrows in the shoulders while placing heavier probability mass in the extreme tails; at <strong>99% CVaR</strong>, [Student-t MC] reaches <strong>{t_cvar99_1d_pct:.2f}% ({base_ccy} {t_cvar99_1d_ccy:,.2f})</strong> versus <strong>{n_cvar99_1d_pct:.2f}% ({base_ccy} {n_cvar99_1d_ccy:,.2f})</strong> under [Gaussian MC].</p>
    </div>
    {model_comp_df.to_html(border=0)}

    <h2>6. Horizon Scaling & 1-Year Monte Carlo Drift Sensitivity Analysis</h2>
    <div class="guide-box">
        <p><strong>What this shows:</strong> How tail risk scales from 1 Day to 10 Trading Days (2-week governance horizon, zero drift) and 1 Year (252 trading days) across three drift assumptions plus a 2x volatility stress regime.</p>
        <p><strong>How to read it:</strong> Over short horizons (1d and 10d), standard risk practice sets drift μ = 0. Over 1 year, terminal wealth depends heavily on assumed drift; because a {summary_stats['n_years']:.1f}-year sample mean has a standard error of <strong>±{summary_stats['mean_se_pct']:.2f}% p.a.</strong> (σ / √T), historical drift is shown only as a sensitivity reference.</p>
        <p><strong>What to take away:</strong> Headline 10-Day 95% VaR [Parametric] is <strong>-{p_var95_10d_pct:.2f}% ({base_ccy} {p_var95_10d_ccy:,.0f})</strong> ([Student-t]: -{t_var95_10d_pct:.2f}%), and 10-Day 99% CVaR [Student-t] is <strong>-{t_cvar99_10d_pct:.2f}% ({base_ccy} {t_cvar99_10d_ccy:,.0f})</strong> ([Parametric]: -{p_cvar99_10d_pct:.2f}%).</p>
    </div>
    {horizon_drift_df.to_html(index=False, border=0)}

    <h2>7. Sector, Currency & Position-Level Euler Risk Attribution</h2>
    <div class="guide-box">
        <p><strong>What this shows:</strong> Top tables aggregate Capital Weight (%) vs. Euler Risk Contribution (%) by Sector and Currency; the bottom table partitions 100% of 1-day 95% VaR [Parametric] (<strong>{base_ccy} {attr_df[f'Component VaR (1d {base_ccy})'].sum():,.2f}</strong>) across all {n_holdings} positions.</p>
        <p><strong>How to read it:</strong> <strong>Benchmark Beta</strong> measures sensitivity to <code>{bm_ticker}</code>. <strong>Risk / Weight Ratio</strong> (equal to internal portfolio beta β<sub>i,p</sub>) divides Risk Contribution (%) by Weight (%): values &gt; 1.25x are Risk Amplifiers, 0.80x–1.25x are Core/Neutral, and &lt; 0.80x are Diversifiers.</p>
        <p><strong>What to take away:</strong> {sector_takeaway} {ccy_takeaway}</p>
    </div>
    <div class="two-col">
        <div>{sector_df.round(2).to_html(index=False, border=0)}</div>
        <div>{ccy_df.round(2).to_html(index=False, border=0)}</div>
    </div>
    {attr_df.round(2).to_html(index=False, border=0)}

    <h2>8. Out-of-Sample Rolling Backtest & Historical Crisis Stress Tests</h2>
    <div class="guide-box">
        <p><strong>What this shows:</strong> Top table evaluates out-of-sample 1-day 95% VaR calibration via the Kupiec (1995) Proportion of Failures likelihood-ratio test; bottom table replays historical crises and configurable factor shocks in {base_ccy}.</p>
        <p><strong>How to read it:</strong> A Kupiec p-value &ge; 0.05 passes unconditional coverage. In the stress table, <em>Data Coverage</em> explicitly discloses the exact percentage of portfolio weight using live stock prices versus sector ETF proxies when a holding was not yet publicly listed.</p>
        <p><strong>What to take away:</strong> {bt_takeaway_bullet} Across historical replays, the 2008–09 GFC produces <strong>{s_2008_ret:.2f}% ({base_ccy} {s_2008_ccy:,.2f})</strong>, the 2020 COVID crash produces <strong>{s_2020_ret:.2f}% ({base_ccy} {s_2020_ccy:,.2f})</strong>, and the 2022 rate shock produces <strong>{s_2022_ret:.2f}% ({base_ccy} {s_2022_ccy:,.2f})</strong>.</p>
    </div>
    {backtest_html_block}
    {stress_df.round(2).to_html(index=False, border=0)}


    <h2>9. Model Limitations & Key Assumptions</h2>
    <div class="lim-grid">
        <div class="lim-card">
            <strong>1. Static Current-Holdings Backtest (Hindsight & Survivorship Bias):</strong>
            Historical simulations, drawdowns, and backtests apply today's {n_holdings}-position weights retroactively across {start_yr}–{end_yr}. Because current holdings survived and were selected after historical performance, backtested returns reflect survivorship and selection bias.
        </div>
        <div class="lim-card">
            <strong>2. Sample Window & Drift Estimation Error ({start_yr}–{end_yr}):</strong>
            By Merton's (1980) theorem, annualized sample mean returns carry a standard error of σ/√T (<strong>±{summary_stats['mean_se_pct']:.2f}% p.a.</strong> over {summary_stats['n_years']:.1f} years), making historical drift unreliable for 1-year forecasts. Short-horizon VaR enforces μ = 0, and 1-year simulations report zero-drift and equilibrium ERP (+{eq_erp*100:.1f}%) baselines.
        </div>
        <div class="lim-card">
            <strong>3. Static Covariance & Crisis Correlation Breakdown:</strong>
            Parametric and Cholesky Monte Carlo models assume a static linear covariance matrix Σ. During acute market crises, cross-asset correlations empirically converge toward 1.0, causing static covariance models to understate multi-week drawdowns.
        </div>
        <div class="lim-card">
            <strong>4. Elliptical Symmetry of Multivariate Student-t:</strong>
            Scaling correlated Gaussian shocks by a shared radial χ²(ν) variable captures heavy tails (df = {fitted_df:.1f}) and simultaneous tail dependence, but models symmetric upside and downside tails rather than asymmetric negative skewness.
        </div>
        <div class="lim-card">
            <strong>5. Unhedged Spot FX Translation:</strong>
            Holdings denominated outside {base_ccy} are translated at daily spot Yahoo Finance FX rates assuming zero currency hedging and zero transaction or custody spreads.
        </div>
        <div class="lim-card">
            <strong>6. Pre-IPO Sector Proxy Approximation in Historical Crises:</strong>
            Holdings lacking trading history during a historical crisis window are mapped to liquid sector ETFs configured in <code>sector_proxy_map</code> (and pre-inception benchmark periods use the configured crisis proxy), which smooths out single-stock idiosyncratic crash risk.
        </div>
    </div>

    <h2>10. Glossary of Quantitative Risk Terms</h2>
    <div class="lim-grid">
        <div class="lim-card">
            <strong>Value at Risk (VaR):</strong> The maximum portfolio loss not exceeded with a given confidence level (e.g., 95% or 99%) over a specified time horizon under normal market conditions. VaR is a quantile cutoff and does not measure tail severity beyond that cutoff.
        </div>
        <div class="lim-card">
            <strong>Expected Shortfall (CVaR):</strong> The conditional expected loss given that the loss exceeds the VaR threshold. Unlike VaR, CVaR is a coherent (sub-additive) risk measure. Note: Basel III FRTB specifies Expected Shortfall at the 97.5% level; this report evaluates 95% and 99% VaR and CVaR for direct quantile-for-quantile comparison across models.
        </div>
        <div class="lim-card">
            <strong>Marginal & Component VaR (Euler Decomposition):</strong> Marginal VaR (∂VaR/∂w<sub>i</sub>) measures the sensitivity of total portfolio VaR to a small change in weight w<sub>i</sub>. Component VaR (w<sub>i</sub> · Marginal VaR<sub>i</sub>) partitions 100% of Parametric VaR additively across positions.
        </div>
        <div class="lim-card">
            <strong>Risk / Weight Ratio (Internal Portfolio Beta β<sub>i,p</sub>):</strong> Ratio of a position's percentage risk contribution to its percentage capital weight. Equals Cov(r<sub>i</sub>, r<sub>p</sub>)/Var(r<sub>p</sub>); values &gt; 1.25x indicate risk amplifiers, while values &lt; 0.80x indicate diversifiers.
        </div>
        <div class="lim-card">
            <strong>Cholesky Decomposition & Multivariate Student-t:</strong> Factorizing Σ = LLᵀ transforms independent shocks Z into correlated asset shocks (Z·Lᵀ). Dividing by a shared radial χ²(ν) draw introduces fat tails and simultaneous tail dependence while preserving Σ.
        </div>
        <div class="lim-card">
            <strong>Kupiec (1995) POF Test & Diversification Ratio:</strong> The Kupiec Proportion of Failures test checks whether out-of-sample VaR breaches match the expected failure rate (5.0%). Diversification Ratio (∑w<sub>i</sub>σ<sub>i</sub> / σ<sub>p</sub>) measures volatility reduction from imperfect correlations.
        </div>
    </div>
</div>
</body>
</html>"""

    output_html.write_text(html, encoding="utf-8")
    print(f"Saved interactive report to: {output_html.resolve()}")
    webbrowser.open(output_html.resolve().as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Currency Portfolio Risk & Attribution Engine (v1.2)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/portfolio.example.json",
        help="Path to portfolio JSON config (default: config/portfolio.example.json)",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Mask ticker symbols with aliases and normalize NAV to 100,000",
    )
    args = parser.parse_args()

    cfg = load_config_file(args.config)
    out_dir = resolve_output_dir(cfg)
    settings = cfg["settings"]
    seed = int(settings.get("random_seed", 42))
    rf_rate = float(settings.get("risk_free_rate", 0.025))
    eq_erp = float(settings.get("equilibrium_erp", 0.07))
    bm_ticker = settings["benchmark_ticker"]

    print(f"1. Loading '{cfg['portfolio_name']}' and fetching market data...")
    bundle = fetch_portfolio_data_from_config(cfg, anonymize=args.anonymize)
    prices_base = bundle["prices_base"]
    log_returns = bundle["log_returns"]
    weights = bundle["weights"]
    total_nav = bundle["total_nav"]
    base_ccy = bundle["base_currency"]
    bm_prices_base = bundle["bm_prices_base"]
    bm_log_returns = bundle["bm_log_returns"]
    n_holdings = len(weights)
    start_yr = log_returns.index[0].year
    end_yr = log_returns.index[-1].year

    # 2. Summary Stats, Attribution & Drawdown
    print("2. Computing benchmark comparison, Euler attribution, and drawdowns...")
    summary_stats = compute_portfolio_vs_benchmark_summary(
        log_returns, weights, bm_log_returns, rf_rate_annual=rf_rate
    )
    attr_df = compute_risk_attribution(
        log_returns, weights, total_nav, bm_log_returns=bm_log_returns, alpha=0.95
    )
    # Rename currency columns dynamically to match base_ccy
    attr_df = attr_df.rename(columns={
        "Position EUR": f"Position ({base_ccy})",
        "Component VaR (1d EUR)": f"Component VaR (1d {base_ccy})",
    })

    sector_df, ccy_df = compute_group_exposures(attr_df, bundle["holding_meta"], anonymize=False)
    dd_info = compute_max_drawdown(prices_base, weights, bm_prices_eur=bm_prices_base)
    bt_info = run_rolling_backtest_and_kupiec(
        log_returns,
        weights,
        estimation_window=int(settings["rolling_window_days"]),
        min_total_days=int(settings["min_backtest_days"]),
        alpha=0.95,
    )

    # 3. 1-Day VaR/CVaR Models
    hist_m = historical_simulation_var(log_returns, weights, horizon_days=1)
    param_m = parametric_vcov_var(log_returns, weights, horizon_days=1)
    mc_norm_m, sim_norm = monte_carlo_var(log_returns, weights, horizon_days=1, distribution="normal", seed=seed)
    mc_t_m, sim_t = monte_carlo_var(log_returns, weights, horizon_days=1, distribution="student_t", seed=seed)
    fitted_df = float(mc_t_m["fitted_df"])

    model_comp_raw = pd.DataFrame({
        "Historical Simulation (Empirical)": hist_m,
        "Parametric (Variance-Covariance)": param_m,
        "Monte Carlo (Gaussian)": mc_norm_m,
        f"Monte Carlo (Student-t, df={fitted_df:.1f})": {k: v for k, v in mc_t_m.items() if k != "fitted_df"},
    }).T

    model_comp_df = pd.DataFrame(index=model_comp_raw.index)
    for col, label in [("VaR_95", "1d 95% VaR"), ("CVaR_95", "1d 95% CVaR"), ("VaR_99", "1d 99% VaR"), ("CVaR_99", "1d 99% CVaR")]:
        model_comp_df[f"{label} (%)"] = model_comp_raw[col].apply(lambda v: f"{-v*100.0:.2f}%")
        model_comp_df[f"{label} ({base_ccy})"] = model_comp_raw[col].apply(lambda v: f"-{base_ccy} {v*total_nav:,.2f}")

    # Benchmark Comparison Table (100% dynamic years & currency)
    rc_dec = attr_df["Risk Contribution (%)"].values / 100.0
    enb_risk = float(1.0 / np.sum(rc_dec ** 2))
    bm_comp_table = pd.DataFrame([
        {
            "Metric": f"Annualized Volatility ({base_ccy})",
            "Portfolio": f"{summary_stats['port_ann_vol_pct']:.2f}%",
            f"Benchmark ({bm_ticker} in {base_ccy})": f"{summary_stats['bm_ann_vol_pct']:.2f}%",
            "Reference Standard / Interpretation": f"Relative to Benchmark ({summary_stats['port_ann_vol_pct'] - summary_stats['bm_ann_vol_pct']:+.2f}% active vol; Beta = {summary_stats['port_beta_bm']:.2f})",
        },
        {
            "Metric": f"Annualized Compound Return ({start_yr}-{end_yr}*)",
            "Portfolio": f"{summary_stats['port_ann_ret_pct']:+.2f}%",
            f"Benchmark ({bm_ticker} in {base_ccy})": f"{summary_stats['bm_ann_ret_pct']:+.2f}%",
            "Reference Standard / Interpretation": "*In-sample current holdings (subject to selection/survivorship bias)",
        },
        {
            "Metric": f"Sharpe / Sortino Ratio (Rf = {rf_rate*100:.2f}%)",
            "Portfolio": f"{summary_stats['port_sharpe']:.2f} / {summary_stats['port_sortino']:.2f}",
            f"Benchmark ({bm_ticker} in {base_ccy})": f"{summary_stats['bm_sharpe']:.2f} / {summary_stats['bm_sortino']:.2f}",
            "Reference Standard / Interpretation": "Sortino penalizes downside volatility only",
        },
        {
            "Metric": "1-Day 95% VaR / 95% CVaR [Parametric]",
            "Portfolio": f"-{summary_stats['port_param_1d']['VaR_95']*100:.2f}% / -{summary_stats['port_param_1d']['CVaR_95']*100:.2f}%",
            f"Benchmark ({bm_ticker} in {base_ccy})": f"-{summary_stats['bm_param_1d']['VaR_95']*100:.2f}% / -{summary_stats['bm_param_1d']['CVaR_95']*100:.2f}%",
            "Reference Standard / Interpretation": "Daily monitoring standard (1-in-20 day cutoff & tail average)",
        },
        {
            "Metric": "10-Day 95% VaR [Parametric] / 99% CVaR [Student-t]",
            "Portfolio": f"-{summary_stats['port_param_10d']['VaR_95']*100:.2f}% / -{summary_stats['port_t_10d']['CVaR_99']*100:.2f}%",
            f"Benchmark ({bm_ticker} in {base_ccy})": f"-{summary_stats['bm_param_10d']['VaR_95']*100:.2f}% / -{summary_stats['bm_param_10d']['CVaR_99']*100:.2f}% (Param.)",
            "Reference Standard / Interpretation": "2-Week Governance Horizon (Zero-Drift)",
        },
        {
            "Metric": "Maximum Historical Drawdown (Peak-to-Trough)",
            "Portfolio": f"{dd_info['max_drawdown_pct']:.2f}% ({dd_info['peak_date']} to {dd_info['trough_date']})",
            f"Benchmark ({bm_ticker} in {base_ccy})": f"{dd_info['bm_max_drawdown_pct']:.2f}% ({dd_info['bm_peak_date']} to {dd_info['bm_trough_date']})",
            "Reference Standard / Interpretation": f"Peak-to-trough decline over {start_yr}-{end_yr} sample",
        },
        {
            "Metric": "Diversification Ratio & Effective Bets (N_eff)",
            "Portfolio": f"DR = {summary_stats['diversification_ratio']:.2f}x | {summary_stats['enb_capital']:.1f} Cap / {enb_risk:.1f} Risk Bets",
            f"Benchmark ({bm_ticker} in {base_ccy})": "Broad Market Index",
            "Reference Standard / Interpretation": f"Theoretical Baseline: DR > 1.0x shows correlation reduction; Ideal N_eff = {n_holdings:.1f}",
        },
    ])

    # 4. Horizon & Drift Sensitivity Table
    print("3. Simulating multi-horizon & 1-year drift sensitivity trajectories...")
    sim_1yr = simulate_horizon_and_drift_sensitivity(
        log_returns, weights, total_nav, days=252, num_simulations=3000, equilibrium_erp_annual=eq_erp, seed=seed
    )
    _, sim_10d_t = monte_carlo_var(log_returns, weights, horizon_days=10, distribution="student_t", df_student=fitted_df, seed=seed)

    def _summarize_dist(arr: np.ndarray, horizon_label: str, assumption_note: str) -> dict:
        q95 = np.percentile(arr, 5)
        cvar95 = arr[arr <= q95].mean()
        q99 = np.percentile(arr, 1)
        cvar99 = arr[arr <= q99].mean()
        return {
            "Horizon & Regime": horizon_label,
            "Model & Drift / Vol Assumption": assumption_note,
            "Median Return": f"{np.median(arr)*100:+.2f}%",
            "95% VaR (%)": f"{q95*100:.2f}%",
            f"95% VaR ({base_ccy})": f"-{base_ccy} {-q95*total_nav:,.0f}",
            "95% CVaR (%)": f"{cvar95*100:.2f}%",
            "99% CVaR (%)": f"{cvar99*100:.2f}%",
            "Prob. of Loss (<0%)": f"{np.mean(arr < 0)*100:.1f}%",
        }

    horizon_drift_df = pd.DataFrame([
        _summarize_dist(sim_t, "1-Day (Daily Risk Monitoring)", f"[Student-t MC] Zero Drift (μ = 0%), df={fitted_df:.1f}"),
        _summarize_dist(sim_10d_t, "10-Day / 2-Week (Governance Horizon)", f"[Student-t MC] Zero Drift (μ = 0%), df={fitted_df:.1f}"),
        _summarize_dist(sim_1yr["ret_1yr_zero"], "1-Year (252d Conservative Pure Risk)", "[Gaussian MC] Zero Drift (μ = 0% p.a.), 1x Vol"),
        _summarize_dist(sim_1yr["ret_1yr_eq"], "1-Year (252d Equilibrium ERP Baseline)", f"[Gaussian MC] Long-Run ERP (μ = +{eq_erp*100:.1f}% p.a.), 1x Vol"),
        _summarize_dist(sim_1yr["ret_1yr_hist"], f"1-Year (252d {start_yr}-{end_yr} Sample Drift*)", f"[Gaussian MC] *Sample Mean (SE = ±{summary_stats['mean_se_pct']:.1f}% p.a. - Biased)"),
        _summarize_dist(sim_1yr["ret_1yr_stress"], "1-Year (252d Stressed Crisis Regime)", "[Gaussian MC] Stressed Drift (-20% shock) & 2.0x Vol"),
    ])

    # 5. Historical & Configurable Hypothetical Stress Tests
    print("4. Running historical crisis replays & configurable stress scenarios...")
    top_risk_sector = str(sector_df.index[0])
    stress_df = run_crisis_stress_tests(
        weights=weights,
        total_nav_eur=total_nav,
        log_returns=log_returns,
        raw_weights_by_ticker=bundle["raw_weights_by_ticker"],
        holding_meta=bundle["holding_meta"],
        config=cfg,
        top_risk_sector=top_risk_sector,
        anonymize=args.anonymize,
    )

    # 6. Rule-Based Advisory & Rebalancing Simulator
    print("5. Evaluating advisory governance checks & rebalancing impact...")
    flags_df, rebal_meta = evaluate_advisory_guidelines(
     log_returns=log_returns,
     weights=weights,
     total_nav=total_nav,
     base_ccy=base_ccy,
     summary_stats=summary_stats,
     attr_df=attr_df,
     sector_df=sector_df,
     ccy_df=ccy_df,
     dd_info=dd_info,
     bt_info=bt_info,
     thresholds=cfg["advisory_thresholds"],
     holding_meta=bundle["holding_meta"],
    )

    # 7. Build Annotated 6-Panel Visual Dashboard (100% dynamic coordinates)
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(3, 2, figsize=(16, 15.5))

    # Panel A
    bins = np.linspace(-0.065, 0.065, 105)
    axes[0, 0].hist(sim_norm * 100, bins=bins * 100, density=True, alpha=0.5, label="Gaussian MC", color="#3b82f6")
    axes[0, 0].hist(sim_t * 100, bins=bins * 100, density=True, alpha=0.5, label=f"Student-t MC (df={fitted_df:.1f})", color="#e11d48")
    axes[0, 0].axvline(-mc_norm_m["CVaR_99"] * 100, color="#1d4ed8", linestyle="--", linewidth=1.8, label=f"Gaussian 99% CVaR (-{mc_norm_m['CVaR_99']*100:.2f}%)")
    axes[0, 0].axvline(-mc_t_m["CVaR_99"] * 100, color="#9f1239", linestyle="-", linewidth=2.0, label=f"Student-t 99% CVaR (-{mc_t_m['CVaR_99']*100:.2f}%)")
    axes[0, 0].annotate(
        f"Fat-Tail Gap:\n[Student-t] 99% CVaR is\n-{mc_t_m['CVaR_99']*100:.2f}% vs -{mc_norm_m['CVaR_99']*100:.2f}%",
        xy=(-mc_t_m["CVaR_99"] * 100, 0.15),
        xytext=(0.04, 0.72),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#9f1239", lw=1.3),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffe4e6", ec="#9f1239", lw=0.8),
    )
    axes[0, 0].set_title("A. 1-Day Return Density: Gaussian vs. Fat-Tailed Student-t")
    axes[0, 0].set_xlabel("Daily Portfolio Return (%)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(loc="upper right", fontsize=8.5)

    # Panel B (Adapts to any N >= 2)
    n_show = min(12, len(attr_df))
    top_n = attr_df.head(n_show)
    x = np.arange(n_show)
    axes[0, 1].bar(x - 0.18, top_n["Weight (%)"], width=0.35, label="Capital Weight (%)", color="#64748b")
    axes[0, 1].bar(x + 0.18, top_n["Risk Contribution (%)"], width=0.35, label="Euler Risk Contribution (%)", color="#ea580c")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(top_n.index, rotation=35, ha="right")
    k_callout = min(3, n_show)
    axes[0, 1].annotate(
        f"Top {k_callout} holdings = {attr_df.head(k_callout)['Weight (%)'].sum():.1f}% weight\nbut drive {attr_df.head(k_callout)['Risk Contribution (%)'].sum():.1f}% of total risk",
        xy=(0.18, float(top_n["Risk Contribution (%)"].iloc[0])),
        xytext=(0.45, 0.72),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#9a3412", lw=1.3),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffedd5", ec="#ea580c", lw=0.8),
    )
    axes[0, 1].set_title(f"B. Capital Weight vs. Euler Risk Contribution (Top {n_show})")
    axes[0, 1].set_ylabel("Percentage (%)")
    axes[0, 1].legend(loc="upper right", fontsize=8.5)

    # Panel C (Handles enabled vs disabled short-history backtest gracefully)
    if bt_info["enabled"]:
        bt_dates = bt_info["dates"]
        realized = bt_info["realized_returns"]
        var_line = -bt_info["param_var_series"]
        breaches = realized < var_line
        kp_stat = bt_info["kupiec_parametric"]
        axes[1, 0].plot(bt_dates, realized * 100, color="#94a3b8", alpha=0.55, linewidth=0.8, label="Daily Realized Return (%)")
        axes[1, 0].plot(bt_dates, var_line * 100, color="#1d4ed8", linewidth=1.5, label="Rolling 95% VaR [Parametric]")
        axes[1, 0].scatter(
            bt_dates[breaches], realized[breaches] * 100, color="#dc2626", zorder=5, s=18,
            label=f"VaR Breaches (n={kp_stat['actual_breaches']}/{kp_stat['total_days']}, p={kp_stat['p_value']:.2f})"
        )
        if breaches.sum() > 0:
            worst_b_idx = int(np.argmin(realized[breaches]))
            wb_date = bt_dates[breaches][worst_b_idx]
            wb_val = float(realized[breaches][worst_b_idx] * 100)
            axes[1, 0].annotate(
                f"Worst breach: {wb_val:.2f}%\n({wb_date.date()})",
                xy=(wb_date, wb_val),
                xytext=(0.05, 0.12),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#dc2626", lw=0.8),
            )
        axes[1, 0].set_title(f"C. Out-of-Sample Rolling Backtest (Kupiec p={kp_stat['p_value']:.2f})")
        axes[1, 0].set_ylabel("Daily Return (%)")
        axes[1, 0].legend(loc="upper right", fontsize=8.5)
    else:
        axes[1, 0].text(
            0.5, 0.5, bt_info["reason"], ha="center", va="center", wrap=True, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.6", fc="#fffbeb", ec="#d97706")
        )
        axes[1, 0].set_title("C. Out-of-Sample Rolling Backtest (Disabled - Short Sample)")
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])

    # Panel D
    axes[1, 1].fill_between(dd_info["drawdown_series"].index, dd_info["drawdown_series"].values * 100, 0, color="#dc2626", alpha=0.35, label=f"Portfolio (Max DD: {dd_info['max_drawdown_pct']:.2f}%)")
    axes[1, 1].plot(dd_info["bm_drawdown_series"].index, dd_info["bm_drawdown_series"].values * 100, color="#0f172a", linewidth=1.3, linestyle="--", label=f"{bm_ticker} (Max DD: {dd_info['bm_max_drawdown_pct']:.2f}%)")
    axes[1, 1].annotate(
        f"Max Drawdown: {dd_info['max_drawdown_pct']:.2f}%\n({dd_info['trough_date']})",
        xy=(pd.Timestamp(dd_info["trough_date"]), dd_info["max_drawdown_pct"]),
        xytext=(0.55, 0.25),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#991b1b", lw=0.8),
    )
    axes[1, 1].set_title(f"D. Historical Drawdown: Portfolio vs. {bm_ticker} ({base_ccy})")
    axes[1, 1].set_ylabel("Drawdown from Peak (%)")
    axes[1, 1].legend(loc="lower right", fontsize=8.5)

    # Panel E
    days_axis = np.arange(1, 253)
    paths_eq = sim_1yr["paths_eq_eur"]
    paths_zero = sim_1yr["paths_zero_eur"]
    axes[2, 0].plot(days_axis, paths_eq[:, :250], color="#3b82f6", alpha=0.035)
    axes[2, 0].plot(days_axis, np.median(paths_eq, axis=1), color="#1e3a8a", linewidth=2.2, label=f"Equilibrium ERP Median (+{eq_erp*100:.1f}% p.a.)")
    axes[2, 0].plot(days_axis, np.median(paths_zero, axis=1), color="#475569", linestyle="-.", linewidth=1.6, label="Zero-Drift Median (μ = 0% p.a.)")
    axes[2, 0].plot(days_axis, np.percentile(paths_eq, 95, axis=1), color="#15803d", linestyle="--", linewidth=1.5, label="95th Percentile (Upside Cone)")
    axes[2, 0].plot(days_axis, np.percentile(paths_eq, 5, axis=1), color="#b91c1c", linestyle="--", linewidth=1.6, label="5th Percentile (95% VaR Cone)")
    axes[2, 0].axhline(total_nav, color="black", linestyle=":", linewidth=1.0)
    axes[2, 0].annotate(
        f"1Yr 95% VaR [Gaussian MC]:\n{base_ccy} {np.percentile(paths_eq[-1, :], 5):,.0f} ({np.percentile(sim_1yr['ret_1yr_eq'], 5)*100:+.1f}%)",
        xy=(252, float(np.percentile(paths_eq[-1, :], 5))),
        xytext=(0.48, 0.14),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#b91c1c", lw=0.8),
    )
    axes[2, 0].set_title(f"E. 1-Year (252d) Monte Carlo Fan Chart ({base_ccy})")
    axes[2, 0].set_xlabel("Trading Days Forward")
    axes[2, 0].set_ylabel(f"Portfolio Value ({base_ccy})")
    axes[2, 0].legend(loc="upper left", fontsize=8.2)

    # Panel F
    shared_bins = np.linspace(-0.75, 1.00, 75) * 100
    v_eq = float(np.percentile(sim_1yr["ret_1yr_eq"], 5) * 100)
    v_zero = float(np.percentile(sim_1yr["ret_1yr_zero"], 5) * 100)
    v_str = float(np.percentile(sim_1yr["ret_1yr_stress"], 5) * 100)
    axes[2, 1].hist(sim_1yr["ret_1yr_eq"] * 100, bins=shared_bins, alpha=0.5, color="#3b82f6", edgecolor="white", label=f"Equilibrium +{eq_erp*100:.1f}% Drift (VaR95: {v_eq:.1f}%)")
    axes[2, 1].hist(sim_1yr["ret_1yr_zero"] * 100, bins=shared_bins, alpha=0.45, color="#64748b", edgecolor="white", label=f"Zero Drift μ=0% (VaR95: {v_zero:.1f}%)")
    axes[2, 1].hist(sim_1yr["ret_1yr_stress"] * 100, bins=shared_bins, alpha=0.45, color="#f43f5e", edgecolor="white", label=f"Stressed 2x Vol (VaR95: {v_str:.1f}%)")
    axes[2, 1].axvline(v_eq, color="#1d4ed8", linestyle="--", linewidth=1.6)
    axes[2, 1].axvline(v_zero, color="#334155", linestyle="-.", linewidth=1.6)
    axes[2, 1].axvline(v_str, color="#be123c", linestyle="--", linewidth=1.8)
    axes[2, 1].annotate(
        f"1Yr 95% VaR [Gaussian MC]:\n{v_eq:.1f}% (Eq. Drift) vs {v_zero:.1f}% (Zero Drift)",
        xy=(v_zero, 20),
        xytext=(0.03, 0.72),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f1f5f9", ec="#334155", lw=0.8),
    )
    axes[2, 1].set_title("F. 1-Year Horizon Drift & Volatility Sensitivity Overlay")
    axes[2, 1].set_xlabel("Annual Portfolio Return (%)")
    axes[2, 1].set_ylabel("Simulated Frequency (of 3,000)")
    axes[2, 1].legend(loc="upper right", fontsize=8.2)

    plt.tight_layout()
    chart_path = out_dir / "portfolio_risk_dashboard.png"
    plt.savefig(chart_path, dpi=300)
    plt.close(fig)

    html_filename = "risk_report_anonymized.html" if args.anonymize else "risk_report_dashboard.html"
    output_html = out_dir / html_filename

    build_and_open_browser_report(
        bundle=bundle,
        summary_stats=summary_stats,
        bm_comp_table=bm_comp_table,
        model_comp_df=model_comp_df,
        horizon_drift_df=horizon_drift_df,
        sector_df=sector_df,
        ccy_df=ccy_df,
        attr_df=attr_df,
        dd_info=dd_info,
        bt_info=bt_info,
        stress_df=stress_df,
        flags_df=flags_df,
        rebal_meta=rebal_meta,
        fitted_df=fitted_df,
        anonymize=args.anonymize,
        chart_path=chart_path,
        output_html=output_html,
    )


if __name__ == "__main__":
    main()