"""
Module 5: Rule-Based Portfolio Risk Advisory & Rebalancing Simulator (v1.2.1)
Evaluates configurable governance thresholds with transparent traffic-light statuses
and simulates the risk-only impact of trimming flagged risk amplifiers into diversifiers.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.risk_models import parametric_vcov_var, monte_carlo_var


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def evaluate_advisory_guidelines(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    total_nav: float,
    base_ccy: str,
    summary_stats: Dict[str, Any],
    attr_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    ccy_df: pd.DataFrame,
    dd_info: Dict[str, Any],
    bt_info: Dict[str, Any],
    thresholds: Dict[str, Any],
    holding_meta: Dict[str, Dict[str, Any]] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates 8 transparent governance checks using configurable thresholds and computes
    a 'what-if' risk-only rebalancing simulation using strictly the flagged amplifiers/diversifiers.
    """
    n_holdings = len(weights)
    equal_pos_share_pct = 100.0 / n_holdings
    pos_mult = float(thresholds.get("single_pos_equal_share_mult", 1.65))
    pos_cap_pct = pos_mult * equal_pos_share_pct

    rwr_high = float(thresholds.get("risk_weight_ratio_high", 1.25))
    rwr_low = float(thresholds.get("risk_weight_ratio_low", 0.80))

    n_sectors = len(sector_df)
    min_sec_flag = int(thresholds.get("min_sectors_for_flag", 4))
    sec_mult = float(thresholds.get("sector_equal_share_mult", 1.50))
    sec_cap_pct = sec_mult * (100.0 / max(n_sectors, 1))

    fx_max_pct = float(thresholds.get("foreign_ccy_risk_max_pct", 65.0))
    enb_min_ratio = float(thresholds.get("enb_to_n_min_ratio", 0.60))
    dd_excess_max = float(thresholds.get("max_dd_excess_vs_bm_pct", 3.0))
    kupiec_p_min = float(thresholds.get("kupiec_p_value_min", 0.05))
    avg_corr_high = float(thresholds.get("avg_corr_high", 0.50))
    max_pair_high = float(thresholds.get("max_pair_corr_high", 0.75))

    flags: List[Dict[str, str]] = []
    n_triggered = 0

    # 1. Single-Position Risk Contribution
    top_pos_name = str(attr_df.index[0])
    top_pos_rc = float(attr_df.iloc[0]["Risk Contribution (%)"])
    top_pos_w = float(attr_df.iloc[0]["Weight (%)"])
    n_over_pos = int((attr_df["Risk Contribution (%)"] > pos_cap_pct).sum())
    if n_over_pos > 0:
        n_triggered += 1
        st_1 = "FLAG / REVIEW"
        verb_1 = "exceeds" if n_over_pos == 1 else "exceed"
        msg_1 = (
            f"{_plural(n_over_pos, 'position')} {verb_1} the {pos_cap_pct:.1f}% single-position risk guideline "
            f"(led by {top_pos_name} at {top_pos_rc:.1f}% of risk vs. {top_pos_w:.1f}% weight). "
            f"Consider reviewing position sizing caps at the next committee meeting."
        )
    else:
        st_1 = "PASS"
        msg_1 = (
            f"Largest risk contributor ({top_pos_name} at {top_pos_rc:.1f}% of risk) sits within "
            f"the {pos_cap_pct:.1f}% single-position guideline."
        )
    flags.append({
        "Governance Check": "1. Single-Position Risk Share",
        "Status": st_1,
        "Current Value": f"{top_pos_name}: {top_pos_rc:.1f}% risk",
        "Configured Threshold & Rationale": f"<= {pos_cap_pct:.1f}% ({pos_mult:.2f}x equal share 100%/{n_holdings}; prevents single-stock tail dominance)",
        "Plain-English Interpretation & Action": msg_1,
    })

    # 2. Risk / Weight Ratio (Risk Amplifiers vs. Diversifiers)
    amplifiers = attr_df[attr_df["Risk / Weight Ratio"] > rwr_high].sort_values("Risk Contribution (%)", ascending=False)
    diversifiers = attr_df[attr_df["Risk / Weight Ratio"] < rwr_low].sort_values("Risk / Weight Ratio", ascending=True)
    top_amp_name = str(attr_df.sort_values("Risk / Weight Ratio", ascending=False).index[0])
    top_amp_val = float(attr_df["Risk / Weight Ratio"].max())
    excess_pct_label = (rwr_high - 1.0) * 100.0
    if len(amplifiers) > 0:
        n_triggered += 1
        st_2 = "FLAG / REVIEW"
        amp_list = ", ".join(list(amplifiers.index[:4]))
        verb_2 = "has a Risk/Weight ratio" if len(amplifiers) == 1 else "have Risk/Weight ratios"
        msg_2 = (
            f"{_plural(len(amplifiers), 'holding')} {verb_2} > {rwr_high:.2f}x ({amp_list}), meaning each "
            f"unit of capital adds >{excess_pct_label:.0f}% more portfolio volatility than average. Flag for discussion when sizing new pitches."
        )
    else:
        st_2 = "PASS"
        msg_2 = f"No holdings exceed the {rwr_high:.2f}x Risk/Weight ratio threshold (max: {top_amp_name} at {top_amp_val:.2f}x)."
    flags.append({
        "Governance Check": "2. Risk / Weight Efficiency",
        "Status": st_2,
        "Current Value": f"{_plural(len(amplifiers), 'Amplifier')} (max {top_amp_val:.2f}x)",
        "Configured Threshold & Rationale": f"<= {rwr_high:.2f}x (1.00x = neutral sensitivity; >{rwr_high:.2f}x signals high beta/correlation to portfolio)",
        "Plain-English Interpretation & Action": msg_2,
    })

    # 3. Sector Risk Concentration (only flagged when >= min_sectors_for_flag)
    top_sec_name = str(sector_df.index[0])
    top_sec_rc = float(sector_df.iloc[0]["Risk Contribution (%)"])
    top_sec_w = float(sector_df.iloc[0]["Weight (%)"])
    if n_sectors < min_sec_flag:
        st_3 = "INFO"
        thresh_3 = f"Informational only (< {min_sec_flag} sectors present; {_plural(n_sectors, 'sector')} in portfolio)"
        msg_3 = (
            f"Portfolio spans {_plural(n_sectors, 'sector')}, with {top_sec_name} accounting for {top_sec_rc:.1f}% of risk "
            f"({top_sec_w:.1f}% weight). Sector concentration flags apply only when >= {min_sec_flag} sectors are held."
        )
    else:
        thresh_3 = f"<= {sec_cap_pct:.1f}% ({sec_mult:.2f}x equal sector share 100%/{n_sectors}; avoids single-sector crash)"
        if top_sec_rc > sec_cap_pct:
            n_triggered += 1
            st_3 = "FLAG / REVIEW"
            msg_3 = (
                f"{top_sec_name} contributes {top_sec_rc:.1f}% of total risk (vs. {top_sec_w:.1f}% capital weight), "
                f"exceeding the {sec_cap_pct:.1f}% guideline. Consider requiring replacement trades for new pitches in this sector."
            )
        else:
            st_3 = "PASS"
            msg_3 = f"Top sector ({top_sec_name} at {top_sec_rc:.1f}% of risk) is within the {sec_cap_pct:.1f}% guideline."
    flags.append({
        "Governance Check": "3. Sector Risk Concentration",
        "Status": st_3,
        "Current Value": f"{top_sec_name}: {top_sec_rc:.1f}% risk",
        "Configured Threshold & Rationale": thresh_3,
        "Plain-English Interpretation & Action": msg_3,
    })

    # 4. Foreign Currency Risk Concentration
    base_ccy_rc = float(ccy_df["Risk Contribution (%)"].get(base_ccy, 0.0))
    foreign_ccy_rc = max(0.0, 100.0 - base_ccy_rc)
    foreign_ccy_w = max(0.0, 100.0 - float(ccy_df["Weight (%)"].get(base_ccy, 0.0)))
    if len(ccy_df) == 1 and base_ccy in ccy_df.index:
        st_4 = "INFO"
        msg_4 = f"100% of holdings trade in {base_ccy}; zero unhedged foreign currency risk."
    elif foreign_ccy_rc > fx_max_pct:
        n_triggered += 1
        st_4 = "FLAG / REVIEW"
        msg_4 = (
            f"Non-{base_ccy} holdings represent {foreign_ccy_w:.1f}% of capital but drive {foreign_ccy_rc:.1f}% of total risk "
            f"(above the {fx_max_pct:.0f}% guideline). Flag for discussion whether FX swings could dominate stock selection."
        )
    else:
        st_4 = "PASS"
        msg_4 = f"Non-{base_ccy} risk share ({foreign_ccy_rc:.1f}%) is within the {fx_max_pct:.0f}% currency risk guideline."
    flags.append({
        "Governance Check": f"4. Foreign Currency Risk (Non-{base_ccy})",
        "Status": st_4,
        "Current Value": f"{foreign_ccy_rc:.1f}% risk ({foreign_ccy_w:.1f}% weight)",
        "Configured Threshold & Rationale": f"<= {fx_max_pct:.0f}% (prevents unhedged FX volatility from overwhelming equity selection)",
        "Plain-English Interpretation & Action": msg_4,
    })

    # 5. Effective Number of Risk Bets vs. Holdings
    rc_dec = attr_df["Risk Contribution (%)"].values / 100.0
    enb_risk = float(1.0 / np.sum(rc_dec ** 2))
    enb_ratio = enb_risk / n_holdings
    if enb_ratio < enb_min_ratio:
        n_triggered += 1
        st_5 = "FLAG / REVIEW"
        msg_5 = (
            f"Effective risk bets ({enb_risk:.1f} out of {n_holdings} holdings, {enb_ratio*100:.0f}%) fall below the "
            f"{enb_min_ratio*100:.0f}% breadth target, indicating correlated holdings behave like fewer independent bets."
        )
    else:
        st_5 = "PASS"
        msg_5 = (
            f"Effective risk bets ({enb_risk:.1f} out of {n_holdings} holdings, {enb_ratio*100:.0f}%) exceed the "
            f"{enb_min_ratio*100:.0f}% breadth guideline."
        )
    flags.append({
        "Governance Check": "5. Effective Risk Diversification (N_eff)",
        "Status": st_5,
        "Current Value": f"{enb_risk:.1f} / {n_holdings} ({enb_ratio*100:.0f}%)",
        "Configured Threshold & Rationale": f">= {enb_min_ratio*100:.0f}% of N (1/sum(RC^2); checks whether holdings act as independent bets)",
        "Plain-English Interpretation & Action": msg_5,
    })

    # 6. Maximum Drawdown vs. Benchmark
    port_dd = abs(float(dd_info["max_drawdown_pct"]))
    bm_dd = abs(float(dd_info["bm_max_drawdown_pct"]))
    dd_diff = port_dd - bm_dd
    if dd_diff > dd_excess_max:
        n_triggered += 1
        st_6 = "FLAG / REVIEW"
        msg_6 = (
            f"Historical peak-to-trough drawdown (-{port_dd:.2f}%) exceeded the benchmark (-{bm_dd:.2f}%) by "
            f"{dd_diff:.2f} pp (above the {dd_excess_max:.1f} pp tolerance). Review downside resilience in risk-off regimes."
        )
    else:
        st_6 = "PASS"
        msg_6 = (
            f"Historical max drawdown (-{port_dd:.2f}%) is within {dd_diff:+.2f} pp of the benchmark (-{bm_dd:.2f}%), "
            f"inside the +{dd_excess_max:.1f} pp tolerance."
        )
    flags.append({
        "Governance Check": "6. Peak-to-Trough Drawdown vs. Benchmark",
        "Status": st_6,
        "Current Value": f"-{port_dd:.2f}% vs -{bm_dd:.2f}% ({dd_diff:+.2f} pp)",
        "Configured Threshold & Rationale": f"Excess DD <= +{dd_excess_max:.1f} pp vs. benchmark (gauges historical downside resilience)",
        "Plain-English Interpretation & Action": msg_6,
    })

    # 7. Out-of-Sample Backtest Calibration & Clustering
    if not bt_info.get("enabled", False):
        st_7 = "INFO"
        val_7 = f"Disabled ({bt_info['total_sample_days']}d < {bt_info['min_required_days']}d)"
        msg_7 = bt_info["reason"]
    else:
        kp = bt_info["kupiec_parametric"]
        p_val = float(kp["p_value"])
        breaches = bt_info["realized_returns"] < (-bt_info["param_var_series"])
        bt_years = pd.Series(breaches, index=bt_info["dates"]).groupby(bt_info["dates"].year).agg(["sum", "count"])
        bt_years["rate"] = bt_years["sum"] / bt_years["count"]
        peak_yr = int(bt_years["rate"].idxmax())
        peak_rate = float(bt_years["rate"].max()) * 100.0
        overall_rate = float(kp["actual_rate_pct"])

        if p_val < kupiec_p_min:
            n_triggered += 1
            st_7 = "FLAG / REVIEW"
            msg_7 = (
                f"Parametric 95% VaR failed Kupiec unconditional coverage (p = {p_val:.3f} < {kupiec_p_min:.2f}; "
                f"{kp['actual_breaches']} breaches vs {kp['expected_breaches']:.1f} expected). Rely on Student-t CVaR."
            )
        elif peak_rate > 1.5 * overall_rate:
            n_triggered += 1
            st_7 = "FLAG / REVIEW"
            msg_7 = (
                f"Passes Kupiec overall count (p = {p_val:.3f}), but breaches cluster in {peak_yr} ({peak_rate:.1f}% breach rate "
                f"vs {overall_rate:.1f}% average) due to static 252d covariance lag. Pair daily VaR with stress tests."
            )
        else:
            st_7 = "PASS"
            msg_7 = f"Parametric 95% VaR passes Kupiec calibration (p = {p_val:.3f}) without severe annual clustering."
        val_7 = f"p = {p_val:.3f} ({kp['actual_breaches']}/{kp['total_days']} breaches)"

    flags.append({
        "Governance Check": "7. Backtest Calibration & Regime Clustering",
        "Status": st_7,
        "Current Value": val_7,
        "Configured Threshold & Rationale": f"Kupiec p >= {kupiec_p_min:.2f} & peak-year breach rate <= 1.5x avg (validates VaR accuracy)",
        "Plain-English Interpretation & Action": msg_7,
    })

    # 8. Pairwise Correlation Clustering
    corr_mat = log_returns.corr().values
    iu = np.triu_indices_from(corr_mat, k=1)
    pair_corrs = corr_mat[iu]
    avg_corr = float(np.mean(pair_corrs))
    max_corr = float(np.max(pair_corrs))
    max_idx = int(np.argmax(pair_corrs))
    pair_a = str(log_returns.columns[iu[0][max_idx]])
    pair_b = str(log_returns.columns[iu[1][max_idx]])

    if avg_corr > avg_corr_high or max_corr > max_pair_high:
        n_triggered += 1
        st_8 = "FLAG / REVIEW"
        msg_8 = (
            f"Average pairwise correlation is {avg_corr:.2f} and highest pair ({pair_a} / {pair_b}) reaches {max_corr:.2f} "
            f"(above {max_pair_high:.2f} overlap threshold). Consider reviewing duplicate factor exposure."
        )
    else:
        st_8 = "PASS"
        msg_8 = f"Average pairwise correlation ({avg_corr:.2f}) and max pair ({pair_a}/{pair_b}: {max_corr:.2f}) are within thresholds."
    flags.append({
        "Governance Check": "8. Pairwise Correlation Clustering",
        "Status": st_8,
        "Current Value": f"Avg {avg_corr:.2f} | Max {max_corr:.2f} ({pair_a}/{pair_b})",
        "Configured Threshold & Rationale": f"Avg <= {avg_corr_high:.2f} & Max Pair <= {max_pair_high:.2f} (flags overlapping price behavior)",
        "Plain-English Interpretation & Action": msg_8,
    })

    flags_df = pd.DataFrame(flags)

    # ==========================================================================
    # QUANTITATIVE REBALANCING IMPACT SIMULATOR (STRICTLY FLAGGED AMPLIFIERS)
    # ==========================================================================
    trim_pct = float(thresholds.get("rebalance_trim_pct_per_pos", 1.5))
    top_k = min(int(thresholds.get("rebalance_top_k", 3)), len(weights) // 2)

    # Trim strictly holdings flagged as Risk Amplifiers (> rwr_high) into Diversifiers (< rwr_low)
    amp_candidates = amplifiers.head(top_k)
    div_candidates = diversifiers.head(top_k)

    new_weights = weights.copy()
    moved_total_pct = 0.0
    trimmed_names: List[str] = []
    added_names: List[str] = []

    if len(amp_candidates) >= 1 and len(div_candidates) >= 1:
        for t in amp_candidates.index:
            actual_trim_dec = min(trim_pct / 100.0, float(new_weights[t]) * 0.5)
            new_weights[t] -= actual_trim_dec
            moved_total_pct += actual_trim_dec * 100.0
            trimmed_names.append(f"{t} (-{actual_trim_dec*100:.2f}%)")

        n_div_used = len(div_candidates)
        add_each_dec = (moved_total_pct / 100.0) / n_div_used
        for t in div_candidates.index:
            new_weights[t] += add_each_dec
            added_names.append(f"{t} (+{add_each_dec*100:.2f}%)")

        new_weights = new_weights / new_weights.sum()

    cov_ann = log_returns.cov().values * 252.0
    vol_before = float(np.sqrt(weights.values.T @ cov_ann @ weights.values)) * 100.0
    vol_after = float(np.sqrt(new_weights.values.T @ cov_ann @ new_weights.values)) * 100.0

    p1_before = parametric_vcov_var(log_returns, weights, horizon_days=1)
    p1_after = parametric_vcov_var(log_returns, new_weights, horizon_days=1)

    p10_before = parametric_vcov_var(log_returns, weights, horizon_days=10)
    p10_after = parametric_vcov_var(log_returns, new_weights, horizon_days=10)

    t1_before, _ = monte_carlo_var(log_returns, weights, horizon_days=1, distribution="student_t", seed=42)
    t1_after, _ = monte_carlo_var(log_returns, new_weights, horizon_days=1, distribution="student_t", seed=42)

    cov_d = log_returns.cov().values
    p_vol_after = float(np.sqrt(new_weights.values.T @ cov_d @ new_weights.values))
    rc_after_vec = (new_weights.values * (cov_d @ new_weights.values)) / (p_vol_after ** 2)
    enb_risk_after = float(1.0 / np.sum(rc_after_vec ** 2))

    rows_rebal = [
        {
            "Portfolio Risk Metric": f"Annualized Volatility ({base_ccy})",
            "Current Portfolio": f"{vol_before:.2f}%",
            "Simulated Rebalanced": f"{vol_after:.2f}%",
            "Change (Risk Reduction)": f"{vol_after - vol_before:+.2f} pp",
        },
        {
            "Portfolio Risk Metric": "1-Day 95% VaR [Parametric]",
            "Current Portfolio": f"-{p1_before['VaR_95']*100:.2f}% (-{base_ccy} {p1_before['VaR_95']*total_nav:,.0f})",
            "Simulated Rebalanced": f"-{p1_after['VaR_95']*100:.2f}% (-{base_ccy} {p1_after['VaR_95']*total_nav:,.0f})",
            "Change (Risk Reduction)": f"{(p1_after['VaR_95'] - p1_before['VaR_95'])*100:+.2f} pp ({base_ccy} {(p1_after['VaR_95'] - p1_before['VaR_95'])*total_nav:+,.0f})",
        },
        {
            "Portfolio Risk Metric": "10-Day 95% VaR [Parametric]",
            "Current Portfolio": f"-{p10_before['VaR_95']*100:.2f}% (-{base_ccy} {p10_before['VaR_95']*total_nav:,.0f})",
            "Simulated Rebalanced": f"-{p10_after['VaR_95']*100:.2f}% (-{base_ccy} {p10_after['VaR_95']*total_nav:,.0f})",
            "Change (Risk Reduction)": f"{(p10_after['VaR_95'] - p10_before['VaR_95'])*100:+.2f} pp ({base_ccy} {(p10_after['VaR_95'] - p10_before['VaR_95'])*total_nav:+,.0f})",
        },
        {
            "Portfolio Risk Metric": "1-Day 99% CVaR [Student-t]",
            "Current Portfolio": f"-{t1_before['CVaR_99']*100:.2f}% (-{base_ccy} {t1_before['CVaR_99']*total_nav:,.0f})",
            "Simulated Rebalanced": f"-{t1_after['CVaR_99']*100:.2f}% (-{base_ccy} {t1_after['CVaR_99']*total_nav:,.0f})",
            "Change (Risk Reduction)": f"{(t1_after['CVaR_99'] - t1_before['CVaR_99'])*100:+.2f} pp ({base_ccy} {(t1_after['CVaR_99'] - t1_before['CVaR_99'])*total_nav:+,.0f})",
        },
        {
            "Portfolio Risk Metric": "Effective Risk Bets (N_eff)",
            "Current Portfolio": f"{enb_risk:.1f} / {n_holdings}",
            "Simulated Rebalanced": f"{enb_risk_after:.1f} / {n_holdings}",
            "Change (Risk Reduction)": f"{enb_risk_after - enb_risk:+.1f} bets",
        },
    ]

    if holding_meta and len(ccy_df) > 1:
        non_base_mask = np.array([holding_meta[t]["currency"] != base_ccy for t in weights.index])
        fx_risk_after = float(np.sum(rc_after_vec[non_base_mask])) * 100.0
        rows_rebal.append({
            "Portfolio Risk Metric": f"Non-{base_ccy} Currency Risk Share",
            "Current Portfolio": f"{foreign_ccy_rc:.1f}%",
            "Simulated Rebalanced": f"{fx_risk_after:.1f}%",
            "Change (Risk Reduction)": f"{fx_risk_after - foreign_ccy_rc:+.1f} pp",
        })

    rebal_table = pd.DataFrame(rows_rebal)

    rebal_meta = {
        "n_triggered": n_triggered,
        "moved_total_pct": moved_total_pct,
        "trimmed_str": ", ".join(trimmed_names) if trimmed_names else f"None (no holdings exceeded {rwr_high:.2f}x Risk/Weight ratio)",
        "added_str": ", ".join(added_names) if added_names else f"None (no holdings below {rwr_low:.2f}x Risk/Weight ratio)",
        "rebal_table": rebal_table,
    }

    return flags_df, rebal_meta