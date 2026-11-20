"""
Module 2: Multi-Model VaR/CVaR Engine, Benchmark Comparison & Drift Sensitivity (v1.1)
Includes Historical, Parametric, Normal MC, and Student-t MC across 1-Day, 10-Day,
and 1-Year horizons, plus portfolio vs. benchmark summary metrics.
"""
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


def compute_var_cvar_from_pnl(
    pnl_distribution: np.ndarray,
    alphas: Tuple[float, float] = (0.95, 0.99)
) -> Dict[str, float]:
    """
    Given an array of simple returns (negative = loss), returns positive Loss VaR and CVaR
    decimals at the requested confidence levels.
    """
    results: Dict[str, float] = {}
    for alpha in alphas:
        q = np.percentile(pnl_distribution, (1.0 - alpha) * 100.0)
        cvar = pnl_distribution[pnl_distribution <= q].mean()
        label = int(alpha * 100)
        results[f"VaR_{label}"] = float(-q)
        results[f"CVaR_{label}"] = float(-cvar)
    return results


def historical_simulation_var(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int = 1
) -> Dict[str, float]:
    """Method 1: Non-parametric Historical Simulation."""
    simple_returns = np.exp(log_returns.values) - 1.0
    port_daily_returns = simple_returns @ weights.values

    if horizon_days > 1:
        port_series = pd.Series(np.log(1.0 + port_daily_returns))
        rolling_log_ret = port_series.rolling(window=horizon_days).sum().dropna().values
        port_horizon_returns = np.exp(rolling_log_ret) - 1.0
    else:
        port_horizon_returns = port_daily_returns

    return compute_var_cvar_from_pnl(port_horizon_returns)


def parametric_vcov_var(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int = 1,
    include_drift: bool = False
) -> Dict[str, float]:
    """Method 2: Parametric Variance-Covariance (Delta-Normal) Method."""
    w = weights.values
    cov_matrix = log_returns.cov().values * horizon_days
    port_variance = float(w.T @ cov_matrix @ w)
    port_std = np.sqrt(port_variance)
    port_mean = float(np.dot(log_returns.mean().values, w) * horizon_days) if include_drift else 0.0

    results: Dict[str, float] = {}
    for alpha in (0.95, 0.99):
        z = stats.norm.ppf(alpha)
        var_loss = -(port_mean - z * port_std)
        cvar_loss = -(port_mean - port_std * (stats.norm.pdf(z) / (1.0 - alpha)))
        label = int(alpha * 100)
        results[f"VaR_{label}"] = float(var_loss)
        results[f"CVaR_{label}"] = float(cvar_loss)
    return results


def monte_carlo_var(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int = 1,
    num_simulations: int = 25000,
    distribution: str = "normal",
    df_student: float | None = None,
    include_drift: bool = False,
    seed: int = 42
) -> Tuple[Dict[str, float], np.ndarray]:
    """Method 3 & 4: Vectorized Cholesky Monte Carlo (Gaussian vs. Variance-Matched Student-t)."""
    rng = np.random.default_rng(seed)
    n_assets = log_returns.shape[1]
    cov_daily = log_returns.cov().values
    var_daily = np.diag(cov_daily)

    raw_mean = log_returns.mean().values if include_drift else np.zeros(n_assets)
    ito_drift = (raw_mean - 0.5 * var_daily) * horizon_days

    cov_horizon = cov_daily * horizon_days
    L = np.linalg.cholesky(cov_horizon)
    Z = rng.standard_normal(size=(num_simulations, n_assets))

    if distribution == "normal":
        correlated_shocks = Z @ L.T
    elif distribution == "student_t":
        if df_student is None:
            port_hist = log_returns.values @ weights.values
            fitted_df, _, _ = stats.t.fit(port_hist)
            df_student = float(np.clip(fitted_df, 3.0, 15.0))

        chi2_draws = rng.chisquare(df=df_student, size=(num_simulations, 1))
        t_scaling = np.sqrt(df_student / chi2_draws) * np.sqrt((df_student - 2.0) / df_student)
        correlated_shocks = (Z @ L.T) * t_scaling
    else:
        raise ValueError("distribution must be 'normal' or 'student_t'")

    sim_log_returns = correlated_shocks + ito_drift
    sim_simple_returns = np.exp(sim_log_returns) - 1.0
    port_sim_returns = sim_simple_returns @ weights.values

    metrics = compute_var_cvar_from_pnl(port_sim_returns)
    if distribution == "student_t" and df_student is not None:
        metrics["fitted_df"] = float(df_student)

    return metrics, port_sim_returns


def compute_portfolio_vs_benchmark_summary(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    bm_log_returns: pd.Series,
    rf_rate_annual: float = 0.025,
) -> Dict[str, Any]:
    """
    Computes annualized return, volatility, Sharpe, Sortino, Beta to Benchmark,
    Diversification Ratio, Effective Number of Bets, and 1-Day / 10-Day VaR & CVaR
    for both the Portfolio and the MSCI World (EUR) Benchmark over the identical window.
    """
    w = weights.values
    port_simple = (np.exp(log_returns.values) - 1.0) @ w
    bm_simple = np.exp(bm_log_returns.values) - 1.0

    # Annualized geom/compound return and volatility
    n_years = len(port_simple) / 252.0
    port_cum = float(np.prod(1.0 + port_simple))
    bm_cum = float(np.prod(1.0 + bm_simple))
    port_ann_ret = (port_cum ** (1.0 / n_years)) - 1.0
    bm_ann_ret = (bm_cum ** (1.0 / n_years)) - 1.0

    port_ann_vol = float(np.sqrt(w.T @ (log_returns.cov().values * 252.0) @ w))
    bm_ann_vol = float(np.std(bm_log_returns.values, ddof=1) * np.sqrt(252.0))

    # Sharpe & Sortino (using downside deviation below daily risk-free rate)
    rf_daily = rf_rate_annual / 252.0
    port_sharpe = (port_ann_ret - rf_rate_annual) / port_ann_vol
    bm_sharpe = (bm_ann_ret - rf_rate_annual) / bm_ann_vol

    port_downside = np.minimum(0.0, port_simple - rf_daily)
    bm_downside = np.minimum(0.0, bm_simple - rf_daily)
    port_sortino = (port_ann_ret - rf_rate_annual) / (np.sqrt(np.mean(port_downside ** 2)) * np.sqrt(252.0))
    bm_sortino = (bm_ann_ret - rf_rate_annual) / (np.sqrt(np.mean(bm_downside ** 2)) * np.sqrt(252.0))

    # Portfolio Beta to external Benchmark & Correlation
    cov_pb = np.cov(port_simple, bm_simple, ddof=1)
    port_beta_bm = float(cov_pb[0, 1] / cov_pb[1, 1])
    port_corr_bm = float(np.corrcoef(port_simple, bm_simple)[0, 1])

    # Choueifaty Diversification Ratio: (sum w_i * sigma_i) / sigma_p
    asset_vols = np.sqrt(np.diag(log_returns.cov().values) * 252.0)
    weighted_avg_vol = float(np.dot(w, asset_vols))
    port_log_vol = float(np.sqrt(w.T @ (log_returns.cov().values * 252.0) @ w))
    diversification_ratio = weighted_avg_vol / port_log_vol

    # Effective Number of Capital Bets (1 / HHI_weights)
    enb_capital = float(1.0 / np.sum(w ** 2))

    # Benchmark 1-Day and 10-Day Historical & Parametric VaR/CVaR
    bm_df = pd.DataFrame({"BM": bm_log_returns})
    bm_w = pd.Series([1.0], index=["BM"])
    bm_hist_1d = historical_simulation_var(bm_df, bm_w, horizon_days=1)
    bm_param_1d = parametric_vcov_var(bm_df, bm_w, horizon_days=1)
    bm_param_10d = parametric_vcov_var(bm_df, bm_w, horizon_days=10)

    port_hist_1d = historical_simulation_var(log_returns, weights, horizon_days=1)
    port_param_1d = parametric_vcov_var(log_returns, weights, horizon_days=1)
    port_param_10d = parametric_vcov_var(log_returns, weights, horizon_days=10)
    port_t_10d, _ = monte_carlo_var(log_returns, weights, horizon_days=10, distribution="student_t", seed=42)

    # Standard error of historical mean return: sigma / sqrt(T_years)
    mean_se_pct = (port_ann_vol / np.sqrt(n_years)) * 100.0

    return {
        "n_years": n_years,
        "port_ann_ret_pct": port_ann_ret * 100.0,
        "bm_ann_ret_pct": bm_ann_ret * 100.0,
        "port_ann_vol_pct": port_ann_vol * 100.0,
        "bm_ann_vol_pct": bm_ann_vol * 100.0,
        "port_sharpe": port_sharpe,
        "bm_sharpe": bm_sharpe,
        "port_sortino": port_sortino,
        "bm_sortino": bm_sortino,
        "port_beta_bm": port_beta_bm,
        "port_corr_bm": port_corr_bm,
        "diversification_ratio": diversification_ratio,
        "weighted_avg_vol_pct": weighted_avg_vol * 100.0,
        "enb_capital": enb_capital,
        "mean_se_pct": mean_se_pct,
        "bm_hist_1d": bm_hist_1d,
        "bm_param_1d": bm_param_1d,
        "bm_param_10d": bm_param_10d,
        "port_hist_1d": port_hist_1d,
        "port_param_1d": port_param_1d,
        "port_param_10d": port_param_10d,
        "port_t_10d": port_t_10d,
    }


def simulate_horizon_and_drift_sensitivity(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    total_nav_eur: float,
    days: int = 252,
    num_simulations: int = 3000,
    equilibrium_erp_annual: float = 0.07,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulates 1-Year (252-day) paths under 4 transparent drift/volatility assumptions to avoid
    relying solely on a bull-market historical mean:
      1. Zero-Drift Pure Risk Baseline (mu = 0% p.a.)
      2. Equilibrium ERP Baseline (mu = +7.0% p.a. across assets) -> used for Panel E Fan Chart
      3. Historical In-Sample Mean (2021-2026 sample drift, shown with estimation-error caveat)
      4. Stressed Crisis Regime (-20% annual return shock, 2x volatility)
    """
    rng = np.random.default_rng(seed)
    n_assets = log_returns.shape[1]
    w = weights.values

    cov_daily = log_returns.cov().values
    var_daily = np.diag(cov_daily)
    hist_mean_daily = log_returns.mean().values
    L_normal = np.linalg.cholesky(cov_daily)

    Z = rng.standard_normal(size=(days, num_simulations, n_assets))
    corr_shocks_normal = Z @ L_normal.T

    # 1. Zero-Drift (mu = 0)
    ito_zero = -0.5 * var_daily
    cum_log_zero = np.cumsum(corr_shocks_normal + ito_zero, axis=0)
    paths_zero_eur = total_nav_eur * (np.exp(cum_log_zero) @ w)
    ret_1yr_zero = (paths_zero_eur[-1, :] / total_nav_eur) - 1.0

    # 2. Equilibrium ERP Drift (mu = 7.0% p.a.)
    eq_daily = np.full(n_assets, np.log(1.0 + equilibrium_erp_annual) / days)
    ito_eq = eq_daily - 0.5 * var_daily
    cum_log_eq = np.cumsum(corr_shocks_normal + ito_eq, axis=0)
    paths_eq_eur = total_nav_eur * (np.exp(cum_log_eq) @ w)
    ret_1yr_eq = (paths_eq_eur[-1, :] / total_nav_eur) - 1.0

    # 3. Historical Sample Drift (2021-2026)
    ito_hist = hist_mean_daily - 0.5 * var_daily
    cum_log_hist = np.cumsum(corr_shocks_normal + ito_hist, axis=0)
    ret_1yr_hist = (np.exp(cum_log_hist[-1, :, :]) @ w) - 1.0

    # 4. Stressed Crisis Regime (-20% return shock from equilibrium, 2x volatility)
    vol_mult = 2.0
    cov_stress = cov_daily * (vol_mult ** 2)
    var_stress = np.diag(cov_stress)
    L_stress = np.linalg.cholesky(cov_stress)
    stress_daily = np.full(n_assets, (np.log(1.0 + equilibrium_erp_annual) - 0.20) / days)
    ito_stress = stress_daily - 0.5 * var_stress
    cum_log_stress = np.sum((Z @ L_stress.T) + ito_stress, axis=0)
    ret_1yr_stress = (np.exp(cum_log_stress) @ w) - 1.0

    return {
        "paths_eq_eur": paths_eq_eur,
        "paths_zero_eur": paths_zero_eur,
        "ret_1yr_zero": ret_1yr_zero,
        "ret_1yr_eq": ret_1yr_eq,
        "ret_1yr_hist": ret_1yr_hist,
        "ret_1yr_stress": ret_1yr_stress,
    }