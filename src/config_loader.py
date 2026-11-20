"""
Module 0: Configuration Schema Loader, Validator & Privacy Output Router (v1.2)
Validates user-supplied portfolio JSON files with friendly error messages and routes
private runs to gitignored directories.
"""
from typing import Dict, Any, List
import json
from pathlib import Path


SUPPORTED_CURRENCIES = {"EUR", "USD", "GBP", "CHF", "JPY", "CAD", "AUD", "SEK", "NOK", "DKK"}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "base_currency": "EUR",
    "start_date": "2021-01-01",
    "end_date": None,
    "max_ffill_days": 5,
    "min_backtest_days": 500,
    "rolling_window_days": 252,
    "benchmark_ticker": "IWDA.AS",
    "benchmark_currency": "EUR",
    "benchmark_crisis_proxy": "SPY",
    "benchmark_crisis_proxy_currency": "USD",
    "risk_free_rate": 0.025,
    "equilibrium_erp": 0.07,
    "confidence_levels": [0.95, 0.99],
    "horizons_days": [1, 10, 252],
    "random_seed": 42,
}

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "single_pos_equal_share_mult": 1.65,
    "risk_weight_ratio_high": 1.25,
    "risk_weight_ratio_low": 0.80,
    "sector_equal_share_mult": 1.50,
    "min_sectors_for_flag": 4,
    "foreign_ccy_risk_max_pct": 65.0,
    "enb_to_n_min_ratio": 0.60,
    "max_dd_excess_vs_bm_pct": 3.0,
    "kupiec_p_value_min": 0.05,
    "avg_corr_high": 0.50,
    "max_pair_corr_high": 0.75,
    "rebalance_trim_pct_per_pos": 1.5,
    "rebalance_top_k": 3,
}


class ConfigValidationError(ValueError):
    """Raised when a user-supplied portfolio configuration fails validation."""


def validate_config_dict(raw_cfg: Dict[str, Any], source_label: str = "config") -> Dict[str, Any]:
    """
    Validates a configuration dictionary, normalizes settings/thresholds, and checks
    every holding's sizing mode ('shares', 'amount', or 'weight'), currency, sector, and region.
    """
    if not isinstance(raw_cfg, dict):
        raise ConfigValidationError(f"[{source_label}] Root configuration must be a JSON object/dict.")

    holdings = raw_cfg.get("holdings")
    if not isinstance(holdings, list) or len(holdings) < 2:
        raise ConfigValidationError(
            f"[{source_label}] 'holdings' must be a list of at least 2 positions (found {len(holdings) if isinstance(holdings, list) else 0})."
        )

    settings = {**DEFAULT_SETTINGS, **(raw_cfg.get("settings") or {})}
    thresholds = {**DEFAULT_THRESHOLDS, **(raw_cfg.get("advisory_thresholds") or {})}

    base_ccy = str(settings["base_currency"]).upper().strip()
    if base_ccy not in SUPPORTED_CURRENCIES:
        raise ConfigValidationError(
            f"[{source_label}] Unsupported base_currency '{base_ccy}'. Supported: {sorted(SUPPORTED_CURRENCIES)}"
        )
    settings["base_currency"] = base_ccy

    for alpha in settings["confidence_levels"]:
        if not (0.50 <= float(alpha) < 1.0):
            raise ConfigValidationError(
                f"[{source_label}] confidence_levels must be between 0.50 and 1.0 (got {alpha})."
            )

    seen_tickers = set()
    normalized_holdings: List[Dict[str, Any]] = []
    sizing_modes = set()

    for idx, item in enumerate(holdings):
        if not isinstance(item, dict):
            raise ConfigValidationError(f"[{source_label}] Holding #{idx + 1} must be an object/dict.")

        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker:
            raise ConfigValidationError(f"[{source_label}] Holding #{idx + 1} is missing a 'ticker' string.")
        if ticker in seen_tickers:
            raise ConfigValidationError(f"[{source_label}] Duplicate ticker '{ticker}' in holdings list.")
        seen_tickers.add(ticker)

        ccy = str(item.get("currency", "")).strip().upper()
        if ccy not in SUPPORTED_CURRENCIES:
            raise ConfigValidationError(
                f"[{source_label}] Holding '{ticker}' has unsupported currency '{ccy}'. Supported: {sorted(SUPPORTED_CURRENCIES)}"
            )

        sector = str(item.get("sector", "Unclassified")).strip() or "Unclassified"
        region = str(item.get("region", "Global")).strip() or "Global"
        alias = str(item.get("alias", f"POS_{idx + 1:02d}")).strip()

        # Sizing validation: exactly one of 'shares', 'amount', or 'weight'
        present_sizing = [k for k in ("shares", "amount", "weight") if k in item and item[k] is not None]
        if len(present_sizing) == 0:
            raise ConfigValidationError(
                f"[{source_label}] Holding '{ticker}' must specify one of 'shares', 'amount', or 'weight'."
            )
        mode = present_sizing[0]
        val = float(item[mode])
        if val <= 0.0:
            raise ConfigValidationError(
                f"[{source_label}] Holding '{ticker}' must have a strictly positive '{mode}' (got {val})."
            )
        sizing_modes.add(mode)

        normalized_holdings.append({
            "ticker": ticker,
            "sizing_mode": mode,
            "sizing_value": val,
            "currency": ccy,
            "sector": sector,
            "region": region,
            "alias": alias,
        })

    if len(sizing_modes) > 1:
        raise ConfigValidationError(
            f"[{source_label}] Mixed position sizing modes {sorted(sizing_modes)} detected. "
            "Please use a consistent sizing mode ('shares', 'amount', or 'weight') across all holdings."
        )

    is_private = bool(raw_cfg.get("is_private", "private" in source_label.lower()))

    return {
        "portfolio_name": str(raw_cfg.get("portfolio_name", "Equity Portfolio")),
        "is_private": is_private,
        "illustrative_disclaimer": str(
            raw_cfg.get(
                "illustrative_disclaimer",
                "Educational risk analysis tool only. Does not constitute investment advice.",
            )
        ),
        "settings": settings,
        "advisory_thresholds": thresholds,
        "sector_proxy_map": raw_cfg.get("sector_proxy_map") or {},
        "stress_scenarios": raw_cfg.get("stress_scenarios") or [
            {
                "name": "auto_top_sector",
                "target_sector": "AUTO_TOP_RISK_SECTOR",
                "sector_shock": -0.25,
                "other_equity_shock": -0.08,
                "non_base_fx_shock": -0.08,
            }
        ],
        "holdings": normalized_holdings,
    }


def load_config_file(config_path: str | Path) -> Dict[str, Any]:
    """Loads and validates a JSON portfolio configuration file from disk."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigValidationError(
            f"Configuration file not found at '{path}'. "
            "Try running with '--config config/portfolio.example.json'."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Invalid JSON syntax in '{path}': {exc}") from exc

    return validate_config_dict(raw, source_label=path.name)


def resolve_output_dir(cfg: Dict[str, Any], base_dir: str | Path = "outputs") -> Path:
    """
    Routes output files to 'outputs/private/' (gitignored) when running on private club configs,
    and 'outputs/public/' when running on public/demo configs.
    """
    sub = "private" if cfg.get("is_private", False) else "public"
    out_dir = Path(base_dir) / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir