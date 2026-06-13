"""
Configuration loader module.
Validates required keys from settings.yaml.
Fail-fast: missing any required key raises immediately on startup.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

REQUIRED_KEYS = [
    "environment",
    "base_asset",
    "quote_asset",
    "broker_mode",
    "timeframe",
    "initial_capital",
    "trading_fee",
    "risk_per_trade_pct",
    "atr_multiplier",
    "max_drawdown",
    "max_daily_loss",
    "reward_risk_ratio",
]

REQUIRED_ML_KEYS = [
    "lookback",
    "horizon",
    "model_dir",
]

REQUIRED_SIGNAL_KEYS = [
    "ema_fast",
    "ema_slow",
    "rsi_window",
    "rsi_overbought",
    "rsi_oversold",
]


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Loads and validates the configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required key is missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- Top-level keys ---
    missing_top = [k for k in REQUIRED_KEYS if k not in config]
    if missing_top:
        raise KeyError(f"Missing required top-level config keys: {missing_top}")

    # --- ML block ---
    if "ml" not in config or not isinstance(config["ml"], dict):
        raise KeyError("Missing required nested configuration dictionary: 'ml'")

    missing_ml = [k for k in REQUIRED_ML_KEYS if k not in config["ml"]]
    if missing_ml:
        raise KeyError(f"Missing required config keys in 'ml' block: {missing_ml}")

    # --- Signals block ---
    if "signals" not in config or not isinstance(config["signals"], dict):
        raise KeyError("Missing required nested configuration dictionary: 'signals'")

    missing_sig = [k for k in REQUIRED_SIGNAL_KEYS if k not in config["signals"]]
    if missing_sig:
        raise KeyError(
            f"Missing required config keys in 'signals' block: {missing_sig}"
        )

    return config
