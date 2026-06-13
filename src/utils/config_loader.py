"""
Configuration loader module.
Validates required keys from settings.yaml.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

REQUIRED_KEYS = [
    "environment",
    "base_asset",
    "quote_asset",
    "broker_mode",
    "initial_capital",
    "trading_fee",
    "risk_per_trade_pct",
    "atr_multiplier",
    "max_drawdown",
    "max_daily_loss"
]

REQUIRED_ML_KEYS = [
    "lookback",
    "horizon",
    "model_dir"
]

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Loads and validates the configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required key is missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    for key in REQUIRED_KEYS:
        if key not in config:
            raise KeyError(f"Missing required configuration key: {key}")
            
    if "ml" not in config or not isinstance(config["ml"], dict):
        raise KeyError("Missing required nested configuration dictionary: 'ml'")
        
    for ml_key in REQUIRED_ML_KEYS:
        if ml_key not in config["ml"]:
            raise KeyError(f"Missing required configuration key in 'ml' block: {ml_key}")
            
    return config
