"""
ML features module.
Generates lagged features without lookahead bias.

Every feature uses ONLY data available at time t or earlier.
No .shift(-n), no forward-looking operations.

Feature categories (35 total):
  - Price returns (5): ret_1, ret_2, ret_5, ret_10, ret_20
  - Volatility (4): rolling_std_5, rolling_std_10, rolling_std_20, atr_14
  - Momentum (4): rsi_14, rsi_7, macd_hist, adx_14
  - Trend (3): ema_ratio_12, ema_ratio_26, ema_ratio_50
  - Volume (3): vol_pct_1, vol_pct_5, vol_ratio_sma20
  - Bollinger (1): bollinger_pb_20
  - Cross-sectional (2): hl_range_pct, co_range_pct
  - Lagged momentum (6): lag1/lag2 of rsi_14, macd_hist, adx_14
  - Lagged volatility (4): lag1/lag2 of rolling_std_5, atr_14
  - Return momentum (3): ret_5_lag5, ret_10_lag10, ret_20_lag20
"""

from typing import List

import pandas as pd

from src.features.indicators import (
    calculate_atr,
    calculate_adx,
    calculate_bollinger_pb,
    calculate_ema,
    calculate_macd,
    calculate_rolling_std,
    calculate_rsi,
)
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Named constants for rolling / lookback windows
# ---------------------------------------------------------------------------
RETURN_PERIODS: List[int] = [1, 2, 5, 10, 20]
VOLATILITY_WINDOWS: List[int] = [5, 10, 20]
ATR_WINDOW: int = 14
RSI_WINDOWS: List[int] = [14, 7]
ADX_WINDOW: int = 14
EMA_SPANS: List[int] = [12, 26, 50]
VOLUME_PCT_PERIODS: List[int] = [1, 5]
VOLUME_SMA_WINDOW: int = 20
BOLLINGER_WINDOW: int = 20
BOLLINGER_NUM_STD: float = 2.0
LAGGED_FEATURE_LAGS: List[int] = [1, 2]
RETURN_MOMENTUM_SPECS: List[tuple] = [(5, 5), (10, 10), (20, 20)]

# Minimum expected OHLCV columns
_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged features from OHLCV data with zero lookahead bias.

    Every feature column added uses only information available at or before
    the current bar.  The function returns a **copy** of the input DataFrame
    with new feature columns appended and NaN rows (produced by rolling
    windows) dropped.

    Args:
        df: DataFrame with at least columns
            ``['open', 'high', 'low', 'close', 'volume']``.
            The index should be a ``DatetimeIndex`` sorted ascending.

    Returns:
        A new DataFrame with the original OHLCV columns plus ~35 feature
        columns and no NaN values.

    Raises:
        ValueError: If required OHLCV columns are missing.
    """
    _validate_input(df)

    out = df.copy()
    logger.info("Building features for %d bars", len(out))

    # ------------------------------------------------------------------
    # 1. Price returns  (pct_change looks backward → no lookahead)
    # ------------------------------------------------------------------
    for period in RETURN_PERIODS:
        out[f"ret_{period}"] = out["close"].pct_change(periods=period)

    # ------------------------------------------------------------------
    # 2. Volatility
    # ------------------------------------------------------------------
    returns = out["close"].pct_change()
    for window in VOLATILITY_WINDOWS:
        out[f"rolling_std_{window}"] = calculate_rolling_std(returns, window)

    out["atr_14"] = calculate_atr(
        out["high"], out["low"], out["close"], window=ATR_WINDOW
    )

    # ------------------------------------------------------------------
    # 3. Momentum
    # ------------------------------------------------------------------
    for window in RSI_WINDOWS:
        out[f"rsi_{window}"] = calculate_rsi(out["close"], window=window)

    macd_df = calculate_macd(out["close"])
    out["macd_hist"] = macd_df["hist"]

    out["adx_14"] = calculate_adx(
        out["high"], out["low"], out["close"], window=ADX_WINDOW
    )

    # ------------------------------------------------------------------
    # 4. Trend — EMA ratios (close / EMA_n)
    # ------------------------------------------------------------------
    for span in EMA_SPANS:
        ema = calculate_ema(out["close"], window=span)
        out[f"ema_ratio_{span}"] = out["close"] / ema.replace(0, float("nan"))

    # ------------------------------------------------------------------
    # 5. Volume features
    # ------------------------------------------------------------------
    for period in VOLUME_PCT_PERIODS:
        out[f"vol_pct_{period}"] = out["volume"].pct_change(periods=period)

    vol_sma = out["volume"].rolling(window=VOLUME_SMA_WINDOW).mean()
    out["vol_ratio_sma20"] = out["volume"] / vol_sma.replace(0, float("nan"))

    # ------------------------------------------------------------------
    # 6. Bollinger %B
    # ------------------------------------------------------------------
    out["bollinger_pb_20"] = calculate_bollinger_pb(
        out["close"], window=BOLLINGER_WINDOW, num_std=BOLLINGER_NUM_STD
    )

    # ------------------------------------------------------------------
    # 7. Cross-sectional bar statistics
    # ------------------------------------------------------------------
    out["hl_range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["co_range_pct"] = (out["close"] - out["open"]) / out["close"]

    # ------------------------------------------------------------------
    # 8. Lagged momentum indicators (shift(+n) → backward, safe)
    # ------------------------------------------------------------------
    for lag in LAGGED_FEATURE_LAGS:
        out[f"rsi_14_lag{lag}"] = out["rsi_14"].shift(lag)
        out[f"macd_hist_lag{lag}"] = out["macd_hist"].shift(lag)
        out[f"adx_14_lag{lag}"] = out["adx_14"].shift(lag)

    # ------------------------------------------------------------------
    # 9. Lagged volatility indicators
    # ------------------------------------------------------------------
    for lag in LAGGED_FEATURE_LAGS:
        out[f"rolling_std_5_lag{lag}"] = out["rolling_std_5"].shift(lag)
        out[f"atr_14_lag{lag}"] = out["atr_14"].shift(lag)

    # ------------------------------------------------------------------
    # 10. Return momentum (lagged returns of different horizons)
    # ------------------------------------------------------------------
    for ret_period, lag in RETURN_MOMENTUM_SPECS:
        out[f"ret_{ret_period}_lag{lag}"] = out[f"ret_{ret_period}"].shift(lag)

    # ------------------------------------------------------------------
    # Drop NaN rows produced by rolling windows, log final shape
    # ------------------------------------------------------------------
    pre_drop_len = len(out)
    out = out.dropna()
    logger.info(
        "Features built: %d columns, %d bars (%d dropped for warm-up)",
        len(out.columns) - len(_REQUIRED_COLUMNS),
        len(out),
        pre_drop_len - len(out),
    )
    return out


def _validate_input(df: pd.DataFrame) -> None:
    """Check that the DataFrame has the minimum required OHLCV columns.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required OHLCV columns: {missing}"
        )
