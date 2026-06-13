"""
Technical indicators module.
Implements vectorized, leakage-safe indicators.
"""

import pandas as pd
import numpy as np

def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculates MACD."""
    fast_ema = calculate_ema(series, fast)
    slow_ema = calculate_ema(series, slow)
    macd = fast_ema - slow_ema
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return pd.DataFrame({'macd': macd, 'signal': signal_line, 'hist': histogram})

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    plus_dm_series = pd.Series(plus_dm, index=high.index)
    minus_dm_series = pd.Series(minus_dm, index=high.index)
    
    tr = calculate_atr(high, low, close, window=1) # True range per period
    
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm_series.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm_series.rolling(window=window).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=window).mean()
    return adx

def calculate_bollinger_pb(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Calculates Bollinger %B."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return (series - lower_band) / (upper_band - lower_band).replace(0, np.nan)

def calculate_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Calculates rolling standard deviation."""
    return series.rolling(window=window).std()
