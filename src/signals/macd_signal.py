"""
MACD Crossover signal.
Generates buy when MACD line crosses above signal line, sell when it crosses below.
Filters with ADX to avoid trading in low-momentum markets.
"""

import pandas as pd
from typing import Dict, Any
from src.signals.base import SignalBase
from src.features.indicators import calculate_macd, calculate_adx
from src.utils.logger import logger


class MACDSignal(SignalBase):
    """
    MACD crossover strategy with ADX filter.

    - BUY:  MACD histogram > 0 (MACD above signal) AND ADX > threshold
    - SELL: MACD histogram < 0 (MACD below signal) AND ADX > threshold
    - HOLD: ADX too low (no clear trend)

    Best in: Trending markets with clear momentum.
    Worst in: Sideways / choppy markets (whipsaws).
    """

    ADX_MIN_THRESHOLD: float = 20.0

    def __init__(self, config: Dict[str, Any]) -> None:
        signal_cfg = config.get("signals", {})
        self.macd_fast: int = int(signal_cfg.get("macd_fast", 12))
        self.macd_slow: int = int(signal_cfg.get("macd_slow", 26))
        self.macd_signal: int = int(signal_cfg.get("macd_signal", 9))
        self.adx_threshold: float = float(signal_cfg.get("adx_threshold", self.ADX_MIN_THRESHOLD))
        logger.info(
            "MACDSignal: fast=%d, slow=%d, signal=%d, adx_thresh=%.1f",
            self.macd_fast, self.macd_slow, self.macd_signal, self.adx_threshold,
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        min_bars = self.macd_slow + self.macd_signal + 14
        if len(data) < min_bars:
            return 0

        macd_df = calculate_macd(data["close"], self.macd_fast, self.macd_slow, self.macd_signal)
        adx = calculate_adx(data["high"], data["low"], data["close"], window=14)

        current_hist = float(macd_df["hist"].iloc[-1])
        prev_hist = float(macd_df["hist"].iloc[-2])
        current_adx = float(adx.iloc[-1])

        # ADX filter — don't trade in weak trends
        if current_adx < self.adx_threshold:
            return 0

        # Crossover detection: histogram flips sign
        if current_hist > 0 and prev_hist <= 0:
            return 1   # bullish crossover
        elif current_hist < 0 and prev_hist >= 0:
            return -1  # bearish crossover

        return 0
