"""
Breakout signal.
Identifies price breakouts above N-period high or below N-period low.
Uses ATR for volatility confirmation (avoids false breakouts in dead markets).
"""

import pandas as pd
from typing import Dict, Any
from src.signals.base import SignalBase
from src.features.indicators import calculate_atr
from src.utils.logger import logger


class BreakoutSignal(SignalBase):
    """
    N-period high/low breakout strategy with ATR filter.

    - BUY:  Close breaks above the 20-bar high AND ATR above its median
    - SELL: Close breaks below the 20-bar low AND ATR above its median
    - HOLD: Price inside the range or volatility too low

    Best in: Markets transitioning from consolidation to trend.
    Worst in: Trendless, low-vol markets (many false breakouts).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        signal_cfg = config.get("signals", {})
        self.lookback: int = int(signal_cfg.get("breakout_lookback", 20))
        self.atr_window: int = int(signal_cfg.get("breakout_atr_window", 14))
        logger.info(
            "BreakoutSignal: lookback=%d, atr_window=%d",
            self.lookback, self.atr_window,
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        min_bars = max(self.lookback, self.atr_window) + 5
        if len(data) < min_bars:
            return 0

        close = data["close"]
        current_close = float(close.iloc[-1])

        # N-period high/low (excluding current bar to avoid lookahead)
        lookback_high = float(close.iloc[-(self.lookback + 1):-1].max())
        lookback_low = float(close.iloc[-(self.lookback + 1):-1].min())

        # ATR filter: only trade breakouts in volatile markets
        atr = calculate_atr(data["high"], data["low"], data["close"], self.atr_window)
        current_atr = float(atr.iloc[-1])
        median_atr = float(atr.iloc[-50:].median()) if len(atr) >= 50 else float(atr.median())

        if current_atr < median_atr:
            return 0  # volatility too low for a real breakout

        # Breakout above N-period high
        if current_close > lookback_high:
            return 1
        # Breakdown below N-period low
        elif current_close < lookback_low:
            return -1

        return 0
