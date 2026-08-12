"""
Triple EMA Trend signal.
Uses 3 EMAs (fast/mid/slow) for strong trend confirmation.
Only enters when all three EMAs align in the same direction.
"""

import pandas as pd
from typing import Dict, Any
from src.signals.base import SignalBase
from src.features.indicators import calculate_ema, calculate_atr
from src.utils.logger import logger


class TripleEMASignal(SignalBase):
    """
    Triple EMA trend-following strategy.

    Uses EMA(8), EMA(21), EMA(55) — enters only when all three
    are aligned (fast > mid > slow for bullish, or vice versa).

    - BUY:  EMA_8 > EMA_21 > EMA_55 (strong uptrend)
    - SELL: EMA_8 < EMA_21 < EMA_55 (strong downtrend)
    - HOLD: EMAs are mixed (no clear trend)

    Best in: Strong trending markets (catches big moves).
    Worst in: Choppy markets (slow to enter, late to exit).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        signal_cfg = config.get("signals", {})
        self.ema_fast: int = int(signal_cfg.get("triple_ema_fast", 8))
        self.ema_mid: int = int(signal_cfg.get("triple_ema_mid", 21))
        self.ema_slow: int = int(signal_cfg.get("triple_ema_slow", 55))
        logger.info(
            "TripleEMASignal: fast=%d, mid=%d, slow=%d",
            self.ema_fast, self.ema_mid, self.ema_slow,
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        min_bars = self.ema_slow + 5
        if len(data) < min_bars:
            return 0

        fast = float(calculate_ema(data["close"], self.ema_fast).iloc[-1])
        mid = float(calculate_ema(data["close"], self.ema_mid).iloc[-1])
        slow = float(calculate_ema(data["close"], self.ema_slow).iloc[-1])

        # All three aligned bullish
        if fast > mid > slow:
            return 1
        # All three aligned bearish
        elif fast < mid < slow:
            return -1

        return 0
