"""
RSI Momentum signal.
Pure momentum strategy using dual-timeframe RSI.
"""

import pandas as pd
from typing import Dict, Any
from src.signals.base import SignalBase
from src.features.indicators import calculate_rsi
from src.utils.logger import logger


class RSIMomentumSignal(SignalBase):
    """
    Dual-timeframe RSI momentum strategy.

    Uses a fast RSI (7) for timing and slow RSI (14) for confirmation.

    - BUY:  Fast RSI crosses above oversold AND slow RSI < 50 (room to run up)
    - SELL: Fast RSI crosses below overbought AND slow RSI > 50 (room to fall)
    - HOLD: No clear momentum divergence

    Best in: Volatile markets with frequent pullbacks.
    Worst in: Low-volatility grind-up markets (too many false signals).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        signal_cfg = config.get("signals", {})
        self.rsi_fast: int = int(signal_cfg.get("rsi_fast", 7))
        self.rsi_slow: int = int(signal_cfg.get("rsi_slow", 14))
        self.overbought: float = float(signal_cfg.get("rsi_overbought", 70))
        self.oversold: float = float(signal_cfg.get("rsi_oversold", 30))
        logger.info(
            "RSIMomentumSignal: fast=%d, slow=%d, OB=%.0f, OS=%.0f",
            self.rsi_fast, self.rsi_slow, self.overbought, self.oversold,
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        min_bars = self.rsi_slow + 5
        if len(data) < min_bars:
            return 0

        rsi_fast = calculate_rsi(data["close"], self.rsi_fast)
        rsi_slow = calculate_rsi(data["close"], self.rsi_slow)

        curr_fast = float(rsi_fast.iloc[-1])
        prev_fast = float(rsi_fast.iloc[-2])
        curr_slow = float(rsi_slow.iloc[-1])

        # Fast RSI crossing UP through oversold + slow RSI has room
        if prev_fast < self.oversold and curr_fast >= self.oversold and curr_slow < 50:
            return 1
        # Fast RSI crossing DOWN through overbought + slow RSI has room
        elif prev_fast > self.overbought and curr_fast <= self.overbought and curr_slow > 50:
            return -1

        return 0
