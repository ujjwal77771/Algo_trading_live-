"""
Bollinger Band mean-reversion signal.
Buys when price drops below the lower band (oversold), sells when price
rises above the upper band (overbought).
"""

import pandas as pd
from typing import Dict, Any
from src.signals.base import SignalBase
from src.features.indicators import calculate_bollinger_pb, calculate_rsi
from src.utils.logger import logger


class BollingerSignal(SignalBase):
    """
    Bollinger Band mean-reversion strategy with RSI confirmation.

    - BUY:  %B < 0 (below lower band) AND RSI < oversold
    - SELL: %B > 1 (above upper band) AND RSI > overbought
    - HOLD: price inside bands

    Best in: Range-bound / sideways markets.
    Worst in: Strong trending markets (catches falling knives).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        signal_cfg = config.get("signals", {})
        self.bb_window: int = int(signal_cfg.get("bb_window", 20))
        self.bb_std: float = float(signal_cfg.get("bb_std", 2.0))
        self.rsi_window: int = int(signal_cfg.get("rsi_window", 14))
        self.rsi_overbought: float = float(signal_cfg.get("rsi_overbought", 70))
        self.rsi_oversold: float = float(signal_cfg.get("rsi_oversold", 30))
        logger.info(
            "BollingerSignal: bb_window=%d, bb_std=%.1f, rsi=%d",
            self.bb_window, self.bb_std, self.rsi_window,
        )

    def generate_signal(self, data: pd.DataFrame) -> int:
        min_bars = max(self.bb_window, self.rsi_window) + 5
        if len(data) < min_bars:
            return 0

        pb = calculate_bollinger_pb(data["close"], self.bb_window, self.bb_std)
        rsi = calculate_rsi(data["close"], self.rsi_window)

        current_pb = float(pb.iloc[-1])
        current_rsi = float(rsi.iloc[-1])

        # Below lower band + RSI oversold = mean-reversion buy
        if current_pb < 0.0 and current_rsi < self.rsi_oversold:
            return 1
        # Above upper band + RSI overbought = mean-reversion sell
        elif current_pb > 1.0 and current_rsi > self.rsi_overbought:
            return -1

        return 0
