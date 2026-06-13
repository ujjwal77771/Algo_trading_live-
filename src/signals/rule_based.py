"""
Rule-based signal module.
Generates signals based on technical indicators.
"""

import pandas as pd
from typing import Dict, Any
from src.signals.base import SignalBase
from src.features.indicators import calculate_ema, calculate_rsi


class RuleBasedSignal(SignalBase):
    """
    Generates signals based on Trend (EMA cross) and Momentum (RSI) filter.

    Config keys (under 'signals' block in settings.yaml):
        ema_fast: int — fast EMA window (e.g. 12)
        ema_slow: int — slow EMA window (e.g. 26)
        rsi_window: int — RSI lookback (e.g. 14)
        rsi_overbought: float — RSI threshold for overbought (e.g. 70)
        rsi_oversold: float — RSI threshold for oversold (e.g. 30)
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        signal_config = config.get('signals')
        if signal_config is None:
            raise KeyError(
                "Missing required configuration block: 'signals'. "
                "Add ema_fast, ema_slow, rsi_window, rsi_overbought, "
                "rsi_oversold to config/settings.yaml under 'signals:'."
            )

        required_keys = ['ema_fast', 'ema_slow', 'rsi_window', 'rsi_overbought', 'rsi_oversold']
        missing = [k for k in required_keys if k not in signal_config]
        if missing:
            raise KeyError(
                f"Missing required signal config keys: {missing}. "
                f"Add them under 'signals:' in settings.yaml."
            )

        self.fast_window: int = int(signal_config['ema_fast'])
        self.slow_window: int = int(signal_config['ema_slow'])
        self.rsi_window: int = int(signal_config['rsi_window'])
        self.rsi_overbought: float = float(signal_config['rsi_overbought'])
        self.rsi_oversold: float = float(signal_config['rsi_oversold'])

        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"ema_fast ({self.fast_window}) must be < ema_slow ({self.slow_window})"
            )

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Returns 1 (Buy), -1 (Sell), or 0 (Hold).
        """
        min_bars = max(self.slow_window, self.rsi_window) + 1
        if len(data) < min_bars:
            return 0

        fast_ema = calculate_ema(data['close'], self.fast_window)
        slow_ema = calculate_ema(data['close'], self.slow_window)
        rsi = calculate_rsi(data['close'], self.rsi_window)

        current_fast = float(fast_ema.iloc[-1])
        current_slow = float(slow_ema.iloc[-1])
        current_rsi = float(rsi.iloc[-1])

        # Bullish trend + RSI not overbought
        if current_fast > current_slow and current_rsi < self.rsi_overbought:
            return 1
        # Bearish trend + RSI not oversold
        elif current_fast < current_slow and current_rsi > self.rsi_oversold:
            return -1

        return 0
