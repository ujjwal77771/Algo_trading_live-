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
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.fast_window = int(config.get('sma_short', 3))
        self.slow_window = int(config.get('sma_long', 5))
        self.rsi_window = 14
        
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Returns 1 (Buy), -1 (Sell), or 0 (Hold).
        """
        if len(data) < self.slow_window or len(data) < self.rsi_window:
            return 0
            
        fast_ema = calculate_ema(data['close'], self.fast_window)
        slow_ema = calculate_ema(data['close'], self.slow_window)
        rsi = calculate_rsi(data['close'], self.rsi_window)
        
        current_fast = float(fast_ema.iloc[-1])
        current_slow = float(slow_ema.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        
        # Bullish trend + RSI not overbought
        if current_fast > current_slow and current_rsi < 70:
            return 1
        # Bearish trend + RSI not oversold
        elif current_fast < current_slow and current_rsi > 30:
            return -1
            
        return 0
