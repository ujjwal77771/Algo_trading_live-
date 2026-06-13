"""
Risk manager module.
Handles position sizing, SL/TP, and drawdowns.
All parameters are config-driven — no magic numbers.
"""

from typing import Dict, Any, Optional, Tuple
from src.utils.logger import logger


class RiskManager:
    """
    Manages trading risk parameters and position sizing.

    Config keys (top-level in settings.yaml):
        risk_per_trade_pct: float — fraction of capital risked per trade
        atr_multiplier: float — ATR multiple for stop-loss distance
        max_drawdown: float — peak-to-trough drawdown threshold to halt
        max_daily_loss: float — intra-day loss threshold to halt
        reward_risk_ratio: float — TP distance = SL distance × this value
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes RiskManager with configuration.

        Raises:
            KeyError: If required config keys are missing.
        """
        required_keys = [
            'risk_per_trade_pct', 'atr_multiplier',
            'max_drawdown', 'max_daily_loss', 'reward_risk_ratio',
        ]
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise KeyError(
                f"Missing required risk config keys: {missing}. "
                f"Add them to config/settings.yaml."
            )

        self.risk_per_trade: float = float(config['risk_per_trade_pct'])
        self.atr_multiplier: float = float(config['atr_multiplier'])
        self.max_drawdown: float = float(config['max_drawdown'])
        self.max_daily_loss: float = float(config['max_daily_loss'])
        self.reward_risk_ratio: float = float(config['reward_risk_ratio'])

        # State
        self.peak_equity: float = 0.0
        self.start_of_day_equity: float = 0.0
        self.halted: bool = False

    def update_equity(self, current_equity: float, is_new_day: bool = False) -> None:
        """
        Updates internal equity state and checks drawdown halts.
        """
        if self.peak_equity == 0.0:
            self.peak_equity = current_equity
            self.start_of_day_equity = current_equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if is_new_day:
            self.start_of_day_equity = current_equity

        current_dd = (self.peak_equity - current_equity) / self.peak_equity
        daily_loss = (
            (self.start_of_day_equity - current_equity) / self.start_of_day_equity
            if self.start_of_day_equity > 0
            else 0.0
        )

        if current_dd >= self.max_drawdown:
            logger.warning(
                f"Max drawdown reached: {current_dd:.2%}. Halting trading."
            )
            self.halted = True

        if daily_loss >= self.max_daily_loss:
            logger.warning(
                f"Max daily loss reached: {daily_loss:.2%}. Halting trading."
            )
            self.halted = True

    def calculate_position_size(
        self, capital: float, current_price: float, atr: float
    ) -> float:
        """
        Calculates position size based on risk per trade and ATR.
        Returns 0 if halted or inputs are invalid.
        """
        if self.halted or atr <= 0.0 or current_price <= 0.0:
            return 0.0

        risk_amount = capital * self.risk_per_trade
        stop_loss_distance = atr * self.atr_multiplier

        if stop_loss_distance <= 0.0:
            return 0.0

        position_size = risk_amount / stop_loss_distance

        # Ensure we don't exceed capital
        max_size = capital / current_price
        return min(position_size, max_size)

    def calculate_sl_tp(
        self, entry_price: float, direction: int, atr: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates Stop Loss and Take Profit prices based on ATR.
        direction: 1 for long, -1 for short.
        """
        if atr <= 0.0 or direction == 0:
            return None, None

        stop_dist = atr * self.atr_multiplier
        tp_dist = stop_dist * self.reward_risk_ratio

        if direction == 1:
            return entry_price - stop_dist, entry_price + tp_dist
        else:
            return entry_price + stop_dist, entry_price - tp_dist
