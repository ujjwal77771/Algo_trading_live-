"""
Backtest engine module.
Event-driven backtesting engine with explicit entry tracking and net-of-fees PnL.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from src.execution.paper_broker import PaperBroker
from src.risk.risk_manager import RiskManager
from src.signals.base import SignalBase
from src.utils.logger import logger


class BacktestEngine:
    """
    Event-driven engine for running historical backtests.

    Tracks entry price explicitly so PnL is correct regardless of
    trade-list ordering. PnL is net of entry + exit fees.
    Day boundaries are detected from the timestamp index to correctly
    reset the RiskManager's daily-loss tracking.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        broker: PaperBroker,
        risk_manager: RiskManager,
        signal_generator: SignalBase,
        trading_fee: float = 0.001,
    ) -> None:
        self.data = data
        self.broker = broker
        self.risk_manager = risk_manager
        self.signal_generator = signal_generator
        self.trading_fee = trading_fee
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

        # Explicit entry tracking — not derived from self.trades[-1]
        self._entry_price: Optional[float] = None
        self._entry_qty: Optional[float] = None
        self._entry_fee: float = 0.0

    def run(self) -> pd.DataFrame:
        """
        Executes the backtest row by row to simulate real-time feed.
        Returns the equity curve dataframe.
        """
        logger.info("Starting backtest...")

        symbol = 'BTC/USDT'

        # Pre-compute ATR for risk management
        from src.features.indicators import calculate_atr
        atr_series = calculate_atr(
            self.data['high'], self.data['low'], self.data['close'], window=14
        )

        prev_date = None

        for i in range(50, len(self.data)):
            # Simulate real-time data availability
            current_data = self.data.iloc[:i + 1]
            current_row = current_data.iloc[-1]
            current_price = float(current_row['close'])
            current_time = current_data.index[-1]
            current_atr = float(atr_series.iloc[i])

            # --- Bug 6 fix: detect day boundary for daily-loss reset ---
            current_date = pd.Timestamp(current_time).date()
            is_new_day = prev_date is not None and current_date != prev_date
            prev_date = current_date

            # Evaluate equity
            pos_qty = self.broker.get_positions().get(symbol, 0.0)
            current_equity = self.broker.get_balance() + (pos_qty * current_price)
            self.risk_manager.update_equity(current_equity, is_new_day=is_new_day)

            self.equity_curve.append(
                {'timestamp': current_time, 'equity': current_equity}
            )

            if self.risk_manager.halted:
                # Liquidate if halted
                if pos_qty > 0:
                    self.broker.submit_order(
                        symbol, pos_qty, 'sell', price=current_price
                    )
                    self._entry_price = None
                    self._entry_qty = None
                    self._entry_fee = 0.0
                continue

            signal = self.signal_generator.generate_signal(current_data)

            if signal == 1 and pos_qty == 0:
                # Buy
                qty = self.risk_manager.calculate_position_size(
                    self.broker.get_balance(), current_price, current_atr
                )
                if qty > 0:
                    res = self.broker.submit_order(
                        symbol, qty, 'buy', price=current_price
                    )
                    if res.get('status') == 'filled':
                        entry_fee = res['price'] * qty * self.trading_fee
                        self._entry_price = res['price']
                        self._entry_qty = qty
                        self._entry_fee = entry_fee
                        self.trades.append({
                            'timestamp': current_time,
                            'side': 'buy',
                            'price': res['price'],
                            'qty': qty,
                            'fee': entry_fee,
                        })

            elif signal == -1 and pos_qty > 0 and self._entry_price is not None:
                # Sell — compute PnL net of both entry and exit fees
                res = self.broker.submit_order(
                    symbol, pos_qty, 'sell', price=current_price
                )
                if res.get('status') == 'filled':
                    exit_fee = res['price'] * pos_qty * self.trading_fee
                    gross_pnl = (res['price'] - self._entry_price) * pos_qty
                    net_pnl = gross_pnl - self._entry_fee - exit_fee

                    self.trades.append({
                        'timestamp': current_time,
                        'side': 'sell',
                        'price': res['price'],
                        'qty': pos_qty,
                        'fee': exit_fee,
                        'pnl': net_pnl,
                    })

                    # Reset entry state
                    self._entry_price = None
                    self._entry_qty = None
                    self._entry_fee = 0.0

        logger.info("Backtest completed.")
        return pd.DataFrame(self.equity_curve).set_index('timestamp')
