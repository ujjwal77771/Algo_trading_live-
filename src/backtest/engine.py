"""
Backtest engine module.
Event-driven backtesting engine with explicit entry tracking, net-of-fees PnL,
and SL/TP enforcement.

Exit priority per bar (checked in this order):
  1. RiskManager drawdown/daily-loss halt → force liquidate
  2. Stop-loss hit (current low <= SL for longs)
  3. Take-profit hit (current high >= TP for longs)
  4. Signal flip to -1 → signal-based exit
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

    Tracks entry price, SL, and TP explicitly. Exits are triggered by
    stop-loss/take-profit breaches (checked before signal evaluation)
    or signal-based flips. PnL is net of entry + exit fees.
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
        self._stop_loss: Optional[float] = None
        self._take_profit: Optional[float] = None

    def _record_exit(
        self,
        symbol: str,
        pos_qty: float,
        exit_price: float,
        current_time: Any,
        reason: str,
    ) -> None:
        """
        Executes a sell order and records the trade with net PnL.
        Resets all entry state afterward.
        """
        res = self.broker.submit_order(
            symbol, pos_qty, 'sell', price=exit_price
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
                'exit_reason': reason,
            })

            logger.info(
                f"Exit ({reason}): price={res['price']:.2f}, "
                f"pnl={net_pnl:.2f}"
            )

        # Always reset regardless — order rejection shouldn't leave stale state
        self._entry_price = None
        self._entry_qty = None
        self._entry_fee = 0.0
        self._stop_loss = None
        self._take_profit = None

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
            current_high = float(current_row['high'])
            current_low = float(current_row['low'])
            current_time = current_data.index[-1]
            current_atr = float(atr_series.iloc[i])

            # Detect day boundary for daily-loss reset
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

            # --- EXIT PRIORITY 1: Risk manager halt → force liquidate ---
            if self.risk_manager.halted:
                if pos_qty > 0 and self._entry_price is not None:
                    self._record_exit(
                        symbol, pos_qty, current_price, current_time,
                        reason='risk_halt',
                    )
                continue

            # --- EXIT PRIORITY 2 & 3: SL/TP check (before signal) ---
            if pos_qty > 0 and self._entry_price is not None:
                # Stop-loss: use current_low to simulate intra-bar breach.
                # Fill at SL price (assumes limit/stop order would fill there).
                if self._stop_loss is not None and current_low <= self._stop_loss:
                    self._record_exit(
                        symbol, pos_qty, self._stop_loss, current_time,
                        reason='stop_loss',
                    )
                    continue  # No further action this bar after SL exit

                # Take-profit: use current_high to simulate intra-bar breach.
                if self._take_profit is not None and current_high >= self._take_profit:
                    self._record_exit(
                        symbol, pos_qty, self._take_profit, current_time,
                        reason='take_profit',
                    )
                    continue  # No further action this bar after TP exit

            # --- EXIT PRIORITY 4: Signal-based exit ---
            signal = self.signal_generator.generate_signal(current_data)

            if signal == -1 and pos_qty > 0 and self._entry_price is not None:
                self._record_exit(
                    symbol, pos_qty, current_price, current_time,
                    reason='signal',
                )

            elif signal == 1 and pos_qty == 0:
                # --- ENTRY ---
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

                        # Compute SL/TP at entry using RiskManager
                        sl, tp = self.risk_manager.calculate_sl_tp(
                            entry_price=res['price'],
                            direction=1,  # long only for now
                            atr=current_atr,
                        )
                        self._stop_loss = sl
                        self._take_profit = tp

                        self.trades.append({
                            'timestamp': current_time,
                            'side': 'buy',
                            'price': res['price'],
                            'qty': qty,
                            'fee': entry_fee,
                            'sl': sl,
                            'tp': tp,
                        })

                        logger.info(
                            f"Entry: price={res['price']:.2f}, qty={qty:.6f}, "
                            f"SL={sl:.2f}, TP={tp:.2f}"
                        )

        logger.info("Backtest completed.")
        return pd.DataFrame(self.equity_curve).set_index('timestamp')
