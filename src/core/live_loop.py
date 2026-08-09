"""
Live trading loop module.
Orchestrates real-time paper/live trading using the ensemble signal,
risk manager, broker, journal, and optional voice alerts.

Loop lifecycle
--------------
1. Fetch latest OHLCV bar from the data feed.
2. Update equity and check RiskManager halt status.
3. Check SL/TP on any open position (same priority as BacktestEngine).
4. Generate ensemble signal on the updated history window.
5. Execute buy or sell via the broker.
6. Log every tick to the TradeJournal (price + equity + trades).
7. Fire optional voice alerts.
8. Sleep until the next bar boundary.

Graceful shutdown
-----------------
Send SIGINT (Ctrl-C) or set ``loop.stop()`` to exit cleanly after the
current bar completes. All open positions are NOT force-liquidated on
shutdown by default — pass ``liquidate_on_exit=True`` if desired.
"""

import signal
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.execution.broker_base import BrokerBase
from src.notifications.voice_alerts import alert_entry, alert_exit, alert_halt
from src.risk.risk_manager import RiskManager
from src.signals.base import SignalBase
from src.state.journal import TradeJournal
from src.utils.logger import logger

# Seconds between polling the data feed when the bar has not yet closed.
_POLL_INTERVAL_SECONDS: int = 5

# Number of historical bars to maintain in the rolling window fed to signals.
_HISTORY_WINDOW: int = 200


class LiveLoop:
    """
    Real-time paper / live trading orchestrator.

    Args:
        config: Application configuration dict (from load_config).
        broker: Concrete broker implementation (PaperBroker or live CCXT broker).
        risk_manager: Initialised RiskManager instance.
        signal_generator: Any ``SignalBase`` implementation (typically EnsembleSignal).
        journal: Initialised TradeJournal for persistence.
        symbol: Trading pair, e.g. ``'BTC/USDT'``.
        liquidate_on_exit: Whether to force-sell the position on shutdown.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        broker: BrokerBase,
        risk_manager: RiskManager,
        signal_generator: SignalBase,
        journal: TradeJournal,
        symbol: str = "BTC/USDT",
        liquidate_on_exit: bool = False,
    ) -> None:
        self.config = config
        self.broker = broker
        self.risk_manager = risk_manager
        self.signal_generator = signal_generator
        self.journal = journal
        self.symbol = symbol
        self.liquidate_on_exit = liquidate_on_exit
        self.voice_enabled: bool = bool(config.get("voice_alerts", False))
        self.trading_fee: float = float(config.get("trading_fee", 0.001))

        # Mutable per-position state (mirrors BacktestEngine pattern)
        self._entry_price: Optional[float] = None
        self._entry_qty: Optional[float] = None
        self._entry_fee: float = 0.0
        self._stop_loss: Optional[float] = None
        self._take_profit: Optional[float] = None

        self._running: bool = False
        self._history: pd.DataFrame = pd.DataFrame()

        # Register SIGINT / SIGTERM for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(
            "LiveLoop initialised — symbol=%s, voice=%s, liquidate_on_exit=%s",
            self.symbol,
            self.voice_enabled,
            self.liquidate_on_exit,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, feed: Any) -> None:
        """Start the live trading loop.

        Args:
            feed: Data feed object implementing ``fetch_ohlcv(symbol, timeframe)``
                  and returning a DataFrame with OHLCV columns.  Compatible
                  with ``src.data.feed.DataFeed``.
        """
        self._running = True
        logger.info("LiveLoop started. Press Ctrl-C to stop.")

        timeframe: str = self.config.get("timeframe", "1h")
        prev_date: Optional[Any] = None

        while self._running:
            try:
                new_bar: pd.DataFrame = feed.fetch_ohlcv(
                    self.symbol, timeframe, limit=_HISTORY_WINDOW
                )
                if new_bar is None or new_bar.empty:
                    logger.warning("Feed returned empty bar — skipping tick.")
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    continue

                self._history = new_bar.tail(_HISTORY_WINDOW)
                current_row = self._history.iloc[-1]
                current_price = float(current_row["close"])
                current_high = float(current_row["high"])
                current_low = float(current_row["low"])
                current_time = str(self._history.index[-1])

                # -- Day boundary detection (daily-loss reset) --
                current_date = pd.Timestamp(self._history.index[-1]).date()
                is_new_day = prev_date is not None and current_date != prev_date
                prev_date = current_date

                # -- Equity update --
                pos_qty = self.broker.get_positions().get(self.symbol, 0.0)
                current_equity = (
                    self.broker.get_balance() + pos_qty * current_price
                )
                self.risk_manager.update_equity(
                    current_equity, is_new_day=is_new_day
                )

                # -- Journal: price + equity tick --
                self.journal.log_price(current_time, current_price)
                self.journal.log_equity(current_time, current_equity)

                # ── EXIT PRIORITY 1: risk halt ──
                if self.risk_manager.halted:
                    logger.warning("RiskManager halted — no new trades accepted.")
                    alert_halt("max drawdown or daily loss breached", self.voice_enabled)
                    if pos_qty > 0 and self._entry_price is not None:
                        self._execute_exit(
                            pos_qty, current_price, current_time, reason="risk_halt"
                        )
                    time.sleep(_POLL_INTERVAL_SECONDS)
                    continue

                # ── EXIT PRIORITY 2 & 3: SL / TP ──
                if pos_qty > 0 and self._entry_price is not None:
                    if (
                        self._stop_loss is not None
                        and current_low <= self._stop_loss
                    ):
                        self._execute_exit(
                            pos_qty, self._stop_loss, current_time, reason="stop_loss"
                        )
                        time.sleep(_POLL_INTERVAL_SECONDS)
                        continue

                    if (
                        self._take_profit is not None
                        and current_high >= self._take_profit
                    ):
                        self._execute_exit(
                            pos_qty, self._take_profit, current_time, reason="take_profit"
                        )
                        time.sleep(_POLL_INTERVAL_SECONDS)
                        continue

                # ── EXIT PRIORITY 4 / ENTRY: signal ──
                signal_val = self.signal_generator.generate_signal(self._history)

                if signal_val == -1 and pos_qty > 0 and self._entry_price is not None:
                    self._execute_exit(
                        pos_qty, current_price, current_time, reason="signal"
                    )

                elif signal_val == 1 and pos_qty == 0:
                    from src.features.indicators import calculate_atr
                    atr_series = calculate_atr(
                        self._history["high"],
                        self._history["low"],
                        self._history["close"],
                        window=14,
                    )
                    current_atr = float(atr_series.iloc[-1])
                    qty = self.risk_manager.calculate_position_size(
                        self.broker.get_balance(), current_price, current_atr
                    )
                    if qty > 0:
                        self._execute_entry(
                            qty, current_price, current_time, current_atr
                        )

            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("LiveLoop tick error: %s", exc, exc_info=True)
                time.sleep(_POLL_INTERVAL_SECONDS)

        self._shutdown()

    def stop(self) -> None:
        """Signal the loop to exit after the current tick completes."""
        logger.info("LiveLoop stop requested.")
        self._running = False

    # ── Internal helpers ─────────────────────────────────────────────────

    def _execute_entry(
        self,
        qty: float,
        price: float,
        timestamp: str,
        atr: float,
    ) -> None:
        """Submit a buy order and record entry state."""
        res = self.broker.submit_order(self.symbol, qty, "buy", price=price)
        if res.get("status") == "filled":
            entry_fee = res["price"] * qty * self.trading_fee
            self._entry_price = res["price"]
            self._entry_qty = qty
            self._entry_fee = entry_fee
            sl, tp = self.risk_manager.calculate_sl_tp(
                entry_price=res["price"], direction=1, atr=atr
            )
            self._stop_loss = sl
            self._take_profit = tp

            trade = {
                "timestamp": timestamp,
                "side": "buy",
                "price": res["price"],
                "qty": qty,
                "fee": entry_fee,
                "sl": sl,
                "tp": tp,
            }
            self.journal.log_trade(trade, symbol=self.symbol)
            alert_entry(self.symbol, "buy", res["price"], self.voice_enabled)
            logger.info(
                "BUY %s qty=%.6f @ %.2f | SL=%.2f TP=%.2f",
                self.symbol, qty, res["price"],
                sl or 0, tp or 0,
            )

    def _execute_exit(
        self,
        pos_qty: float,
        exit_price: float,
        timestamp: str,
        reason: str,
    ) -> None:
        """Submit a sell order, compute net PnL, and persist to journal."""
        res = self.broker.submit_order(
            self.symbol, pos_qty, "sell", price=exit_price
        )
        if res.get("status") == "filled":
            exit_fee = res["price"] * pos_qty * self.trading_fee
            gross_pnl = (res["price"] - (self._entry_price or res["price"])) * pos_qty
            net_pnl = gross_pnl - self._entry_fee - exit_fee

            trade = {
                "timestamp": timestamp,
                "side": "sell",
                "price": res["price"],
                "qty": pos_qty,
                "fee": exit_fee,
                "pnl": net_pnl,
                "exit_reason": reason,
            }
            self.journal.log_trade(trade, symbol=self.symbol)
            alert_exit(
                self.symbol, reason, res["price"], net_pnl, self.voice_enabled
            )
            logger.info(
                "SELL %s qty=%.6f @ %.2f | reason=%s net_pnl=%.2f",
                self.symbol, pos_qty, res["price"], reason, net_pnl,
            )

        # Always reset entry state
        self._entry_price = None
        self._entry_qty = None
        self._entry_fee = 0.0
        self._stop_loss = None
        self._take_profit = None

    def _shutdown(self) -> None:
        """Clean up resources on loop exit."""
        logger.info("LiveLoop shutting down.")
        if self.liquidate_on_exit:
            pos_qty = self.broker.get_positions().get(self.symbol, 0.0)
            if pos_qty > 0 and self._entry_price is not None:
                # Best-effort liquidation at last known price
                if not self._history.empty:
                    last_price = float(self._history.iloc[-1]["close"])
                    ts = str(datetime.now(timezone.utc).isoformat())
                    self._execute_exit(pos_qty, last_price, ts, reason="shutdown")
        self.journal.close()
        logger.info("LiveLoop stopped cleanly.")

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Signal handler for SIGINT / SIGTERM."""
        logger.info("Signal %d received — stopping LiveLoop.", signum)
        self._running = False
