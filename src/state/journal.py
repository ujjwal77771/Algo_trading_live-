"""
Trade journal module.
Persists all trades, price history, and equity curve to a SQLite database.

Schema
------
trades         — every filled order (buy + sell with net PnL)
price_history  — (timestamp, close) — consumed by the PyQt dashboard
equity_history — (timestamp, equity) — consumed by the PyQt dashboard
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.utils.logger import logger

# SQLite file location — override via config['journal_db']
_DEFAULT_DB_PATH: str = "journal.db"

# ── DDL ────────────────────────────────────────────────────────────────────

_DDL_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    symbol        TEXT    NOT NULL,
    side          TEXT    NOT NULL CHECK(side IN ('buy', 'sell')),
    price         REAL    NOT NULL,
    qty           REAL    NOT NULL,
    fee           REAL    NOT NULL DEFAULT 0.0,
    pnl           REAL,               -- NULL for buy rows
    exit_reason   TEXT,               -- 'signal' | 'stop_loss' | 'take_profit' | 'risk_halt'
    sl            REAL,               -- stop-loss price at entry (buy rows only)
    tp            REAL,               -- take-profit price at entry (buy rows only)
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_PRICE_HISTORY = """
CREATE TABLE IF NOT EXISTS price_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL UNIQUE,
    close     REAL    NOT NULL
);
"""

_DDL_EQUITY_HISTORY = """
CREATE TABLE IF NOT EXISTS equity_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL UNIQUE,
    equity    REAL    NOT NULL
);
"""


class TradeJournal:
    """
    SQLite-backed trade journal.

    Holds a single persistent connection per instance so repeated
    writes within a session are cheap.  The dashboard reads directly
    from the same file — no in-memory state drift is possible.

    Args:
        db_path: Path to the SQLite file.  Created on first use.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path: Path = Path(db_path or _DEFAULT_DB_PATH)
        self._conn: sqlite3.Connection = self._open_connection()
        self._create_schema()
        logger.info("TradeJournal initialised at %s", self._db_path)

    # ── Connection management ────────────────────────────────────────────

    def _open_connection(self) -> sqlite3.Connection:
        """Open and configure the SQLite connection."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")   # safe concurrent reads
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
            logger.info("TradeJournal connection closed.")
        except sqlite3.Error as exc:
            logger.error("Error closing journal connection: %s", exc)

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that commits on success or rolls back on error."""
        try:
            yield self._conn
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            logger.error("Journal DB error (rolled back): %s", exc)
            raise

    # ── Schema ───────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        """Create tables if they don't already exist."""
        with self._transaction() as conn:
            conn.execute(_DDL_TRADES)
            conn.execute(_DDL_PRICE_HISTORY)
            conn.execute(_DDL_EQUITY_HISTORY)
        logger.debug("Journal schema verified.")

    # ── Write API ────────────────────────────────────────────────────────

    def log_trade(self, trade: Dict[str, Any], symbol: str = "BTC/USDT") -> None:
        """Persist a single trade record.

        Args:
            trade: Dict with keys: timestamp, side, price, qty, fee,
                   and optionally pnl, exit_reason, sl, tp.
            symbol: Trading pair symbol.
        """
        sql = """
        INSERT INTO trades
            (timestamp, symbol, side, price, qty, fee, pnl, exit_reason, sl, tp)
        VALUES
            (:timestamp, :symbol, :side, :price, :qty, :fee, :pnl, :exit_reason, :sl, :tp)
        """
        row = {
            "timestamp":   str(trade.get("timestamp", datetime.utcnow().isoformat())),
            "symbol":      symbol,
            "side":        trade["side"],
            "price":       float(trade["price"]),
            "qty":         float(trade["qty"]),
            "fee":         float(trade.get("fee", 0.0)),
            "pnl":         float(trade["pnl"]) if trade.get("pnl") is not None else None,
            "exit_reason": trade.get("exit_reason"),
            "sl":          float(trade["sl"]) if trade.get("sl") is not None else None,
            "tp":          float(trade["tp"]) if trade.get("tp") is not None else None,
        }
        with self._transaction() as conn:
            conn.execute(sql, row)
        logger.debug("Trade logged: %s %s @ %.2f", row["side"], symbol, row["price"])

    def log_price(self, timestamp: str, close: float) -> None:
        """Upsert a price tick into price_history.

        Args:
            timestamp: ISO-format timestamp string.
            close: Closing price for the bar.
        """
        sql = """
        INSERT INTO price_history (timestamp, close) VALUES (?, ?)
        ON CONFLICT(timestamp) DO UPDATE SET close=excluded.close
        """
        with self._transaction() as conn:
            conn.execute(sql, (timestamp, float(close)))

    def log_equity(self, timestamp: str, equity: float) -> None:
        """Upsert an equity snapshot into equity_history.

        Args:
            timestamp: ISO-format timestamp string.
            equity: Total account equity (cash + open position value).
        """
        sql = """
        INSERT INTO equity_history (timestamp, equity) VALUES (?, ?)
        ON CONFLICT(timestamp) DO UPDATE SET equity=excluded.equity
        """
        with self._transaction() as conn:
            conn.execute(sql, (timestamp, float(equity)))

    def log_trades_batch(
        self, trades: List[Dict[str, Any]], symbol: str = "BTC/USDT"
    ) -> None:
        """Bulk-insert a list of trade dicts (e.g. after a backtest).

        Args:
            trades: List of trade dicts as returned by BacktestEngine.
            symbol: Trading pair symbol.
        """
        for trade in trades:
            self.log_trade(trade, symbol=symbol)
        logger.info("Batch-logged %d trades to journal.", len(trades))

    # ── Read API ─────────────────────────────────────────────────────────

    def get_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all trades, optionally filtered by symbol.

        Args:
            symbol: If given, only trades for this symbol are returned.

        Returns:
            List of dicts representing each trade row.
        """
        if symbol:
            sql = "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp"
            cursor = self._conn.execute(sql, (symbol,))
        else:
            sql = "SELECT * FROM trades ORDER BY timestamp"
            cursor = self._conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
