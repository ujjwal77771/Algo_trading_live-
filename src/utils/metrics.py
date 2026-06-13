"""
Performance metrics module.
Calculates Sharpe, drawdown, win-rate, etc.

Annualization factor is derived from a configurable timeframe string,
not hardcoded to 365.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

# Maps timeframe strings (matching exchange conventions) to periods per year.
# Crypto markets trade 365 days/year, equities ~252 — caller chooses
# the right base via the timeframe string they pass in.
TIMEFRAME_TO_ANNUAL_PERIODS: Dict[str, float] = {
    '1m':   365 * 24 * 60,
    '5m':   365 * 24 * 12,
    '15m':  365 * 24 * 4,
    '30m':  365 * 24 * 2,
    '1h':   365 * 24,
    '4h':   365 * 6,
    '1d':   365,
    '1w':   52,
}


def resolve_periods_per_year(timeframe: str) -> float:
    """
    Converts a human-readable timeframe string to the number of
    periods per year (crypto calendar — 365 days).

    Raises:
        ValueError: If timeframe is not recognized.
    """
    tf = timeframe.strip().lower()
    if tf not in TIMEFRAME_TO_ANNUAL_PERIODS:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Supported: {list(TIMEFRAME_TO_ANNUAL_PERIODS.keys())}"
        )
    return TIMEFRAME_TO_ANNUAL_PERIODS[tf]


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Calculates drawdown series from equity curve."""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (peak - equity_curve) / peak
    return drawdown


def calculate_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    timeframe: str = '1d',
) -> Dict[str, Any]:
    """
    Calculates key performance metrics.

    Args:
        equity_curve: Series of account equity over time (one value per bar).
        trades: DataFrame containing trade results (must have 'pnl' column
                for sell rows).
        timeframe: Bar timeframe string (e.g. '1h', '1d'). Used to
                   annualize Sharpe and returns correctly.
    """
    periods_per_year = resolve_periods_per_year(timeframe)
    metrics: Dict[str, Any] = {}

    # Equity metrics
    if not equity_curve.empty and len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        metrics['total_return'] = (
            (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        )

        n_periods = len(returns)
        metrics['annualized_return'] = (
            (1 + metrics['total_return']) ** (periods_per_year / n_periods) - 1
            if n_periods > 0
            else 0.0
        )

        annualized_vol = returns.std() * np.sqrt(periods_per_year)
        metrics['sharpe_ratio'] = (
            metrics['annualized_return'] / annualized_vol
            if annualized_vol > 0
            else 0.0
        )

        drawdown = calculate_drawdown(equity_curve)
        metrics['max_drawdown'] = float(drawdown.max())
    else:
        metrics['total_return'] = 0.0
        metrics['annualized_return'] = 0.0
        metrics['sharpe_ratio'] = 0.0
        metrics['max_drawdown'] = 0.0

    # Trade metrics
    if not trades.empty and 'pnl' in trades.columns:
        sell_trades = trades.dropna(subset=['pnl'])
        winning_trades = sell_trades[sell_trades['pnl'] > 0]
        losing_trades = sell_trades[sell_trades['pnl'] <= 0]

        metrics['total_trades'] = len(sell_trades)
        metrics['win_rate'] = (
            len(winning_trades) / len(sell_trades)
            if len(sell_trades) > 0
            else 0.0
        )
        metrics['avg_win'] = (
            float(winning_trades['pnl'].mean())
            if not winning_trades.empty
            else 0.0
        )
        metrics['avg_loss'] = (
            float(losing_trades['pnl'].mean())
            if not losing_trades.empty
            else 0.0
        )
    else:
        metrics['total_trades'] = 0
        metrics['win_rate'] = 0.0
        metrics['avg_win'] = 0.0
        metrics['avg_loss'] = 0.0

    return metrics
