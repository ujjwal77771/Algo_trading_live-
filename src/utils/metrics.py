"""
Performance metrics module.
Calculates Sharpe, drawdown, win-rate, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Calculates drawdown series from equity curve."""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (peak - equity_curve) / peak
    return drawdown

def calculate_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates key performance metrics.
    
    Args:
        equity_curve: Series of account equity over time.
        trades: DataFrame containing trade results (must have 'pnl' column).
    """
    metrics = {}
    
    # Equity metrics
    if not equity_curve.empty and len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        metrics['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (365 / len(returns)) - 1 if len(returns) > 0 else 0
        
        volatility = returns.std() * np.sqrt(365) # Assuming daily returns, adjust if needed
        metrics['sharpe_ratio'] = metrics['annualized_return'] / volatility if volatility > 0 else 0
        
        drawdown = calculate_drawdown(equity_curve)
        metrics['max_drawdown'] = drawdown.max()
    else:
        metrics['total_return'] = 0.0
        metrics['annualized_return'] = 0.0
        metrics['sharpe_ratio'] = 0.0
        metrics['max_drawdown'] = 0.0

    # Trade metrics
    if not trades.empty and 'pnl' in trades.columns:
        winning_trades = trades[trades['pnl'] > 0]
        metrics['total_trades'] = len(trades)
        metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        metrics['avg_win'] = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        losing_trades = trades[trades['pnl'] <= 0]
        metrics['avg_loss'] = losing_trades['pnl'].mean() if not losing_trades.empty else 0
    else:
        metrics['total_trades'] = 0
        metrics['win_rate'] = 0.0
        metrics['avg_win'] = 0.0
        metrics['avg_loss'] = 0.0
        
    return metrics
