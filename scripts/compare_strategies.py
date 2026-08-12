"""
scripts/compare_strategies.py
==============================
Runs ALL available strategies on the same dataset and ranks them by Sharpe ratio.

Usage:
------
    python scripts/compare_strategies.py --data historical_data.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.execution.paper_broker import PaperBroker
from src.risk.risk_manager import RiskManager
from src.utils.config_loader import load_config
from src.utils.metrics import calculate_metrics
from src.utils.logger import logger

# Import all strategy classes
from src.signals.rule_based import RuleBasedSignal
from src.signals.macd_signal import MACDSignal
from src.signals.bollinger_signal import BollingerSignal
from src.signals.rsi_momentum_signal import RSIMomentumSignal
from src.signals.triple_ema_signal import TripleEMASignal
from src.signals.breakout_signal import BreakoutSignal


# ── All available strategies ──────────────────────────────────────────────

STRATEGIES = {
    "EMA Cross + RSI":    RuleBasedSignal,
    "MACD Crossover":     MACDSignal,
    "Bollinger Reversion": BollingerSignal,
    "RSI Momentum":       RSIMomentumSignal,
    "Triple EMA Trend":   TripleEMASignal,
    "Breakout (High/Low)": BreakoutSignal,
}


def _load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    date_candidates = ["timestamp", "date", "datetime", "time"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"No date column found. Columns: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "timestamp"
    return df


def _buy_and_hold_return(df: pd.DataFrame) -> float:
    return (float(df["close"].iloc[-1]) / float(df["close"].iloc[0])) - 1


def main():
    parser = argparse.ArgumentParser(description="Compare all strategies")
    parser.add_argument("--data", default="historical_data.csv")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    df = _load_data(args.data)
    initial_capital = float(config["initial_capital"])
    trading_fee = float(config["trading_fee"])
    timeframe = config["timeframe"]

    print(f"\nDataset: {args.data} ({len(df)} bars)")
    print(f"Capital: ${initial_capital:,.0f}  |  Fee: {trading_fee:.3%}  |  Timeframe: {timeframe}")
    print("=" * 90)

    results = []

    for name, signal_cls in STRATEGIES.items():
        try:
            # Fresh broker + risk manager for each strategy
            broker = PaperBroker(initial_capital, fee_rate=trading_fee)
            risk_mgr = RiskManager(config)
            signal = signal_cls(config)

            engine = BacktestEngine(
                data=df,
                broker=broker,
                risk_manager=risk_mgr,
                signal_generator=signal,
                trading_fee=trading_fee,
            )

            equity_df = engine.run()
            trades_df = pd.DataFrame(engine.trades)

            if equity_df.empty:
                results.append({
                    "strategy": name,
                    "total_return": 0.0,
                    "sharpe": 0.0,
                    "max_dd": 0.0,
                    "trades": 0,
                    "win_rate": 0.0,
                })
                continue

            metrics = calculate_metrics(equity_df["equity"], trades_df, timeframe=timeframe)
            results.append({
                "strategy": name,
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe_ratio"],
                "max_dd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
            })

        except Exception as exc:
            logger.error("Strategy '%s' failed: %s", name, exc)
            results.append({
                "strategy": name,
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_dd": 0.0,
                "trades": 0,
                "win_rate": 0.0,
            })

    # Add buy-and-hold baseline
    bah = _buy_and_hold_return(df)
    results.append({
        "strategy": "Buy & Hold (baseline)",
        "total_return": bah,
        "sharpe": 0.0,
        "max_dd": 0.0,
        "trades": 1,
        "win_rate": 0.0,
    })

    # Sort by Sharpe ratio (best first)
    results.sort(key=lambda r: r["sharpe"], reverse=True)

    # Print ranked results table
    print(f"\n{'Rank':<6} {'Strategy':<24} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'Trades':>8} {'Win Rate':>10}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        marker = " <-- BEST" if i == 1 and r["strategy"] != "Buy & Hold (baseline)" else ""
        print(
            f"{i:<6} {r['strategy']:<24} {r['total_return']:>9.2%} {r['sharpe']:>10.3f} "
            f"{r['max_dd']:>9.2%} {r['trades']:>8} {r['win_rate']:>9.2%}{marker}"
        )
    print("=" * 90)

    # Winner announcement
    best = results[0]
    if best["strategy"] != "Buy & Hold (baseline)":
        print(f"\n>>> WINNER: {best['strategy']}  (Sharpe: {best['sharpe']:.3f}, Return: {best['total_return']:.2%})")
    else:
        print("\n>>> No strategy beat buy-and-hold. Consider tuning parameters or using the ML ensemble.")

    print(f"\nBuy-and-Hold baseline return: {bah:.2%}")
    print("\nTo run the best strategy on its own:")
    print(f'  python scripts/run_backtest.py --data {args.data} --mode rule')
    print("\nTo run the ML ensemble (requires training first):")
    print(f'  python scripts/run_backtest.py --data {args.data} --mode ensemble')


if __name__ == "__main__":
    main()
