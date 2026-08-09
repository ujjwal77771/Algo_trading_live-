"""
scripts/run_backtest.py
=======================
End-to-end backtest runner.

Usage
-----
    python scripts/run_backtest.py [--config config/settings.yaml]
                                   [--data   historical_data.csv]
                                   [--mode   rule|ensemble]

Outputs
-------
  - Prints a metrics table (Sharpe, max-DD, win-rate, total trades)
  - Saves equity curve to journal.db (price_history + equity_history)
  - Saves all trades to journal.db (trades table)
  - Writes results/backtest_metrics.json for notebook consumption
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.execution.paper_broker import PaperBroker
from src.risk.risk_manager import RiskManager
from src.signals.rule_based import RuleBasedSignal
from src.state.journal import TradeJournal
from src.utils.config_loader import load_config
from src.utils.logger import logger
from src.utils.metrics import calculate_metrics


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical backtest")
    parser.add_argument(
        "--config", default="config/settings.yaml", help="Path to settings.yaml"
    )
    parser.add_argument(
        "--data", default="historical_data.csv", help="Path to OHLCV CSV"
    )
    parser.add_argument(
        "--mode",
        choices=["rule", "ensemble"],
        default="rule",
        help="Signal mode: 'rule' (Step 2 baseline) or 'ensemble' (Steps 3+)",
    )
    return parser.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────

def _load_data(path: str) -> pd.DataFrame:
    """Load OHLCV CSV.  Expects columns: timestamp, open, high, low, close, volume."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.sort_index()
    logger.info("Loaded %d bars from %s", len(df), data_path)
    return df


# ── Buy-and-hold baseline ──────────────────────────────────────────────────

def _buy_and_hold_return(df: pd.DataFrame, initial_capital: float) -> dict:
    """Compute simple buy-and-hold metrics for comparison."""
    start_price = float(df["close"].iloc[0])
    end_price   = float(df["close"].iloc[-1])
    total_return = (end_price / start_price) - 1
    return {
        "strategy":      "Buy-and-Hold",
        "total_return":  round(total_return, 4),
        "sharpe_ratio":  None,
        "max_drawdown":  None,
        "total_trades":  1,
        "win_rate":      None,
    }


# ── Signal factory ─────────────────────────────────────────────────────────

def _build_signal(mode: str, config: dict):
    if mode == "rule":
        return RuleBasedSignal(config)

    # ensemble mode: requires a trained model — load if available
    try:
        import lightgbm as lgb
        from src.ml.train import ModelTrainer
        from src.ml.regime import RegimeDetector
        from src.signals.ml_signal import MLSignal
        from src.signals.ensemble import EnsembleSignal

        trainer = ModelTrainer(config)
        model_path = Path(config["ml"]["model_dir"]) / "lgb_model.txt"

        if not model_path.exists():
            logger.warning(
                "No trained model found at %s. Falling back to rule-based signal.",
                model_path,
            )
            return RuleBasedSignal(config)

        model = trainer.load_model()
        rule   = RuleBasedSignal(config)
        ml     = MLSignal(config, model)
        regime = RegimeDetector(
            adx_threshold=float(config.get("adx_threshold", 25.0)),
            vol_lookback=int(config.get("vol_lookback", 20)),
        )
        return EnsembleSignal(config, rule, ml, regime)

    except ImportError as exc:
        logger.warning("Ensemble dependencies missing (%s). Using rule-based.", exc)
        return RuleBasedSignal(config)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    df = _load_data(args.data)

    initial_capital: float = float(config["initial_capital"])
    trading_fee:     float = float(config["trading_fee"])
    timeframe:       str   = config["timeframe"]

    broker       = PaperBroker(initial_capital, fee_rate=trading_fee)
    risk_manager = RiskManager(config)
    signal       = _build_signal(args.mode, config)
    journal      = TradeJournal()

    engine = BacktestEngine(
        data=df,
        broker=broker,
        risk_manager=risk_manager,
        signal_generator=signal,
        trading_fee=trading_fee,
    )

    # ── Run backtest ──────────────────────────────────────────────────────
    logger.info("Running %s backtest on %d bars…", args.mode, len(df))
    equity_curve_df = engine.run()
    trades_df       = pd.DataFrame(engine.trades)

    # ── Compute metrics ───────────────────────────────────────────────────
    equity_series = equity_curve_df["equity"]
    metrics = calculate_metrics(equity_series, trades_df, timeframe=timeframe)
    metrics["strategy"] = f"{args.mode.capitalize()} Signal"

    bah = _buy_and_hold_return(df, initial_capital)

    # ── Persist to journal ────────────────────────────────────────────────
    for row in equity_curve_df.reset_index().itertuples():
        journal.log_price(str(row.timestamp), float(df.loc[row.timestamp, "close"])
                          if row.timestamp in df.index else 0.0)
        journal.log_equity(str(row.timestamp), float(row.equity))

    journal.log_trades_batch(engine.trades)
    journal.close()

    # ── Save metrics JSON for notebook ───────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "backtest_metrics.json"
    combined = {"backtest": metrics, "buy_and_hold": bah}
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'BACKTEST RESULTS':^60}")
    print("=" * 60)
    rows = [
        ("Strategy",        metrics["strategy"]),
        ("Total Return",    f"{metrics['total_return']:.2%}"),
        ("Ann. Return",     f"{metrics['annualized_return']:.2%}"),
        ("Sharpe Ratio",    f"{metrics['sharpe_ratio']:.3f}"),
        ("Max Drawdown",    f"{metrics['max_drawdown']:.2%}"),
        ("Total Trades",    str(metrics["total_trades"])),
        ("Win Rate",        f"{metrics['win_rate']:.2%}"),
        ("Avg Win",         f"${metrics['avg_win']:.2f}"),
        ("Avg Loss",        f"${metrics['avg_loss']:.2f}"),
        (""),
        ("Buy-and-Hold",    f"{bah['total_return']:.2%}"),
    ]
    for row in rows:
        if len(row) == 1:
            print("-" * 60)
        else:
            print(f"  {row[0]:<22} {row[1]:>34}")
    print("=" * 60)
    print(f"\nMetrics saved → {output_path}")
    logger.info("Backtest complete. Results saved to %s", output_path)


if __name__ == "__main__":
    main()
