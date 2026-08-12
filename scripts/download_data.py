"""
scripts/download_data.py
========================
Downloads historical OHLCV data from CCXT exchange and saves to historical_data.csv.

Usage:
------
    python scripts/download_data.py --symbol ETH/USDT --timeframe 1h --limit 1000
"""

import argparse
import ccxt
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Download OHLCV data from CCXT")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair symbol (e.g. BTC/USDT, ETH/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Timeframe (e.g. 1m, 5m, 1h, 1d)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of bars to download")
    parser.add_argument("--output", default="historical_data.csv", help="Output CSV path")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Connecting to Binance via CCXT to fetch {args.limit} bars of {args.symbol} ({args.timeframe})...")
    
    # Initialize exchange
    exchange = ccxt.binance({"enableRateLimit": True})
    
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(args.symbol, args.timeframe, limit=args.limit)
        
        # Parse to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Convert timestamp to human-readable format
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Save to CSV
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"Successfully downloaded {len(df)} bars.")
        print(f"Saved to: {output_path.resolve()}")
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    main()
