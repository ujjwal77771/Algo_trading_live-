import websocket, json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

# --------------------------
# Parameters
# --------------------------
INITIAL_CAPITAL = 10000
FEE = 0.001
SHORT_SMA = 3
LONG_SMA = 5
CANDLE_INTERVAL = 60  # seconds
MAX_CANDLES = 50      # number of candles displayed

# --------------------------
# State
# --------------------------
capital = INITIAL_CAPITAL
position = 0
equity_curve = []
ohlc_data = deque(maxlen=MAX_CANDLES)
trades_buffer = []

# --------------------------
# Setup matplotlib
# --------------------------
plt.ion()  # interactive mode

fig, (ax_candle, ax_equity) = plt.subplots(2, 1, figsize=(12,8), gridspec_kw={'height_ratios':[3,1]})
fig.suptitle("Live BTC/USDT Trading Dashboard", color='lime')

def plot_dashboard(df, equity_curve):
    ax_candle.clear()
    ax_equity.clear()

    # Plot candles
    for i, row in df.iterrows():
        color = 'lime' if row['Close'] >= row['Open'] else 'red'
        ax_candle.plot([i,i],[row['Low'],row['High']], color=color)
        ax_candle.add_patch(plt.Rectangle((i-0.2,row['Open']),0.4,row['Close']-row['Open'], color=color))
    ax_candle.set_title("BTC/USDT Price Chart", color='lime')
    ax_candle.set_ylabel("Price", color='lime')
    ax_candle.tick_params(colors='lime')

    # Plot equity
    ax_equity.plot(range(len(equity_curve)), equity_curve, color='green')
    ax_equity.set_title("Equity Curve", color='lime')
    ax_equity.set_ylabel("Equity", color='lime')
    ax_equity.tick_params(colors='lime')

    plt.tight_layout()
    plt.pause(0.01)

# --------------------------
# Helper functions
# --------------------------
def create_candle(trades):
    if not trades:
        return None
    prices = [t['p'] for t in trades]
    return {
        'time': trades[-1]['time'],
        'Open': prices[0],
        'High': max(prices),
        'Low': min(prices),
        'Close': prices[-1],
        'Volume': sum(t['q'] for t in trades)
    }

def compute_sma(df, window):
    return df['Close'].rolling(window).mean()

# --------------------------
# WebSocket callback
# --------------------------
def on_message(ws, message):
    global trades_buffer, ohlc_data, capital, position, equity_curve

    msg = json.loads(message)
    trade = {
        'p': float(msg['p']),
        'q': float(msg['q']),
        'time': datetime.fromtimestamp(msg['T']/1000)
    }
    trades_buffer.append(trade)

    # Aggregate into candle every CANDLE_INTERVAL
    if len(trades_buffer) > 0:
        first_time = trades_buffer[0]['time']
        last_time = trades_buffer[-1]['time']
        if (last_time - first_time).total_seconds() >= CANDLE_INTERVAL:
            candle = create_candle(trades_buffer)
            trades_buffer.clear()
            if candle:
                ohlc_data.append(candle)

            # Prepare DataFrame
            df = pd.DataFrame(list(ohlc_data)).set_index('time')

            if len(df) >= LONG_SMA:
                df['SMA_short'] = compute_sma(df, SHORT_SMA)
                df['SMA_long'] = compute_sma(df, LONG_SMA)

                # Generate signal
                signal = 0
                if df['SMA_short'].iloc[-1] > df['SMA_long'].iloc[-1] and position == 0:
                    signal = 1
                elif df['SMA_short'].iloc[-1] < df['SMA_long'].iloc[-1] and position > 0:
                    signal = -1

                price = df['Close'].iloc[-1]

                # Execute trade
                if signal == 1 and position == 0:
                    qty = (capital * (1 - FEE)) / price
                    position = qty
                    capital = 0
                    print(f"{datetime.now()} BUY {qty:.6f} at {price:.2f}")

                elif signal == -1 and position > 0:
                    capital = position * price * (1 - FEE)
                    print(f"{datetime.now()} SELL {position:.6f} at {price:.2f}")
                    position = 0

                total_equity = capital + position * price
                equity_curve.append(total_equity)
                print(f"Price: {price:.2f} | Equity: {total_equity:.2f} | Position: {position:.6f} | Cash: {capital:.2f}")

            # Update chart
            df_plot = df[['Open','High','Low','Close']]
            plot_dashboard(df_plot, equity_curve)
-
def on_open(ws):
    print("WebSocket opened.")

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed.")

url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
ws = websocket.WebSocketApp(url,
                            on_message=on_message,
                            on_open=on_open,
                            on_error=on_error,
                            on_close=on_close)
ws.run_forever()
