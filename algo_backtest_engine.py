import sys
import json
import random
import time
import logging
from datetime import datetime
from collections import deque

import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import websocket

# --- Global Configuration ---
SETTINGS = {
    "SIMULATE": True,
    "WS_URL": "wss://stream.binance.com:9443/ws/btcusdt@trade",
    "CANDLE_INTERVAL": 5,
    "MAX_CANDLES": 120,
    "LIVE_BUFFER": 300,
    "INITIAL_CAPITAL": 10000.0,
    "SMA_SHORT": 3,
    "SMA_LONG": 5,
    "TRADING_FEE": 0.001
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class BinanceDataStream(QtCore.QThread):
    """Handles the heavy lifting of data ingestion."""
    price_received = QtCore.pyqtSignal(float, object)
    status_updated = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._active = True

    def run(self):
        if SETTINGS["SIMULATE"]:
            self.status_updated.emit("SIMULATION: Active")
            current_price = 65000.0
            while self._active:
                current_price += random.uniform(-30, 30)
                self.price_received.emit(float(current_price), datetime.now())
                time.sleep(0.05)
            return

        self.status_updated.emit("Connecting to Binance...")
        self._start_websocket()

    def _start_websocket(self):
        def on_msg(ws, message):
            try:
                data = json.loads(message)
                price = float(data['p'])
                ts = datetime.fromtimestamp(data['T'] / 1000.0)
                self.price_received.emit(price, ts)
            except Exception as e:
                self.status_updated.emit(f"Parse Error: {e}")

        ws = websocket.WebSocketApp(
            SETTINGS["WS_URL"],
            on_open=lambda w: self.status_updated.emit("WS connected"),
            on_message=on_msg,
            on_error=lambda w, e: self.status_updated.emit(f"WS Error: {e}"),
            on_close=lambda w, c, m: self.status_updated.emit("WS Closed")
        )
        ws.run_forever(ping_interval=20)

    def stop(self):
        self._active = False
        self.wait(1000)

class CryptoDashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Market Terminal - BTC/USDT")
        self.resize(1400, 820)

        # State management
        self.candles = deque(maxlen=SETTINGS["MAX_CANDLES"])
        self.active_candle = None
        self.last_ts = None
        
        self.tick_prices = deque(maxlen=SETTINGS["LIVE_BUFFER"])
        self.tick_times = deque(maxlen=SETTINGS["LIVE_BUFFER"])

        # Portfolio logic (Keeping your exact logic)
        self.capital = SETTINGS["INITIAL_CAPITAL"]
        self.position = 0.0
        self.equity_history = []
        self.trade_markers = []

        self._init_ui()

        # Threads and Timers
        self.data_thread = BinanceDataStream()
        self.data_thread.price_received.connect(self.handle_new_tick)
        self.data_thread.status_updated.connect(lambda s: self.status_label.setText(f"Status: {s}"))
        self.data_thread.start()

        self.refresh_timer = QtCore.QTimer()
        self.refresh_timer.timeout.connect(self.update_plots)
        self.refresh_timer.start(200)

    def _init_ui(self):
        """Setup the layout and pyqtgraph widgets."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QGridLayout(central_widget)

        pg.setConfigOptions(antialias=True)
        self.chart = pg.PlotWidget(title="Live Price Action")
        self.equity_chart = pg.PlotWidget(title="Equity Growth")
        
        self.status_label = QtWidgets.QLabel("Status: Initializing...")
        self.info_panel = QtWidgets.QLabel("Portfolio: -")
        
        # Style the labels
        for label in [self.status_label, self.info_panel]:
            label.setStyleSheet("color: #ccc; background: #1e1e1e; padding: 8px; font-family: monospace;")

        main_layout.addWidget(self.chart, 0, 0, 2, 4)
        main_layout.addWidget(self.equity_chart, 2, 0, 1, 4)
        main_layout.addWidget(self.status_label, 0, 4)
        main_layout.addWidget(self.info_panel, 1, 4)

        self.chart.showGrid(x=True, y=True)
        self.equity_chart.showGrid(x=True, y=True)

    def handle_new_tick(self, price, ts):
        self.tick_prices.append(price)
        self.tick_times.append(ts)

        if self.active_candle is None:
            self.active_candle = {'open': price, 'high': price, 'low': price, 'close': price, 'time': ts}
            self.last_ts = ts
            return

        # Update current candle
        self.active_candle['high'] = max(self.active_candle['high'], price)
        self.active_candle['low'] = min(self.active_candle['low'], price)
        self.active_candle['close'] = price

        # Close candle on interval
        if (ts - self.last_ts).total_seconds() >= SETTINGS["CANDLE_INTERVAL"]:
            final_candle = self.active_candle.copy()
            self.candles.append(final_candle)
            
            # Execute Strategy (Preserved Logic)
            self._run_strategy(final_candle['close'])
            
            self.active_candle = {'open': price, 'high': price, 'low': price, 'close': price, 'time': ts}
            self.last_ts = ts

    def _run_strategy(self, last_price):
        try:
            df = pd.DataFrame(list(self.candles))
            if len(df) >= SETTINGS["SMA_LONG"]:
                fast_sma = df['close'].rolling(SETTINGS["SMA_SHORT"]).mean().iloc[-1]
                slow_sma = df['close'].rolling(SETTINGS["SMA_LONG"]).mean().iloc[-1]
                
                # Signal logic
                if fast_sma > slow_sma and self.position == 0:
                    qty = (self.capital * (1 - SETTINGS["TRADING_FEE"])) / last_price
                    self.position, self.capital = qty, 0.0
                    self.trade_markers.append(('buy', len(df)-1, last_price))
                elif fast_sma < slow_sma and self.position > 0:
                    self.capital = self.position * last_price * (1 - SETTINGS["TRADING_FEE"])
                    self.position = 0.0
                    self.trade_markers.append(('sell', len(df)-1, last_price))
            
            self.equity_history.append(self.capital + self.position * last_price)
        except Exception as e:
            logging.error(f"Strategy Error: {e}")

    def update_plots(self):
        try:
            self.chart.clear()
            df = pd.DataFrame(list(self.candles))
            count = len(df)

            # Draw Historical Candles
            for i, row in df.iterrows():
                idx = int(i)
                color = (38, 166, 154) if row['close'] >= row['open'] else (239, 83, 80)
                # Wick
                self.chart.plot([idx, idx], [row['low'], row['high']], pen=pg.mkPen(200, 200, 200, 150))
                # Body
                rect = QtWidgets.QGraphicsRectItem(idx - 0.3, min(row['open'], row['close']), 0.6, max(abs(row['close'] - row['open']), 1e-6))
                rect.setBrush(pg.mkBrush(color))
                rect.setPen(pg.mkPen(color))
                self.chart.addItem(rect)

            # Draw Active Candle
            if self.active_candle:
                x_pos = count if count > 0 else 0
                ac = self.active_candle
                self.chart.plot([x_pos, x_pos], [ac['low'], ac['high']], pen=pg.mkPen('y'))
                rect = QtWidgets.QGraphicsRectItem(x_pos - 0.25, min(ac['open'], ac['close']), 0.5, max(abs(ac['close'] - ac['open']), 1e-6))
                rect.setBrush(pg.mkBrush(200, 200, 0))
                self.chart.addItem(rect)

            # Buy/Sell Markers
            for kind, idx, val in self.trade_markers:
                color = (0, 255, 0) if kind == 'buy' else (255, 0, 0)
                scatter = pg.ScatterPlotItem([idx], [val], brush=pg.mkBrush(color), size=12, symbol='t' if kind=='sell' else 't1')
                self.chart.addItem(scatter)

            # Sidebar Info Update
            cur_price = self.tick_prices[-1] if self.tick_prices else 0
            equity = self.capital + (self.position * cur_price)
            self.info_panel.setText(
                f"<b>Portfolio Details</b><br>"
                f"Price: {cur_price:.2f}<br>"
                f"Pos: {self.position:.5f}<br>"
                f"Cash: {self.capital:.2f}<br>"
                f"Equity: {equity:.2f}"
            )

            # Equity Curve
            if self.equity_history:
                self.equity_chart.plot(self.equity_history, pen=pg.mkPen('#00BFFF', width=2), clear=True)

        except Exception as e:
            self.status_label.setText(f"UI Error: {e}")

    def closeEvent(self, event):
        self.data_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CryptoDashboard()
    window.show()
    sys.exit(app.exec_())
