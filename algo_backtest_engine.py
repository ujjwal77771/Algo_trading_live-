# live_dashboard_fixed_rectitem.py
# Requires: pip install pyqt5 pyqtgraph websocket-client pandas
import sys, json, random, time
from datetime import datetime
from collections import deque

import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import websocket

# ---------- CONFIG ----------
SIMULATE = True           # set False to use live Binance stream
WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
CANDLE_INTERVAL = 5      # seconds per candle (set 60 for 1-minute candles)
MAX_CANDLES = 120
LIVE_BUFFER = 300
INITIAL_CAPITAL = 10000.0
SHORT_SMA = 3
LONG_SMA = 5
FEE = 0.001

# ---------- WebSocket thread ----------
class WebSocketThread(QtCore.QThread):
    trade_signal = QtCore.pyqtSignal(float, object)   # price, timestamp
    status_signal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def run(self):
        if SIMULATE:
            self.status_signal.emit("SIMULATING: generating trades")
            price = 65000.0
            while self._running:
                price += random.uniform(-30, 30)
                ts = datetime.now()
                self.trade_signal.emit(float(price), ts)
                time.sleep(0.05)   # many trades per second
            return

        self.status_signal.emit("Connecting to WS...")
        try:
            def on_open(ws):
                self.status_signal.emit("WS: OPEN")

            def on_close(ws, code, msg):
                self.status_signal.emit(f"WS: CLOSED ({code})")

            def on_error(ws, err):
                self.status_signal.emit(f"WS ERROR: {err}")

            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    price = float(data['p'])
                    ts = datetime.fromtimestamp(data['T'] / 1000.0)
                    self.trade_signal.emit(price, ts)
                except Exception as e:
                    self.status_signal.emit(f"parse err: {e}")

            ws = websocket.WebSocketApp(WS_URL,
                                        on_open=on_open,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            self.status_signal.emit(f"WS thread exception: {e}")

    def stop(self):
        self._running = False
        self.wait(1000)

# ---------- Dashboard GUI ----------
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LIVE BTC Dashboard (fixed QtWidgets.QGraphicsRectItem)")
        self.resize(1400, 820)

        # state
        self.ohlc = deque(maxlen=MAX_CANDLES)
        self.current_candle = None
        self.last_candle_time = None

        self.live_prices = deque(maxlen=LIVE_BUFFER)
        self.live_times = deque(maxlen=LIVE_BUFFER)

        self.capital = INITIAL_CAPITAL
        self.position = 0.0
        self.equity_curve = []
        self.markers = []

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(title="Candles + Live price")
        self.eq_plot = pg.PlotWidget(title="Equity Curve")
        layout.addWidget(self.plot, 0, 0, 2, 4)
        layout.addWidget(self.eq_plot, 2, 0, 1, 4)

        self.status_label = QtWidgets.QLabel("Status: init")
        self.status_label.setStyleSheet("color:white; background:black; padding:6px;")
        self.info_label = QtWidgets.QLabel("Info: -")
        self.info_label.setStyleSheet("color:white; background:black; padding:6px;")
        layout.addWidget(self.status_label, 0, 4)
        layout.addWidget(self.info_label, 1, 4)

        self.plot.showGrid(x=True, y=True)
        self.eq_plot.showGrid(x=True, y=True)

        # transient references (not strictly required because we clear plot each redraw)
        self.live_curve_item = None
        self.live_hline = None
        self.live_text = None

        # timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.redraw)
        self.timer.start(200)   # refresh rate

        # websocket thread
        self.ws = WebSocketThread()
        self.ws.trade_signal.connect(self.on_trade)
        self.ws.status_signal.connect(self.on_status)
        self.ws.start()

    def on_status(self, txt):
        self.status_label.setText("Status: " + str(txt))

    def on_trade(self, price: float, ts: datetime):
        # append live price
        try:
            self.live_prices.append(price)
            self.live_times.append(ts)
        except Exception:
            pass

        # initialize current candle if needed
        if self.current_candle is None:
            self.current_candle = {'open': price, 'high': price, 'low': price, 'close': price, 'time': ts}
            self.last_candle_time = ts
            return

        # update current candle OHLC
        self.current_candle['high'] = max(self.current_candle['high'], price)
        self.current_candle['low'] = min(self.current_candle['low'], price)
        self.current_candle['close'] = price

        # finalize candle when interval passes
        if (ts - self.last_candle_time).total_seconds() >= CANDLE_INTERVAL:
            finished = {
                'open': self.current_candle['open'],
                'high': self.current_candle['high'],
                'low': self.current_candle['low'],
                'close': self.current_candle['close'],
                'time': self.current_candle['time']
            }
            self.ohlc.append(finished)

            # append equity datapoint always (so equity curve is visible)
            last_price_for_eq = finished['close']
            self.equity_curve.append(self.capital + self.position * last_price_for_eq)

            # SMA logic (may update position)
            try:
                df = pd.DataFrame(list(self.ohlc))
                if len(df) >= LONG_SMA:
                    df['SMA_s'] = df['close'].rolling(SHORT_SMA).mean()
                    df['SMA_l'] = df['close'].rolling(LONG_SMA).mean()
                    s = df['SMA_s'].iloc[-1]
                    l = df['SMA_l'].iloc[-1]
                    price_now = df['close'].iloc[-1]
                    if s > l and self.position == 0:
                        qty = (self.capital * (1 - FEE)) / price_now
                        self.position = qty
                        self.capital = 0.0
                        self.markers.append(('buy', len(df)-1, price_now))
                    elif s < l and self.position > 0:
                        self.capital = self.position * price_now * (1 - FEE)
                        self.position = 0.0
                        self.markers.append(('sell', len(df)-1, price_now))
                    # update equity after trade
                    self.equity_curve[-1] = self.capital + self.position * price_now
            except Exception as e:
                self.on_status(f"SMA error: {e}")

            # reset for next candle
            self.current_candle = {'open': price, 'high': price, 'low': price, 'close': price, 'time': ts}
            self.last_candle_time = ts

    def redraw(self):
        # defensive: ensure plotting doesn't crash the timer loop
        try:
            self.plot.clear()

            df = pd.DataFrame(list(self.ohlc))
            n = len(df)

            # draw finalized candles
            if n > 0:
                for i, row in df.iterrows():
                    x = int(i)
                    # wick
                    self.plot.plot([x, x], [row['low'], row['high']], pen=pg.mkPen('w'))
                    # body using QtWidgets.QGraphicsRectItem (fix applied)
                    openp, closep = row['open'], row['close']
                    y = min(openp, closep)
                    height = max(abs(closep - openp), 1e-6)
                    color = (0, 200, 0) if closep >= openp else (200, 0, 0)
                    rect = QtWidgets.QGraphicsRectItem(x - 0.3, y, 0.6, height)
                    rect.setBrush(pg.mkBrush(color))
                    rect.setPen(pg.mkPen(color))
                    self.plot.addItem(rect)

            # draw current in-progress candle slightly right of last candle
            if self.current_candle is not None:
                base = n - 0.5 if n > 0 else -0.5
                x = base + 0.5
                cc = self.current_candle
                self.plot.plot([x, x], [cc['low'], cc['high']], pen=pg.mkPen('y'))
                rect = QtWidgets.QGraphicsRectItem(x - 0.25, min(cc['open'], cc['close']), 0.5, max(abs(cc['close'] - cc['open']), 1e-6))
                rect.setBrush(pg.mkBrush((200, 200, 0)))
                rect.setPen(pg.mkPen((200, 200, 0)))
                self.plot.addItem(rect)

            # draw markers
            for kind, xi, yy in self.markers:
                color = (0, 255, 0) if kind == 'buy' else (255, 0, 0)
                sp = pg.ScatterPlotItem([xi], [yy], brush=pg.mkBrush(color), size=10)
                self.plot.addItem(sp)

            # draw live per-trade line to the right of last candle (fractional offsets)
            lp = list(self.live_prices)
            if lp:
                base_index = n
                count = len(lp)
                offsets = [base_index + 0.02 + 0.96*(i / max(count-1,1)) for i in range(count)]
                self.plot.plot(offsets, lp, pen=pg.mkPen('y', width=2))
                last_live = lp[-1]
                hline = pg.InfiniteLine(pos=last_live, angle=0, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))
                self.plot.addItem(hline)
                txt = pg.TextItem(f"{last_live:.2f}", color='y', anchor=(0, 0.5))
                try:
                    txt.setPos(offsets[-1] + 0.02, last_live)
                except Exception:
                    txt.setPos(base_index + 0.98, last_live)
                self.plot.addItem(txt)

            # adjust x-range and autoscale y
            total_display = max(10, n + 3)
            self.plot.setXRange(-1, total_display)
            self.plot.enableAutoRange(axis=pg.ViewBox.YAxis)

            # update side info and equity plot
            last_candle_price = self.ohlc[-1]['close'] if len(self.ohlc) else None
            last_live_price = self.live_prices[-1] if len(self.live_prices) else None
            eq_val = (self.capital + self.position * (last_candle_price if last_candle_price else (last_live_price if last_live_price else 0)))
            info_html = "<b>Info</b><br>"
            if last_candle_price is not None:
                info_html += f"Candle close: {last_candle_price:.2f}<br>"
            if last_live_price is not None:
                info_html += f"Live price: {last_live_price:.2f}<br>"
            info_html += f"Position: {self.position:.6f}<br>"
            info_html += f"Capital: {self.capital:.2f}<br>"
            info_html += f"Equity: {eq_val:.2f}<br>"
            self.info_label.setText(info_html)

            # equity chart
            self.eq_plot.clear()
            if self.equity_curve:
                self.eq_plot.plot(list(range(len(self.equity_curve))), self.equity_curve, pen=pg.mkPen('g', width=2))
                self.eq_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        except Exception as e:
            # show error in status so redraw doesn't silently stop
            self.status_label.setText("Redraw error: " + str(e))

    def closeEvent(self, event):
        try:
            self.ws.stop()
        except Exception:
            pass
        event.accept()

# --------- run ----------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())
