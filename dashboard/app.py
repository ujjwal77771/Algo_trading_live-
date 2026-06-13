"""
PyQt Dashboard for Live Monitoring.
Reads purely from the TradeJournal SQLite DB.
"""

import sys
import sqlite3
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from src.utils.logger import logger

class LiveDashboard(QtWidgets.QMainWindow):
    """
    Dashboard application mapping state from the journal database.
    """
    
    def __init__(self, db_path: str = "journal.db") -> None:
        """
        Initializes the Live Dashboard.
        
        Args:
            db_path (str): Path to the SQLite journal database.
        """
        super().__init__()
        self.db_path = db_path
        self.conn = None
        self._connect_db()
        
        self.setWindowTitle("Market Monitor v2.0 - SQLite Sync")
        self.resize(1000, 800)
        
        self._init_ui()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(1000)

    def _connect_db(self) -> None:
        """Initializes the database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database at {self.db_path}: {e}")

    def _init_ui(self) -> None:
        """Sets up the UI elements."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        self.price_plot = pg.PlotWidget(title="Price Chart")
        self.equity_plot = pg.PlotWidget(title="Equity Curve")
        
        layout.addWidget(self.price_plot)
        layout.addWidget(self.equity_plot)
        
        self.price_curve = self.price_plot.plot(pen='lime')
        self.equity_curve = self.equity_plot.plot(pen='cyan')

    def refresh_data(self) -> None:
        """Reads data from the SQLite DB and updates plots."""
        if not self.conn:
            self._connect_db()
            if not self.conn:
                return

        try:
            # Read purely from SQLite instead of holding state
            df_price = pd.read_sql("SELECT timestamp, close FROM price_history ORDER BY timestamp DESC LIMIT 100", self.conn)
            df_equity = pd.read_sql("SELECT timestamp, equity FROM equity_history ORDER BY timestamp ASC", self.conn)
            
            if not df_price.empty:
                self.price_curve.setData(df_price['close'].values)
            if not df_equity.empty:
                self.equity_curve.setData(df_equity['equity'].values)
                
        except pd.errors.DatabaseError:
            # Handles expected missing tables during initialization silently
            pass
        except sqlite3.Error as e:
            logger.error(f"SQLite error refreshing dashboard data: {e}")

    def closeEvent(self, event: QtCore.QEvent) -> None:
        """Ensures DB connection is closed cleanly on exit."""
        if self.conn:
            try:
                self.conn.close()
            except sqlite3.Error as e:
                logger.error(f"Error closing DB connection: {e}")
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LiveDashboard()
    window.show()
    sys.exit(app.exec_())
