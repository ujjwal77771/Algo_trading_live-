import sys
import random
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque

class LiveDashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Window Setup ....
        self.setWindowTitle("Market Monitor v1.0")
        self.resize(800, 600)
        
        # Data Setup.....
        self.max_points = 100
        self.data_buffer = deque(maxlen=self.max_points)
        
        # Plot UI Setup____________
        self.plot_widget = pg.PlotWidget(title="Real-time Asset Price")
        self.plot_widget.showGrid(x=True, y=True)
        self.setCentralWidget(self.plot_widget)
        
        # Create the curve object once_-___-_-__
        self.curve = self.plot_widget.plot(pen=pg.mkPen('g', width=2))
        
        # Refresh Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(200)

    def refresh_data(self):
        # Generate dummy data
        new_price = random.uniform(30000, 40000)
        self.data_buffer.append(new_price)
        
        # Update the existing curve
        # Note: list() --conversion is necessary for pyqtgraph to process deques
        self.curve.setData(list(self.data_buffer))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = LiveDashboard()
    window.show()
    sys.exit(app.exec_())
