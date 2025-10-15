import sys, time
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import random

prices = deque(maxlen=100)

class TestDashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Test Plot")
        self.plot_widget = pg.PlotWidget(title="Test Price")
        self.setCentralWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen='g')
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(200)

    def update_plot(self):
        prices.append(random.uniform(30000, 40000))  # simulate price
        self.curve.setData(list(range(len(prices))), list(prices))

app = QtWidgets.QApplication(sys.argv)
window = TestDashboard()
window.show()
