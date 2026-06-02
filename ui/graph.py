import math
import sys
import time
from collections import deque

from PyQt5.QtCore import QPointF, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QApplication, QWidget


class graph(QWidget):
    def __init__(self, sample_rate=250, seconds=4, y_min=-1.0, y_max=1.0, parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.max_points = sample_rate * seconds
        self.y_min = y_min
        self.y_max = y_max
        self.data = deque([0.0] * self.max_points, maxlen=self.max_points)
        self.setMinimumSize(480, 240)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.update)
        self.refresh_timer.start(33)

    def receive_data(self, value):
        # deque 到 maxlen 后会自动丢掉最老的点，只保留最近 seconds 秒。
        self.data.append(float(value))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.setRenderHint(QPainter.Antialiasing, False)

        if len(self.data) < 2 or self.y_min == self.y_max:
            return

        values = list(self.data)

        margin = 12
        width = max(1, self.width() - margin * 2)
        height = max(1, self.height() - margin * 2)
        step = width / (self.max_points - 1)

        points = QPolygonF()
        for i, value in enumerate(values):
            x = margin + i * step
            value = max(self.y_min, min(self.y_max, value))
            y = margin + (self.y_max - value) / (self.y_max - self.y_min) * height
            points.append(QPointF(x, y))

        painter.setPen(QPen(QColor("#d0d7de"), 1))
        painter.drawRect(margin, margin, width, height)
        painter.setPen(QPen(QColor("#2563eb"), 2))
        painter.drawPolyline(points)


class GraphTester:
    def __init__(self, target_graph, hz=250):
        self.graph = target_graph
        self.start = time.perf_counter()
        self.timer = QTimer()
        self.timer.timeout.connect(self.send_data)
        self.timer.start(max(1, round(1000 / hz)))

    def send_data(self):
        t = time.perf_counter() - self.start
        self.graph.receive_data(math.sin(2 * math.pi * 3 * t))


def test_graph():
    app = QApplication(sys.argv)
    window = graph()
    window.setWindowTitle("graph test - 250Hz")
    window.resize(700, 320)
    window.show()
    tester = GraphTester(window, hz=250)
    window.tester = tester
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(test_graph())
