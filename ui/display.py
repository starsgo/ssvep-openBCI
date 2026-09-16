import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget

@dataclass
class VisualObj:
    position: tuple
    size: tuple
    blink_frequency: float = 0.0
    image: object = None
    name: str = "visualObj"
    log_blink: bool = False
    # perf_counter 是高精度单调时钟，适合做短时间间隔计算。
    start: float = field(default_factory=time.perf_counter)
    last_edge: int = 0
    _log_path: Path | None = field(default=None, init=False, repr=False)

    def set_record_dir(self, record_dir):
        """Attach this object to one display run's timestamped log directory."""
        if not self.log_blink:
            return
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        log_path = record_dir / f"{self.name}_log.txt"
        if self._log_path == log_path:
            return
        self._log_path = log_path
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"-------------{self.name} initialized at "
                f"{self.start:.6f}-------\n"
            )
        
    def pixmap(self):
        if isinstance(self.image, QPixmap):
            return self.image
        if isinstance(self.image, QImage):
            return QPixmap.fromImage(self.image)
        return QPixmap(self.image) if self.image else None
    
    def edge(self, now=None):
        if self.blink_frequency <= 0:
            return 0
        now = time.perf_counter() if now is None else now
        # 15Hz 的完整周期是 1/15 秒；闪烁亮/灭各占半周期，即 0.5 / frequency。
        # 当前 elapsed 落在第几个半周期，就代表第几次亮灭边沿。
        return int(max(0, now - self.start) / (0.5 / self.blink_frequency))
    
    def visible(self, now=None):
        return self.blink_frequency <= 0 or self.edge(now) % 2 == 0
    
    def update_blink(self, now=None): # 观察edge有没有发生反转，如有打印时间，更新lastedge
        now = time.perf_counter() if now is None else now
        edge = self.edge(now)
        # print(f"debug -----------{edge}---------")
        # edge 变化才说明发生了一次亮/灭翻转；打印的就是实际翻转被处理的时间。
        if edge == self.last_edge:
            return
        if self.log_blink and self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(f"{self.name} blink at {now - self.start:.6f}s edge={self.edge(now)}\n")
            # print(f"{self.name} blink at {now - self.start:.6f}s edge={self.edge(now)}")
        self.last_edge = edge
        
    def next_blink_ms(self, now=None): #计算下一次edge的时间
        if self.blink_frequency <= 0:
            return None
        now = time.perf_counter() if now is None else now
        # 下一次唤醒按“绝对时间轴”计算，而不是按上次 tick 后再延时。
        # 如果某次 Qt 定时器晚了，下次仍会回到 start + N * half_period，误差不会累积。
        next_time = self.start + (self.edge(now) + 1) * 0.5 / self.blink_frequency
        # print(f"debug {next_time=:.6f} now={now:.6f} edge={self.edge(now)}")
        return max(1, round((next_time - now) * 1000))
    
    @property
    def rect(self):
        return QRectF(*self.position, *self.size)

class display(QWidget):
    def __init__(
        self,
        visual_objs=None,
        parent=None,
        background=QColor("#000000"),
        record_root="record",
        run_id=None,
        fullscreen=True,
    ):
        super().__init__(parent)
        self.visual_objs = list(visual_objs or [])
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_dir = Path(record_root) / self.run_id
        for obj in self.visual_objs:
            obj.set_record_dir(self.record_dir)
        self.background = background
        self.fullscreen = bool(fullscreen)
        self._red_dot_rect = None
        self.setMinimumSize(320, 240)
        if self.fullscreen:
            self.setWindowState(self.windowState() | Qt.WindowMaximized)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        # Qt.PreciseTimer 请求 Qt 尽量用高精度定时器；实际精度仍受系统调度影响。
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.schedule()
        
    def tick(self):
        now = time.perf_counter()
        for obj in self.visual_objs:
            obj.update_blink(now)
        self.update()
        self.schedule()
        
    def schedule(self):
        # 多个对象频率可能不同，画布只睡到最近那个对象的下一次翻转。
        waits = [w for obj in self.visual_objs for w in [obj.next_blink_ms()] if w is not None]
        if waits:
            self.timer.start(min(waits))
            
    def add_visual_obj(self, obj):
        obj.set_record_dir(self.record_dir)
        self.visual_objs.append(obj)
        self.schedule(); self.update()

    def set_fullscreen(self, enabled=True):
        """Switch between a maximized switchable window and normal mode."""
        self.fullscreen = bool(enabled)
        if self.fullscreen:
            self.showMaximized()
        else:
            self.showNormal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key_F11:
            self.set_fullscreen(not self.fullscreen)
            return
        super().keyPressEvent(event)

    def add_visual_point(self, x, y):
        radius = 5
        old_rect = self._red_dot_rect
        self._red_dot_rect = QRectF(x - radius, y - radius, radius * 10, radius * 10)
        dirty_rect = self._red_dot_rect if old_rect is None else old_rect.united(self._red_dot_rect)
        self.update(dirty_rect.toAlignedRect())

    def delete_visual_point(self):
        if self._red_dot_rect is not None:
            dirty_rect = self._red_dot_rect.toAlignedRect()
            self._red_dot_rect = None
            self.update(dirty_rect)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.background)
        now = time.perf_counter()
        for obj in self.visual_objs:
            if not obj.visible(now):
                continue
            pix = obj.pixmap()
            if pix and not pix.isNull():
                painter.drawPixmap(obj.rect, pix, QRectF(pix.rect()))
            else:
                painter.fillRect(obj.rect, QColor("#FFFFFF"))
        if self._red_dot_rect is not None:
            painter.setPen(Qt.NoPen)
            painter.setBrush(Qt.red)
            painter.drawEllipse(self._red_dot_rect)

class ObjAdder(QObject):
    add_obj = pyqtSignal(object)
    remove_all_obj = pyqtSignal()
    add_visual_point = pyqtSignal(int, int)
    delete_visual_point = pyqtSignal()

def test_display():
    app = QApplication(sys.argv)
    window = display()
    adder = ObjAdder()
    adder.add_obj.connect(window.add_visual_obj)
    adder.remove_all_obj.connect(lambda: window.visual_objs.clear() or window.update())
    adder.add_visual_point.connect(window.add_visual_point)
    adder.delete_visual_point.connect(window.delete_visual_point)
    
    def add_later():
        for i in range(3):
            time.sleep(1)
            obj = VisualObj((30 + 110 * i, 40), (80, 80), 15, name=f"visualObj[{i + 1}]", log_blink=True)
            adder.add_obj.emit(obj)

    threading.Thread(target=add_later, daemon=True).start()

    window.setWindowTitle("display blink test - 15Hz")
    window.resize(380, 180)
    window.show()
    return app.exec_()

visualObj = VisualObj
if __name__ == "__main__":
    raise SystemExit(test_display())
