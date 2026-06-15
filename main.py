import pygame #@UnusedImport
from mySerial import SerialReader
from analyse.test_four import BlinkSegmentProcessor
import threading
import queue
import time

COMMAND_START = 'start'
COMMAND_BLINK = 'blink'
COMMAND_PAUSE = 'pause'


def sleep_precise(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while True:
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.001:
            time.sleep(remaining - 0.0005)

#MARK: timer
class TimerThread(threading.Thread):
    def __init__(self, adder, serial, blink_duration: float, pause_duration: float):
        super().__init__(daemon=True)
        self.adder = adder
        self.serial = serial
        self.blink_duration = blink_duration
        self.pause_duration = pause_duration
        self.running = True
        self.index = 0

    def run(self) -> None:
        while self.running:
            if self.serial.status == 's':
                self.adder.remove_all_obj.emit()
                self.serial.send_event(COMMAND_PAUSE)
                time.sleep(0.1)
                continue
            
            self.adder.remove_all_obj.emit()
            self.serial.send_event(COMMAND_PAUSE)
            sleep_precise(self.pause_duration)
            
            # self.adder.add_obj.emit(VisualObj((400, 400), (300, 300), 15, name="visualObj[1]", log_blink=True))

            # 1 2
            # 3 4
            self.adder.add_obj.emit(VisualObj((30, 30), (250, 250), 15, name="visualObj[1]", log_blink=True))
            self.adder.add_obj.emit(VisualObj((730, 30), (250, 250), 9, name="visualObj[2]", log_blink=True))
            
            # self.adder.add_obj.emit(VisualObj((380, 280), (250, 250), 10, name="visualObj[5]", log_blink=True))
            
            self.adder.add_obj.emit(VisualObj((730, 530), (250, 250), 11, name="visualObj[3]", log_blink=True))
            self.adder.add_obj.emit(VisualObj((30, 530), (250, 250), 13, name="visualObj[4]", log_blink=True))
            
            # if self.index % 4 == 0:
            #     self.adder.add_obj.emit(VisualObj((400, 400), (300, 300), 9, name="visualObj[1]", log_blink=True))
            # if self.index % 4 == 1:
            #     self.adder.add_obj.emit(VisualObj((400, 400), (300, 300), 10, name="visualObj[3]", log_blink=True))
            # if self.index % 4 == 2:
            #     self.adder.add_obj.emit(VisualObj((400, 400), (300, 300), 11, name="visualObj[4]", log_blink=True))
            # if self.index % 4 == 3:
            #     self.adder.add_obj.emit(VisualObj((400, 400), (300, 300), 12, name="visualObj[2]", log_blink=True))
            # self.index = self.index + 1
             
            self.serial.send_event(COMMAND_BLINK)
            sleep_precise(self.blink_duration)

    def stop(self) -> None:
        self.running = False

#MARK: user input
class InputThread(threading.Thread):
    def __init__(self, serial_reader: SerialReader):
        super().__init__(daemon=True)
        self.serial_reader = serial_reader
        self.running = True

    def run(self) -> None:
        try:
            while self.running:
                user_input = input("Command: ")
                if user_input.lower() == 'exit':
                    break
                self.serial_reader.send_command(user_input + "\r\n")
                
        except KeyboardInterrupt:
            pass
        finally:
            self.serial_reader.stop()
            print("Serial reader stopped.")

#MARK: main
from ui.display import display, VisualObj, ObjAdder
from PyQt5.QtWidgets import QApplication, QWidget
import sys
if __name__ == '__main__':
    now = time.perf_counter()
    # serial
    output_file = 'four_75_ssvep.txt'
    serial_reader = SerialReader('COM6', 115200, output_file)
    serial_reader.start()
    segment_processor = BlinkSegmentProcessor(output_file)
    segment_processor.start()
    # serial_reader.send_command('b\r\n')  # 初始清空缓冲区
    #input
    input_thread = InputThread(serial_reader)
    input_thread.start()
    #app
    app = QApplication(sys.argv)
    window = display()
    adder = ObjAdder()
    adder.add_obj.connect(window.add_visual_obj)
    adder.remove_all_obj.connect(lambda: window.visual_objs.clear() or window.update())
    #timer
    timer = TimerThread(adder, serial_reader, blink_duration=2.0, pause_duration=5.0)
    timer.start()

    window.setWindowTitle("display blink test - 15Hz")
    window.resize(1200, 900)
    window.show()
    app.exec_()
    
    # ========== 窗口关闭后，执行清理 ==========
    print("GUI closed, stopping threads...")
    # 1. 停止 TimerThread（停止闪烁循环）
    timer.stop()
    timer.join(timeout=2.0)  # 等最多2秒，让它自然退出循环
    # 2. 停止 InputThread（input() 阻塞无法中断，只能设标志让它下次循环退出）
    input_thread.running = False
    # input() 本身无法从外部强制打断，如果卡在 input() 这里会等用户按回车
    # 可选：打印一个提示到终端，提醒用户按回车
    print("请按回车键以结束输入线程...")
    input_thread.join(timeout=3.0)
    # 3. 停止 SerialReader
    serial_reader.stop()
    # 4. 停止实时分析线程
    segment_processor.stop()
    segment_processor.join(timeout=2.0)
    print("All stopped.")
    
