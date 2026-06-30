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

import random
BLOCK_SIZE = 200
VISUAL_POINTS = ((150, 150), (850, 150), (850, 650), (150, 650))
INDEX = [random.randint(0, 3) for _ in range(40)]
print(f"INDEX: {INDEX}")

from analyse import TARGET_FREQS
#MARK: timer
class TimerThread(threading.Thread):
    def __init__(self, adder, serial, blink_duration: float, pause_duration: float):
        super().__init__(daemon=True)
        self.adder = adder
        self.serial = serial
        self.blink_duration = blink_duration
        self.pause_duration = pause_duration
        self.running = True

    def run(self) -> None:
        i = 0
        while self.running:
            if self.serial.status == 's':
                self.adder.remove_all_obj.emit()
                self.serial.send_event(COMMAND_PAUSE)
                time.sleep(0.1)
                continue
            if i == 0:
                self.adder.add_obj.emit(VisualObj((150-BLOCK_SIZE/2, 150-BLOCK_SIZE/2), (BLOCK_SIZE, BLOCK_SIZE), TARGET_FREQS[0], name="visualObj[1]", log_blink=True))
                self.adder.add_obj.emit(VisualObj((850-BLOCK_SIZE/2, 150-BLOCK_SIZE/2), (BLOCK_SIZE, BLOCK_SIZE), TARGET_FREQS[1], name="visualObj[2]", log_blink=True))
                self.adder.add_obj.emit(VisualObj((850-BLOCK_SIZE/2, 650-BLOCK_SIZE/2), (BLOCK_SIZE, BLOCK_SIZE), TARGET_FREQS[2], name="visualObj[3]", log_blink=True))
                self.adder.add_obj.emit(VisualObj((150-BLOCK_SIZE/2, 650-BLOCK_SIZE/2), (BLOCK_SIZE, BLOCK_SIZE), TARGET_FREQS[3], name="visualObj[4]", log_blink=True))
            # self.adder.remove_all_obj.emit()         
            
            # PAUSE
            self.adder.delete_visual_point.emit()
            self.serial.send_event(COMMAND_PAUSE)
            sleep_precise(self.pause_duration)
            
            #BLINK
            self.adder.add_visual_point.emit(*VISUAL_POINTS[INDEX[i % len(INDEX)]])
            self.serial.send_event(COMMAND_BLINK)
            sleep_precise(self.blink_duration)
            
            i = i + 1

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
    
    #初始化 serial
    #TODO: 修改输出文件
    output_file = 'four_86_ssvep.txt'
    serial_reader = SerialReader('COM6', 115200, output_file)
    serial_reader.start()
    
    with open(f"labels_{output_file}", '+a') as f:
        f.write(INDEX)
        
    segment_processor = BlinkSegmentProcessor(output_file)
    segment_processor.start()
    
    #初始化input
    input_thread = InputThread(serial_reader)
    input_thread.start()
    
    #初始化app
    app = QApplication(sys.argv)
    window = display()
    adder = ObjAdder()
    adder.add_obj.connect(window.add_visual_obj)
    adder.remove_all_obj.connect(lambda: window.visual_objs.clear() or window.update())
    adder.add_visual_point.connect(window.add_visual_point)
    adder.delete_visual_point.connect(window.delete_visual_point)
    
    #初始化timer
    timer = TimerThread(adder, serial_reader, blink_duration=2.0, pause_duration=3.0)
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
    
