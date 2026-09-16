from mySerial import SerialReader
from analyse.test_four import BlinkSegmentProcessor
import threading
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
VISUAL_POINTS = ((150, 150), (1250, 150), (700, 650))
INDEX = [random.randint(0, 2) for _ in range(40)]
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
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return not self._stop_event.is_set()

    def _wait(self, seconds: float) -> bool:
        """Wait interruptibly; return False when shutdown was requested."""
        return not self._stop_event.wait(max(0.0, seconds))

    def run(self) -> None:
        i = 0
        pause_marker_needed = True
        while self.running:
            if self.serial.status == 's':
                self.adder.remove_all_obj.emit()
                self.adder.delete_visual_point.emit()
                pause_marker_needed = True
                if not self._wait(0.1):
                    break
                continue

            # PAUSE
            self.adder.remove_all_obj.emit()            # 所有闪烁目标消失
            self.adder.delete_visual_point.emit()       # 提示点消失
            if pause_marker_needed:
                self.serial.send_event(COMMAND_PAUSE)   # 首轮开始时发送暂停标记
            if not self._wait(self.pause_duration):     # 休息时间：2.5s
                break
            self.adder.add_visual_point.emit(*VISUAL_POINTS[INDEX[i % len(INDEX)]])     # 提示点出现
            if not self._wait(0.5):                                                     # 提示点0.5s
                break

            #BLINK
            if not self._wait(0.5):                                                     # 眼动0.5s
                break
            objects = [
                VisualObj((150-BLOCK_SIZE/2, 150-BLOCK_SIZE/2), (BLOCK_SIZE, BLOCK_SIZE), TARGET_FREQS[0], name="visualObj[1]", log_blink=True),
                VisualObj((1250-BLOCK_SIZE/2, 150-BLOCK_SIZE/2), (BLOCK_SIZE, BLOCK_SIZE), TARGET_FREQS[1], name="visualObj[2]", log_blink=True),
                VisualObj((700-BLOCK_SIZE/2, 650-BLOCK_SIZE/2), (BLOCK_SIZE, BLOCK_SIZE), TARGET_FREQS[2], name="visualObj[3]", log_blink=True),
            ]
            trial_done = threading.Event()
            self.adder.start_trial.emit(
                objects,
                self.blink_duration,
                lambda: self.serial.send_event(COMMAND_BLINK),
                lambda: self.serial.send_event(COMMAND_PAUSE),
                trial_done,
            )
            while self.running and not trial_done.wait(0.05):
                pass
            if not self.running:
                break
            pause_marker_needed = False

            i = i + 1

    def stop(self) -> None:
        self._stop_event.set()

#MARK: user input
class InputThread(threading.Thread):
    def __init__(self, serial_reader: SerialReader, on_exit=None):
        super().__init__(daemon=True)
        self.serial_reader = serial_reader
        self.on_exit = on_exit
        self.running = True

    def run(self) -> None:
        try:
            # 自动启动
            sleep_precise(1.0)  # 等待1秒，确保串口已准备好
            self.serial_reader.send_command("b\r\n")

            while self.running:
                user_input = input("Command: ")
                if user_input.lower() == 'exit':
                    break
                self.serial_reader.send_command(user_input + "\r\n")

        except KeyboardInterrupt:
            pass
        finally:
            self.serial_reader.stop()
            if self.on_exit is not None:
                self.on_exit()
            print("Serial reader stopped.")

#MARK: main
from ui.display_psy import display, VisualObj, ObjAdder

if __name__ == '__main__':
    now = time.perf_counter()

    #初始化 serial
    #TODO: 修改输出文件
    output_file = 'test_0_ssvep.txt'
    serial_reader = SerialReader('COM5', 115200, output_file)
    serial_reader.start()

    with open(f"labels_{output_file}", '+a') as f:
        for x in INDEX:
            f.write(str(x) + "\n")

    segment_processor = BlinkSegmentProcessor(output_file)
    segment_processor.start()

    # 初始化 PsychoPy 窗口和跨线程信号
    window = display()
    adder = ObjAdder()
    adder.add_obj.connect(window.add_visual_obj)
    adder.remove_all_obj.connect(lambda: window.visual_objs.clear() or window.update())
    adder.add_visual_point.connect(window.add_visual_point)
    adder.delete_visual_point.connect(window.delete_visual_point)
    adder.start_trial.connect(window.start_visual_trial)
    adder.close_requested.connect(window.request_close)

    #初始化input
    input_thread = InputThread(serial_reader, on_exit=adder.close_requested.emit)
    input_thread.start()

    #初始化timer
    timer = TimerThread(adder, serial_reader, blink_duration=1.5, pause_duration=2.5)
    timer.start()

    try:
        window.setWindowTitle("PsychoPy SSVEP display")
        window.show()
        window.run(adder)
    finally:
        # ========== 窗口关闭后，执行清理 ==========
        print("GUI closed, stopping threads...")
        timer.stop()
        timer.join(timeout=2.0)
        input_thread.running = False
        input_thread.join(timeout=1.0)
        serial_reader.stop()
        segment_processor.stop()
        segment_processor.join(timeout=2.0)
        record_dir = window.close()
        print(f"Timing records saved to: {record_dir.resolve()}")
        print("All stopped.")
        
# O1 O2 P3 P4 Ref Pz C3 C4 
