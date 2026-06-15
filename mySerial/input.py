from Cython import struct
import serial
import threading
import time
import queue
from datetime import datetime

class SerialReader(threading.Thread):
    def __init__(self, port, baudrate, output_file):
        super().__init__()
        self.serial_port = serial.Serial(port, baudrate, timeout=1)
        self.output_file = output_file
        self.running = True
        self.command_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.pending_markers = 0
        self.last_command = None
        self.buffer = bytearray()
        self.status = 's'  # s: stop, b: begin

    def parse_data(self, frame):
        """解析33字节数据帧"""
        if len(frame) != 33 or frame[0] != 0xA0 or frame[32] != 0xC0:
            return None
        frame_num = frame[1]
        # 8通道脑电，每通道3字节，大端有符号24位
        eeg = []
        for i in range(2, 26, 3):
            val = (frame[i] << 16) | (frame[i+1] << 8) | frame[i+2]
            if val & 0x800000:
                val -= 0x1000000
            eeg.append(val)
        # 加速度 XH,XL, YH,YL, ZH,ZL，大端16位有符号
        return {'fn': frame_num, 'eeg': eeg}
    
    def run(self):
        with open(self.output_file, 'w') as f:
            
            while self.running:
                try:
                    # 检查是否有命令要发送
                    while not self.command_queue.empty():
                        command = self.command_queue.get()
                        self.serial_port.write(command.encode('utf-8'))

                    # 检查是否有事件标记（例如 blink 事件）
                    while not self.event_queue.empty():
                        event = self.event_queue.get_nowait()
                        if event == 'blink' and self.last_command == 'b\r\n':
                            self.pending_markers += 1
                            self.pending_id = "blink"
                        elif event == 'pause' and self.last_command == 'b\r\n':
                            self.pending_markers += 1
                            self.pending_id = "pause"
                        # elif event == 'start' and self.last_command == 'b\r\n':
                        #     self.pending_markers += 1
                        #     self.pending_id = "start"

                    if self.serial_port.in_waiting > 0:
                        data = self.serial_port.read(self.serial_port.in_waiting)
                        self.buffer.extend(data)
                        # 假设数据以换行符结束，解析为行
                        
                        if self.last_command == 'b\r\n':  # 只有在接收到开始传输命令后才处理数据
                            while len(self.buffer) >= 33:
                                frame = self.parse_data(self.buffer[:33])
                                self.buffer = self.buffer[33:]
                                # 简单解析：添加时间戳（精确到毫秒）
                                # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                timestamp = time.perf_counter()
                                marker = f' [MARKER] {self.pending_id}' if self.pending_markers > 0 else ''
                                if self.pending_markers > 0:
                                    self.pending_markers -= 1
                                output = f"[{timestamp}] {frame}{marker}"
                                f.write(output + '\n')
                                f.flush()
                        else:
                            while b'\x0D\x0A' in self.buffer:  # 换行符
                                line, self.buffer = self.buffer.split(b'\x0D\x0A', 1)
                                parsed_line = line.decode('utf-8', errors='ignore').strip()
                            while b'$$$' in self.buffer:
                                line, self.buffer = self.buffer.split(b'$$$', 1)
                                # print(f"Received: {parsed_line}")
                            
                except Exception as e:
                    print(f"Error: {e}")
                    break

    def send_command(self, command):
        """发送命令到串口"""
        if command == 'b\r\n':
            self.buffer.clear()
            self.status = 'b'
        elif command == 's\r\n':
            self.status = 's'
        self.command_queue.put(command)
        self.last_command = command

    def send_event(self, event):
        """发送串口事件，例如 blink 标记"""
        self.event_queue.put(event)

    def stop(self):
        self.running = False
        self.serial_port.close()

# 使用示例
if __name__ == "__main__":
    # 替换为你的串口端口和波特率
    reader = SerialReader('COM6', 115200, 'serial_output.txt')
    reader.start()
    print("Serial reader started. Enter commands to send (type 'exit' to quit).")
    
    try:
        while True:
            user_input = input("Command: ")
            if user_input.lower() == 'exit':
                break
            reader.send_command(user_input + "\r\n")
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        print("Serial reader stopped.")
        
'''
_1
b: 开始传输
s: 停止传输

'''
