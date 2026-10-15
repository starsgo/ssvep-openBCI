"""Optionally connect gaze before starting devices and the SSVEP experiment."""
from __future__ import annotations

from types import SimpleNamespace
from gaze import GazePointProvider
from gaze.api import ApiGazeTracker


class StartScreen:
    def __init__(self, default_gaze_source="api", api_url="http://127.0.0.1:8765/gaze"):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title("Vibe4 实验")
        self.root.geometry("520x500")
        self.root.resizable(False, False)
        self.root.configure(background="#f8fafc")
        self.source = default_gaze_source
        self.api_url = api_url
        self.tracker = None
        self.gaze_point = GazePointProvider()
        self.selection = None
        tk.Label(self.root, text="Vibe4 实验", background="#f8fafc", foreground="#172033",
                 font=("Microsoft YaHei", 22, "bold")).pack(pady=(20, 8))
        tk.Label(self.root, text="可直接开始实验，也可先获取注视。", background="#f8fafc",
                 foreground="#475569", font=("Microsoft YaHei", 11)).pack()
        self.preview = tk.Canvas(self.root, width=440, height=180, background="#e2e8f0",
                                 highlightthickness=0)
        self.preview.pack(pady=12)
        self.status = tk.Label(self.root, text="未获取注视 · 默认坐标 (0, 0)", background="#f8fafc",
                               foreground="#475569", font=("Microsoft YaHei", 10))
        self.status.configure(wraplength=470)
        self.status.pack(pady=4)
        row = tk.Frame(self.root, background="#f8fafc")
        row.pack(pady=12)
        self.connect_button = tk.Button(row, text="获取注视", command=self.fetch_gaze,
                                        font=("Microsoft YaHei", 13), width=13, height=2)
        self.connect_button.pack(side=tk.LEFT, padx=8)
        self.start_button = tk.Button(row, text="实验开始", command=self.start,
                                      background="#2563eb", foreground="white", relief=tk.FLAT,
                                      font=("Microsoft YaHei", 13), width=13, height=2,
                                      state="normal")
        self.start_button.pack(side=tk.LEFT, padx=8)
        tk.Label(self.root, text="眼动程序：视线估计 → 提供注视；这里再点击获取注视。", background="#f8fafc",
                 foreground="#64748b", font=("Microsoft YaHei", 9)).pack()
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self.root.bind("<Escape>", lambda event: self.cancel())
        self._poll_gaze()

    def fetch_gaze(self):
        self.source = "api"
        if self.tracker is None:
            self.tracker = ApiGazeTracker(self.api_url)
            self.gaze_point.source = self.tracker
            self.tracker.start()
        self.status.configure(text="正在连接眼动程序，等待有效注视坐标……")

    def _poll_gaze(self):
        if self.selection is not None:
            return
        position = self.gaze_point.get_current_position()
        self.preview.delete("gaze")
        x = position[0] / self.gaze_point.window_size[0] * 440
        y = position[1] / self.gaze_point.window_size[1] * 180
        self.preview.create_line(x-8, y, x+8, y, fill="red", width=2, tags="gaze")
        self.preview.create_line(x, y-8, x, y+8, fill="red", width=2, tags="gaze")
        if self.tracker is not None:
            if not self.gaze_point.current_is_valid:
                note = "保留最后坐标" if self.gaze_point.has_measurement else "默认坐标"
                if self.tracker.api_connected:
                    detail = self.tracker.last_error or "尚未收到有效样本"
                    self.status.configure(text=f"当前无有效眼动：{detail} · {note} {position}")
                else:
                    self.status.configure(text=f"未连接眼动服务 · {note} {position}")
            else:
                self.status.configure(text=f"已连接 · 实时坐标 ({position[0]}, {position[1]})")
        self.root.after(20, self._poll_gaze)

    def start(self):
        self.gaze_point.get_current_position()
        self.selection = SimpleNamespace(source=self.source, tracker=self.tracker,
                                         gaze_point=self.gaze_point)
        self.start_button.configure(state="disabled")
        self.root.quit()

    def cancel(self):
        self.selection = None
        self.root.quit()

    def show(self):
        try:
            self.root.mainloop()
            return self.selection
        finally:
            if self.selection is None and self.tracker is not None:
                self.tracker.stop()
                self.tracker.join(timeout=1.)
            self.root.destroy()


def show_start_screen(default_gaze_source="api", api_url="http://127.0.0.1:8765/gaze"):
    return StartScreen(default_gaze_source, api_url).show()
