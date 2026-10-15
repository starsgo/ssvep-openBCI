"""Connect real gaze before starting devices and the SSVEP experiment."""
from __future__ import annotations

from types import SimpleNamespace
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
        self.selection = None
        tk.Label(self.root, text="Vibe4 实验", background="#f8fafc", foreground="#172033",
                 font=("Microsoft YaHei", 22, "bold")).pack(pady=(20, 8))
        tk.Label(self.root, text="先获取注视，再开始实验。", background="#f8fafc",
                 foreground="#475569", font=("Microsoft YaHei", 11)).pack()
        self.preview = tk.Canvas(self.root, width=440, height=180, background="#e2e8f0",
                                 highlightthickness=0)
        self.preview.pack(pady=12)
        self.status = tk.Label(self.root, text="未连接眼动程序", background="#f8fafc",
                               foreground="#475569", font=("Microsoft YaHei", 10))
        self.status.pack(pady=4)
        row = tk.Frame(self.root, background="#f8fafc")
        row.pack(pady=12)
        self.connect_button = tk.Button(row, text="获取注视", command=self.fetch_gaze,
                                        font=("Microsoft YaHei", 13), width=13, height=2)
        self.connect_button.pack(side=tk.LEFT, padx=8)
        self.start_button = tk.Button(row, text="实验开始", command=self.start,
                                      background="#2563eb", foreground="white", relief=tk.FLAT,
                                      font=("Microsoft YaHei", 13), width=13, height=2,
                                      state="disabled" if self.source == "api" else "normal")
        self.start_button.pack(side=tk.LEFT, padx=8)
        tk.Label(self.root, text="眼动程序需先启动坐标服务；红十字表示实时注视位置。", background="#f8fafc",
                 foreground="#64748b", font=("Microsoft YaHei", 9)).pack()
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self.root.bind("<Escape>", lambda event: self.cancel())
        self.root.after(20, self._poll_gaze)

    def fetch_gaze(self):
        self.source = "api"
        self.start_button.configure(state="disabled")
        if self.tracker is None:
            self.tracker = ApiGazeTracker(self.api_url)
            self.tracker.start()
        self.status.configure(text="正在连接眼动程序，等待有效注视坐标……")

    def _poll_gaze(self):
        if self.selection is not None:
            return
        if self.tracker is not None:
            position = self.tracker.get_current_position()
            self.preview.delete("gaze")
            if position is None:
                self.status.configure(text="等待有效注视坐标，请检查眼动程序是否运行")
                self.start_button.configure(state="disabled")
            else:
                x = position[0] / self.tracker.window_size[0] * 440
                y = position[1] / self.tracker.window_size[1] * 180
                self.preview.create_line(x-8, y, x+8, y, fill="red", width=2, tags="gaze")
                self.preview.create_line(x, y-8, x, y+8, fill="red", width=2, tags="gaze")
                self.status.configure(text=f"已连接 · 实时坐标 ({position[0]}, {position[1]})")
                self.start_button.configure(state="normal")
        self.root.after(20, self._poll_gaze)

    def start(self):
        if self.source == "api" and (self.tracker is None or self.tracker.get_current_position() is None):
            self.status.configure(text="请先点击获取注视，并等待有效坐标")
            return
        self.selection = SimpleNamespace(source=self.source, tracker=self.tracker)
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
