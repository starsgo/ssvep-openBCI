# 开发说明

## 核心流程

1. `main.py` 创建 `SerialReader`，连接 EEG 设备串口。
2. `main.py` 创建 PyQt5 窗口，并通过 `TimerThread` 周期性显示四个闪烁刺激。
3. `TimerThread` 在刺激开始时写入 `blink` 标记，在暂停阶段写入 `pause` 标记。
4. `SerialReader` 将 EEG 帧和标记写入 TXT。
5. `BlinkSegmentProcessor` 监听 TXT 新增内容，每遇到完整 `blink -> pause` 区间就执行一次预处理和 FBCCA。
6. 分析结果汇总保存到 `analyse/realtime_results/`。

## 关键文件

- [main.py](D:/documents/vibe2/main.py)：实验运行入口，串口、GUI、计时和分析线程都在这里组装。
- [mySerial/input.py](D:/documents/vibe2/mySerial/input.py)：负责串口命令发送、帧解析和 TXT 写入。
- [ui/display.py](D:/documents/vibe2/ui/display.py)：负责闪烁刺激绘制和高精度边沿调度。
- [analyse/test_four.py](D:/documents/vibe2/analyse/test_four.py)：负责 EEG 预处理、滤波器组、CCA 和实时分段保存。
- [analyse/test_realtime_four.py](D:/documents/vibe2/analyse/test_realtime_four.py)：负责离线回放测试。

## 调整实验参数

常改位置：

- 串口号：`main.py` 中的 `SerialReader('COM6', 115200, output_file)`。
- 输出 TXT：`main.py` 中的 `output_file`。
- 刺激位置和频率：`TimerThread.run()` 中创建 `VisualObj(...)` 的位置。
- 刺激持续时间：`main.py` 中 `TimerThread(..., blink_duration=..., pause_duration=...)`。
- FBCCA 目标频率：`analyse/test_four.py` 中的 `TARGET_FREQS`。
- 采样率：`analyse/test_four.py` 中的 `SAMPLE_RATE`。

修改刺激频率时，请同步修改 `TARGET_FREQS`，并确认离线测试中的期望顺序仍然正确。

## 验证建议

没有串口设备时，优先跑离线测试：

```powershell
python -B -m analyse.test_realtime_four
```

如果只想确认核心模块能导入：

```powershell
python -B -c "from analyse.test_four import BlinkSegmentProcessor; print('imports ok')"
```

## Git 注意事项

当前项目会产生较多实验数据和运行产物，包括：

- `*.txt`
- `*.npy`
- `*.mat`
- `runtime.log`
- `__pycache__/`

提交前建议先查看：

```powershell
git status --short
```

只提交确认为源码、文档或需要保留的小型配置文件的数据。
