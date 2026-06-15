# Vibe2

Vibe2 是一个用于 SSVEP 脑电实验的 Python 项目。项目包含三个主要部分：

- 串口采集：从 EEG 设备读取 8 通道数据，并将数据写入 TXT。
- 视觉刺激：使用 PyQt5 显示不同频率的闪烁方块。
- 实时分析：按 `blink -> pause` 标记切分数据段，并使用 FBCCA 预测当前注视频率。

当前默认目标频率为 `15 Hz`、`13 Hz`、`11 Hz`、`9 Hz`。

## 项目结构

```text
.
├── main.py                     # 主程序：串口、视觉刺激、实时分析线程入口
├── requirements.txt            # Python 依赖
├── mySerial/
│   └── input.py                # 串口读取与事件标记写入
├── ui/
│   ├── display.py              # PyQt5 闪烁刺激窗口
│   └── graph.py                # 简单实时曲线组件
├── analyse/
│   ├── test_four.py            # EEG 预处理、FBCCA、实时分段处理
│   ├── test_realtime_four.py   # 离线回放 TXT 并测试实时分析流程
│   ├── realtime_results/       # 正式运行输出结果
│   └── realtime_test_segments/ # 离线测试输出结果
└── record/                     # 视觉刺激 blink 时间日志
```

## 环境准备

建议使用 Python 3.10 或更新版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果运行时提示缺少 `serial`、`pygame` 或 `Cython`，再安装：

```powershell
pip install pyserial pygame Cython
```

## 运行主程序

主程序会连接串口、启动闪烁刺激窗口，并实时处理采集到的数据。

```powershell
python main.py
```

运行前请检查 [main.py](D:/documents/vibe2/main.py) 中的这些配置：

- `SerialReader('COM6', 115200, output_file)`：串口号和波特率。
- `output_file = 'four_75_ssvep.txt'`：串口原始输出文件名。
- `blink_duration=2.0`：每轮闪烁持续时间，单位秒。
- `pause_duration=5.0`：两轮刺激之间的暂停时间，单位秒。

终端输入串口命令：

- `b`：开始传输 EEG 数据。
- `s`：停止传输 EEG 数据。
- `exit`：结束输入线程。

窗口关闭后，程序会依次停止计时线程、串口线程和实时分析线程。

## 离线回放测试

离线测试不连接串口，也不启动 GUI。它会读取已有 TXT 文件，按 `blink -> pause` 标记回放实时处理流程。

```powershell
python -B -m analyse.test_realtime_four
```

运行前请检查 [analyse/test_realtime_four.py](D:/documents/vibe2/analyse/test_realtime_four.py) 中的 `TXT_FILE` 是否指向实际存在的 TXT 文件。

测试结果会写入：

```text
analyse/realtime_test_segments/<运行时间>/
```

其中 `accuracy_summary.txt` 会记录每个数据段的期望频率、预测频率和最终准确率。

## 输出文件

正式运行的实时分析结果默认写入：

```text
analyse/realtime_results/
```

常见文件：

- `runtime.log`：处理日志。
- `eeg_segments.npy`：每个 blink 区间预处理后的 EEG 数据。
- `fbcca_scores.npy`：每个区间在四个目标频率上的 FBCCA 分数。
- `fbcca_prediction_indexes.npy`：预测频率在 `TARGET_FREQS` 中的索引。
- `fbcca_prediction_frequencies.npy`：预测出的实际频率，单位 Hz。
- `eeg_trials.npy`、`eeg_labels.npy`、`*.mat`：凑齐完整频率 block 后生成的汇总数据。

## 注意事项

- 本项目依赖真实 EEG 串口设备；没有设备时优先使用离线回放测试。
- `.txt` 采集文件、`.npy`、`.mat` 和运行日志通常属于实验数据或中间产物，提交前应确认是否需要纳入版本管理。
- `__pycache__`、临时输出目录和大体积数据文件建议保持忽略，避免仓库过大。
- 视觉刺激频率和 FBCCA 目标频率需要保持一致，否则会影响识别结果。
