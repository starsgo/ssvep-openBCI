# 开发说明

## 核心流程

1. `main.py` 创建 `SerialReader`，连接 EEG 设备串口。
2. `main.py` 从 `object.json` 读取目标，并在主线程创建 PsychoPy 窗口；`TimerThread` 按休息、提示、眼动和闪烁阶段提交原子刺激试次。
3. PsychoPy 在首个目标 flip 上发送 `blink`，在清屏 flip 上发送 `pause`。
4. `SerialReader` 将 EEG 帧和标记写入 TXT。
5. `BlinkSegmentProcessor` 监听 TXT 新增内容，每遇到完整 `blink -> pause` 区间就截取最多 375 点，并执行一次预处理和 FBCCA。
6. 分析结果汇总保存到 `analyse/realtime_results/`。

## 关键文件

- `main.py`：实验运行入口，串口、GUI、计时和分析线程都在这里组装。
- `experiment.py`：目标 JSON 校验和随机 INDEX。
- `gaze/`：鼠标注视采样、PsychoPy 坐标转换和左右半屏判断。
- `mySerial/input.py`：负责串口命令发送、帧解析和 TXT 写入。
- `ui/display_psy.py`：默认后端，负责目标构建、预测文字、VSync 逐帧余弦刺激、相位记录和诊断图。
- `ui/display.py`：保留的旧 PyQt5 后端。
- `analyse/test_four.py`：负责实时分段、预处理、回调和结果保存。
- `analyse/fbcca.py`：负责 5 个滤波器组、4 个谐波的在线 FBCCA 分类。
- `analyse/prediction_results.py`：按预测频率与注视侧反查目标，并保存逐轮结果和正确率。
- `analyse/test_realtime_four.py`：负责离线回放测试。

## 调整实验参数

常改位置：

- 串口号：`main.py` 中的 `SerialReader('COM6', 115200, output_file)`。
- 输出 TXT：`main.py` 中的 `output_file`。
- 刺激所属左右屏、位置、尺寸和频率：根目录 `object.json`；左右归属读取 `screen` 字段，不通过坐标推断。
- 刺激持续时间：`main.py` 中 `TimerThread(..., blink_duration=..., pause_duration=...)`。
- FBCCA 目标频率：运行时按 `object.json` 的目标顺序传入。
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
