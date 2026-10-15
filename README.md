# Vibe4

Vibe2 是一个用于 SSVEP 脑电实验的 Python 项目。项目包含三个主要部分：

- 串口采集：从现有 8 通道帧格式的 EEG 设备读取数据，并将数据写入 TXT。
- 视觉刺激：默认使用 PsychoPy 按显示器刷新帧生成余弦亮度闪烁；旧 PyQt5 后端保留备用。
- 实时分析：按 `blink -> pause` 标记切分每段最多 1.5 秒（375 点）的数据，并使用 FBCCA 预测当前注视频率。
- 实时眼动输入：通过 API 获取真实注视坐标，红十字实时显示，并按眼动阶段末的注视侧选择闪烁目标；鼠标模式可用于显式调试。

当前屏幕由中心连向顶边中点、左下角、右下角的三条细线分为左上、右上、下方三个区域。每区默认配置 `15 Hz`、`12 Hz`、`10 Hz` 目标，共 9 个；每段脑电只在实际闪烁区域现有目标的频率集合中分类，不使用其他区域的候选频率。

## 项目结构

```text
.
├── main.py                     # 主程序：串口、视觉刺激、实时分析线程入口
├── experiment.py               # 目标配置校验与随机 INDEX
├── gaze/                       # 真实眼动 API、坐标转换与左右判断
├── object.json                 # 目标位置、尺寸和频率配置
├── requirements.txt            # 通用 Python 依赖
├── requirements-psychopy.txt  # PsychoPy 实验环境依赖
├── mySerial/
│   └── input.py                # 串口读取与事件标记写入
├── ui/
│   ├── display.py              # 旧 PyQt5 闪烁刺激窗口
│   ├── display_psy.py          # PsychoPy 逐帧刺激、计时记录与诊断图
│   └── graph.py                # 简单实时曲线组件
├── analyse/
│   ├── test_four.py            # EEG 预处理、FBCCA、实时分段处理
│   ├── fbcca.py                # 单段在线 Filter-Bank CCA 分类
│   ├── prediction_results.py   # 目标反查、逐轮结果与正确率保存
│   ├── test_realtime_four.py   # 离线回放 TXT 并测试实时分析流程
│   ├── realtime_results/       # 正式运行输出结果
│   └── realtime_test_segments/ # 离线测试输出结果
└── record/                     # 视觉刺激 blink 时间日志
```

## 环境准备

正式刺激环境使用 Python 3.11，安装包含 PsychoPy 的完整依赖：

```powershell
conda create -n BCI-psychopy python=3.11 pip -y
conda activate BCI-psychopy
python -s -m pip install -r requirements-psychopy.txt
```

只运行离线分析或旧 PyQt5 界面时，可在 Python 3.11 环境中安装基础依赖：

```powershell
python -m pip install -r requirements.txt
```

依赖文件只列出项目直接使用的包，不包含本机路径；pip 会根据当前系统安装间接依赖。

## 运行主程序

主程序先显示开始界面，可直接点击“实验开始”，也可先点击“获取注视”连接眼动程序。未取得坐标时注视点初始为 `(0,0)`，取得后实时更新；断流或无效时保留最近的历史有效点。点击“实验开始”后才连接串口、启动全屏 PsychoPy 刺激窗口。关闭开始界面不会启动设备或实验。

```powershell
python -s main.py
```

运行前请检查 `main.py` 和 `object.json` 中的这些配置：

- `SerialReader('COM5', 115200, output_file)`：串口号和波特率。
- `output_file = 'test_0_ssvep.txt'`：串口原始输出文件名。
- `object.json`：每个目标配置 `screen`（`left`、`right` 或 `bottom`）、左上角坐标、尺寸、`frequency_hz`、`shape`、`color` 和稳定编号 `label`；根字段 `region_layout: three_sectors` 固定三分区。删除目标时可保留其余目标的原编号，不需要重新排序编号。
- 默认阶段时长依次为休息 2.5 秒、提示 0.5 秒、眼动 0.5 秒和闪烁 1.5 秒。

实时分类支持 TXT 中每条 `eeg` 记录含 4 个或 8 个数，同一段 `blink -> pause` 内应保持相同通道数。若设备仍输出 8 个数、后 4 个是补零，也可以直接分类。预处理会忽略恒定的补零通道，并处理数值为 `-8388608` 或 `8388607` 的 ADC 饱和采样点；若一段中所有通道都不可用，该段会记录为跳过，不输出预测。当前串口读取器仍按设备原有的 33 字节、8 通道帧格式解析；如果使用不同的 4 通道二进制帧格式，需要另行调整串口解析。

休息、提示和眼动阶段显示中灰背景、目标配置的静止颜色及数字编号；闪烁阶段隐藏全部目标数字，结束后恢复。默认 gaze 模式在眼动阶段结束时使用最近的历史有效注视点决定左上、右上或下方闪烁区域，历史不局限于本轮 0.5 秒，从未取得坐标时使用初始点 `(0,0)`；加 `--nogaze` 时使用本轮目标的 `screen` 配置决定闪烁区域。上一段预测会在休息阶段以 `predit:<频率>Hz 左边/右边/下方` 显示。系统按预测频率与实际闪烁侧反查目标 INDEX，并将其与实际 INDEX 的逐轮比较保存到 `record/<运行时间>/prediction_results.csv`。

终端输入串口命令：

- `b`：开始传输 EEG 数据。
- `s`：停止传输 EEG 数据。
- `exit`：结束输入线程。

Escape 只退出全屏，不停止窗口或串口；串口 EEG 仍需手动输入 `s` 停止。关闭窗口或在终端输入 `exit` 后，程序会停止所有线程并把逐帧记录保存到 `record/<运行时间>/`。

主程序使用 PsychoPy 真全屏 `fullscr=True` 和逻辑主屏 `screen=0`。旧 PyQt5 后端 `ui/display.py` 默认使用可切换的最大化窗口，F11 可切换窗口模式；传入 `fullscreen=False` 可用于窗口化调试。它的亮灭边沿日志按运行隔离，保存为 `record/<运行时间>/visualObj[x]_log.txt`。PsychoPy 后端不使用这些边沿日志，而是保存逐帧 CSV。

## PsychoPy 视觉计时测试

不连接 EEG 串口，仅运行一次 1.5 秒的 15/12/10 Hz 全屏刺激：

```powershell
conda activate BCI-psychopy
python -s -m ui.display_psy --duration 1.5
```

每个目标生成一个逐帧 CSV，同时输出 `waveform_diagnostics.png`、`waveform_3d_trials.png` 和 `timing_summary.json`。二维诊断图显示实际下发的余弦亮度、亮度误差和共享帧间隔；三维地形图以 trial、时间和亮度分别作为 X、Y、Z 轴。屏幕物理发光时序仍应使用光电二极管验证。

## 离线回放测试

离线测试不连接串口，也不启动 GUI。它会读取已有 TXT 文件，按 `blink -> pause` 标记回放实时处理流程。

```powershell
python -s -B -m analyse.test_realtime_four
```

运行前请检查 `analyse/test_realtime_four.py` 中的 `TXT_FILE` 和 `LABEL_FILE` 是否指向配套文件。

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
- `eeg_channel_counts.npy`：每段 TXT 记录中的通道数；同一次运行混合 4/8 通道时，`eeg_segments.npy` 会将 4 通道段补零到 8 通道，以便保存为统一数组。
- `fbcca_scores.npy`：每段在本次配置全局频率轴上的 FBCCA 分数；未参与本段分类的频率列为 `NaN`。
- `fbcca_target_frequencies.npy`：上述全局频率轴，不能固定假定列顺序或频率数量。
- `fbcca_prediction_indexes.npy`：预测频率在 `fbcca_target_frequencies.npy` 全局轴上的索引。分类 CSV 中的 `prediction_index` 则是该段候选列表的局部索引，并另存 `prediction_global_index`。
- `fbcca_prediction_frequencies.npy`：预测出的实际频率，单位 Hz。
- `classification_results.csv`：逐段采样点数、各目标分数和最终预测，便于直接检查。
- `eeg_trials.npy`、`eeg_labels.npy`、`*.mat`：凑齐完整频率 block 后生成的汇总数据。

## 注意事项

- 本项目依赖真实 EEG 串口设备；没有设备时优先使用离线回放测试。
- `.txt` 采集文件、`.npy`、`.mat` 和运行日志通常属于实验数据或中间产物，提交前应确认是否需要纳入版本管理。
- `__pycache__`、临时输出目录和大体积数据文件建议保持忽略，避免仓库过大。
- 视觉刺激频率和 FBCCA 目标频率需要保持一致，否则会影响识别结果。

## 七鑫实时眼动输入

默认使用真实眼动 API。先运行眼动程序，选择七鑫，进入“视线估计”，按原流程完成本次标定，再点击“提供注视”。使用本次标定结果，窗口自动最小化并持续提供坐标：

```powershell
cd D:\documents\slippage\slippage
conda activate slippage
python main.py
```

眼动程序最小化后保持运行。另开终端启动本项目：

```powershell
cd D:\documents\vibe4
conda activate BCI-psychopy
python -s main.py
```

实时输入地址默认 `http://127.0.0.1:8765/gaze`，可用 `--gaze-api-url` 修改。接口返回左上角原点的原始归一化坐标 `u,v`，可能小于 0 或大于 1。本项目将屏幕外的点映射到最近的屏幕边界，再转换到当前窗口像素，用于红十字显示和注视侧闪烁选择；越界不会使本轮失效。API 模式不会读取鼠标覆盖眼动位置。

眼动阶段仍为 **0.5 秒**，期间持续读取有效测量，结束瞬间无效时保留上一个有效点；历史点可以来自开始界面、此前阶段或此前轮次，不限于这 0.5 秒。若从未获取过坐标，初始点为 `(0,0)`，默认 gaze 模式因此闪烁左侧。坐标缺失不会阻塞实验或延长阶段。超过 200 ms、无效或丢失的数据不作为新的测量点，但不会清除已保存的历史点。API 每次返回数据年龄，接收端还计入请求耗时。此检查不证明人眼已看准目标，也不能保证 0.5 秒内定位成功；需用佩戴后的实际目标切换测量速度及定位误差。注视模型质量不足时，API 会保留该标记，接口接通不代表精度达标。

“实验开始”始终可用，未点击“获取注视”不会自动连接眼动服务；已点击时保留接收线程，开始后连接成功也会更新坐标。开始界面和实验画面均显示红十字：初始点为 `(0,0)`，有有效坐标后实时更新，无效或断流时保留最后位置。需要单独调试旧鼠标模拟时，可显式使用 `--gaze-source mouse`。

使用目标区域控制闪烁：

```powershell
python -s main.py --nogaze
```

`--nogaze` 仅改变闪烁侧选择，仍可点击“获取注视”，接收 API 并显示红十字。例如目标在右侧，即使注视点为 `(0,0)` 或处于左侧，也闪烁右侧；不加参数则按注视点判侧。可与 `--four`、`--ar-green` 组合使用。

结果 CSV 保留 `gaze_x,gaze_y,gaze_side`，新增 `flicker_side`、`region_source`（`gaze`/`target`）和 `gaze_point_source`（`measurement`/`history`/`default`），分别记录实际闪烁侧、判侧方式和坐标来源。分类目标反查、正确率和结果显示使用实际闪烁侧，避免将目标侧误记成测量注视侧。初始 `(0,0)` 是实验的默认点，不伪装成 API 测量帧。


## 三分区目标样式与删除目标

`object.json` 的 `region_layout` 为 `three_sectors`：从屏幕中心连向顶边中点和两个下角，将屏幕划为左上（`left`）、右上（`right`）、下方（`bottom`）。下方区域在线边界上优先，中心以上的竖线边界归右侧；不再仅按横坐标中线判断三区。

默认每区三个目标的中心均匀排列：左上编号 0–2，右上 3–5，下方 6–8。2、4、6 为暗红等边三角形，分别是 10、12、15 Hz，颜色 `#8B0000`，即 RGB `(139,0,0)`；其余为黑色正方形。几何图形在原包围框内居中，三角形保持等边比例，不拉伸为等腰三角形。

配置字段：`shape` 支持 `square`、`triangle`，并兼容原 `rectangle`；`color` 使用 `#RRGGBB`；`label` 为显示及结果使用的稳定编号。三角形从暗红色平滑变化到白色，方形保持黑白闪烁，频率及时序公式不变。目标数字只在非闪烁阶段显示。显式 `--ar-green` 保留原绿色覆盖模式，目标仍保持各自形状。

可以删除任意目标条目，保留剩余条目的 `name`、`label` 和 `frequency_hz`，保存后重启程序生效；不要求每区恰好三个，也不要求编号连续。每个区域内频率必须唯一，目标名称和显示编号也必须唯一。若删空某个区域，仍保留三条分界线；gaze 指向该空区时跳过闪烁，不回退到其他区域分类。至少需要一个目标。

例如右上区域删除 12 Hz 目标后，该区域只以剩余 15、10 Hz 作为 FBCCA 候选，即使其他区域还存在 12 Hz，也不会将其作为本段候选。候选集合按每段已登记的实际闪烁区域读取，延迟返回的分类不会误用下一轮区域。`--nogaze` 仍按目标区域闪烁，并使用该实际区域的候选集合。

删除中间目标后，提示文字及结果中的 `actual_index`、`predicted_target_index` 优先使用稳定的 `label`；没有 `label` 的旧配置沿用数组索引。新增 `actual_config_index`、`predicted_config_index` 保存内部数组位置。结果仍按“实际区域 + 预测频率”反查配置中存在的目标，不用局部频率索引直接当目标编号。

`classification_results.csv` 每段记录自己的 `target_frequencies_hz` 和候选分数；候选数可为 1、2、3 或更多，不会因分数长度不同而堆叠失败。`fbcca_scores.npy` 用全局频率轴对齐，未计算列为 `NaN`，读取时须结合 `fbcca_target_frequencies.npy`。
