# 数据与结果说明

本文档说明项目中常见数据文件的来源、形状和用途。

## 原始 TXT

串口线程将每个 EEG 数据帧写入 TXT，格式大致如下：

```text
[时间戳] {'fn': 帧编号, 'eeg': [通道1, 通道2, ..., 通道8]} [MARKER] blink
```

标记含义：

- `[MARKER] blink`：视觉刺激开始，实时分析线程开始收集当前区间。
- `[MARKER] pause`：视觉刺激结束，实时分析线程停止收集并处理当前区间。

实时处理只使用 `blink -> pause` 之间的 EEG 行。

## 预处理结果

`eeg_segments.npy`

- 形状：`(区间数, 8通道, 统一采样点数)`
- 来源：每个完整 blink 区间最多取前 375 点，独立执行带通滤波、50 Hz 陷波、去均值和标准差归一化；在线路径不执行 PCA。
- 用途：保存全部已处理区间，便于后续复查或重新分析。

不同区间采样点数可能略有差异，保存汇总文件时会统一截取到当前最短长度。

## FBCCA 结果

`fbcca_scores.npy`

- 形状：`(区间数, 目标数)`
- 每一行对应一个 blink 区间。
- 各列按 `object.json` 中频率首次出现的顺序排列；左右目标复用频率时只生成一列，默认是 `[15, 12, 10] Hz`。

`fbcca_prediction_indexes.npy`

- 形状：`(区间数,)`
- 每项为预测频率在去重频率列表中的索引，不包含左右边信息。

`fbcca_prediction_frequencies.npy`

- 形状：`(区间数,)`
- 每项为最终预测频率，单位 Hz。

`classification_results.csv`

- 每行保存段编号、实际使用的采样点数、目标频率、全部 FBCCA 分数和最终预测。
- 适合不加载 NumPy 文件时直接检查在线分类结果。

`record/<运行时间>/prediction_results.csv`

- 每行保存实际 INDEX、目标名称和频率、眼动结束位置、左右边、FBCCA 预测频率，以及由“预测频率＋左右边”反查出的目标 INDEX。
- 同时保存实际目标在 `object.json` 中的位置、大小和所属边，以及眼动左右边是否正确。
- `correct` 表示反查出的目标 INDEX 是否与当轮实际 INDEX 相同。
- `object.json` 的数组位置就是 0-based INDEX，默认目标名称为 `visualObj[0]` 到 `visualObj[5]`。

`record/<运行时间>/prediction_accuracy.json`

- 随每次预测更新，保存累计预测数、正确数、错误数和正确率。

## 离线 notebook 数据

`eeg_trials.npy`、`eeg_labels.npy` 和 MAT 文件属于离线 notebook 工作流，在线处理器不再假设随机 INDEX 能组成固定频率 block，也不会生成这些汇总文件。

## 输出目录

正式运行：

```text
analyse/realtime_results/
```

离线测试：

```text
analyse/realtime_test_segments/<运行时间>/
```

离线测试额外读取配套的 `labels_*.txt` 并生成 `accuracy_summary.txt`，用于逐段核对期望目标和预测结果。
