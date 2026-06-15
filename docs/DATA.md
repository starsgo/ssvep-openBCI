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

- 形状：`(区间数, PCA分量数, 统一采样点数)`
- 来源：每个完整 blink 区间独立执行带通滤波、50 Hz 陷波、去均值、归一化和 PCA。
- 用途：保存全部已处理区间，便于后续复查或重新分析。

不同区间采样点数可能略有差异，保存汇总文件时会统一截取到当前最短长度。

## FBCCA 结果

`fbcca_scores.npy`

- 形状：`(区间数, 4)`
- 每一行对应一个 blink 区间。
- 四列顺序与 `analyse/test_four.py` 中的 `TARGET_FREQS` 一致，当前为 `[15, 13, 11, 9] Hz`。

`fbcca_prediction_indexes.npy`

- 形状：`(区间数,)`
- 每项为预测结果在 `TARGET_FREQS` 中的索引。

`fbcca_prediction_frequencies.npy`

- 形状：`(区间数,)`
- 每项为最终预测频率，单位 Hz。

## Trial 汇总

`eeg_trials.npy`

- 形状：`(trial数, 频率数, 通道数, 采样点数)`
- 当前每 4 个区间组成一个完整 trial。

`eeg_labels.npy`

- 形状：`(trial数, 频率数)`
- 标签按 `TARGET_FREQS` 的顺序生成。

`eeg_data_<txt文件名>.mat`

- MATLAB 格式汇总文件。
- 由 `scipy.io.savemat` 生成，便于后续在 MATLAB 或 notebook 中分析。

## 输出目录

正式运行：

```text
analyse/realtime_results/
```

离线测试：

```text
analyse/realtime_test_segments/<运行时间>/
```

离线测试额外生成 `accuracy_summary.txt`，用于检查每个区间预测是否符合预期顺序。
