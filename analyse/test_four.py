import ast
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.io import savemat
from scipy.signal import butter, cheb1ord, cheby1, filtfilt, iirnotch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

NUM_CHANNELS = 8
SAMPLE_RATE = 250
NUM_FREQS = 4
TARGET_FREQS = np.array([15, 12, 10, 8.57])
FBCCA_NUM_FILTER_BANKS = 5
FBCCA_NUM_HARMONICS = 3
# FBCCA 不使用 blink 刚开始的 0.2 秒，随后取 1.5 秒稳定数据进行识别。
FBCCA_DELAY_SAMPLES = round(0.3 * SAMPLE_RATE)
FBCCA_GAZE_SAMPLES = round(1.5 * SAMPLE_RATE)


def parse_eeg_line(line: str) -> Optional[np.ndarray]:
    """Extract the eight EEG channel values from one SerialReader output line."""
    start = line.find("{")
    end = line.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        frame = ast.literal_eval(line[start : end + 1])
        eeg = np.asarray(frame["eeg"], dtype=float)
    except (KeyError, TypeError, ValueError, SyntaxError):
        return None

    if eeg.shape != (NUM_CHANNELS,):
        return None
    return eeg


def process_blink_segment(samples: list[np.ndarray]) -> np.ndarray:
    """单独处理一个 blink -> pause 区间，不与其他区间混合。"""
    eeg = np.asarray(samples, dtype=float).T
    if eeg.shape[0] != NUM_CHANNELS:
        raise ValueError(f"expected {NUM_CHANNELS} channels, got shape {eeg.shape}")

    band_b, band_a = butter(
        4, [5 / (SAMPLE_RATE / 2), 90 / (SAMPLE_RATE / 2)], btype="band"
    )
    notch_b, notch_a = iirnotch(50.0, 30.0, SAMPLE_RATE)
    minimum_samples = 3 * max(len(band_a), len(band_b))
    if eeg.shape[1] <= minimum_samples:
        raise ValueError(
            f"segment is too short: got {eeg.shape[1]} samples, "
            f"need more than {minimum_samples}"
        )

    # 以下步骤来自 test_four.ipynb：带通滤波、50Hz 陷波、去均值、归一化。
    eeg = filtfilt(band_b, band_a, eeg, axis=1)
    eeg = filtfilt(notch_b, notch_a, eeg, axis=1)
    eeg -= np.mean(eeg, axis=1, keepdims=True)
    variances = np.var(eeg, axis=1, keepdims=True)
    eeg = np.divide(eeg, variances, out=np.zeros_like(eeg), where=variances != 0)
    eeg *= 1000

    # PCA 也只拟合当前 blink -> pause 区间。
    pca = PCA(n_components=min(NUM_CHANNELS, eeg.shape[1]))
    return pca.fit_transform(eeg.T).T.astype(np.float32)


def filterbank(eeg: np.ndarray, index: int) -> np.ndarray:
    """对一个区间应用一组 FBCCA 滤波器。输入形状为：通道数 x 采样点数。"""
    passband = np.array([6, 14, 22, 30, 38, 46, 54, 62, 70, 78])
    stopband = np.array([4, 10, 16, 24, 32, 40, 48, 56, 64, 72])
    if index < 1 or index > len(passband):
        raise ValueError(f"filter bank index must be 1-{len(passband)}, got {index}")

    nyquist = SAMPLE_RATE / 2
    wp = [passband[index - 1] / nyquist, 90 / nyquist]
    ws = [stopband[index - 1] / nyquist, 100 / nyquist]
    order, wn = cheb1ord(wp, ws, 3, 40)
    b, a = cheby1(order, 0.5, wn, btype="bandpass")
    return filtfilt(b, a, eeg, axis=1)


def process_fbcca_segment(processed: np.ndarray) -> tuple[np.ndarray, int, float]:
    """对一个已预处理区间独立执行 FBCCA，返回四个分数、预测索引和预测频率。"""
    start = FBCCA_DELAY_SAMPLES
    end = start + FBCCA_GAZE_SAMPLES
    if processed.shape[1] < end:
        raise ValueError(
            f"FBCCA segment is too short: got {processed.shape[1]} samples, "
            f"need at least {end}"
        )

    eeg = processed[:, start:end]
    timepoints = np.arange(eeg.shape[1]) / SAMPLE_RATE
    coefficients = np.array(
        [
            index ** (-1.25) + 0.25
            for index in range(1, FBCCA_NUM_FILTER_BANKS + 1)
        ]
    )
    filtered = [
        filterbank(eeg, index)
        for index in range(1, FBCCA_NUM_FILTER_BANKS + 1)
    ]
    scores = []

    # 每个目标频率分别计算一个 FBCCA 分数，顺序固定为 [15, 11, 13, 9] Hz。
    for frequency in TARGET_FREQS:
        references = []
        for harmonic in range(1, FBCCA_NUM_HARMONICS + 1):
            references.append(np.sin(2 * np.pi * harmonic * frequency * timepoints))
            references.append(np.cos(2 * np.pi * harmonic * frequency * timepoints))
        reference = np.asarray(references).T
        correlations = []

        for eeg_filterbank in filtered:
            good_channels = np.std(eeg_filterbank, axis=1) > 1e-8
            eeg_good = eeg_filterbank[good_channels]
            if eeg_good.shape[0] < 1:
                correlations.append(0.0)
                continue
            cca = CCA(n_components=1)
            cca.fit(eeg_good.T, reference)
            eeg_component, reference_component = cca.transform(eeg_good.T, reference)
            correlation = np.corrcoef(
                eeg_component[:, 0], reference_component[:, 0]
            )[0, 1]
            correlations.append(0.0 if np.isnan(correlation) else correlation)

        scores.append(np.sum(coefficients * np.square(correlations)))

    scores_array = np.asarray(scores, dtype=np.float32)
    # 分数最高的目标频率就是当前 blink -> pause 区间的识别结果。
    prediction_index = int(np.argmax(scores_array))
    prediction_frequency = float(TARGET_FREQS[prediction_index])
    return scores_array, prediction_index, prediction_frequency


class BlinkSegmentProcessor(threading.Thread):
    """实时读取 TXT；每遇到一个完整 blink -> pause 区间，立即独立处理一次。"""

    def __init__(
        self,
        txt_file: str | Path,
        on_segment: Optional[Callable[[np.ndarray], None]] = None,
        poll_interval: float = 0.05,
        start_at_end: bool = True,
        output_dir: str | Path | None = None,
    ):
        super().__init__(daemon=True)
        self.txt_file = Path(txt_file)
        self.on_segment = on_segment or self._print_segment
        self.poll_interval = poll_interval
        self.start_at_end = start_at_end
        # 所有 blink -> pause 区间的处理结果集中保存在同一个目录。
        # 不为每个区间单独生成文件。默认目录为：
        # analyse/realtime_results/
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(__file__).parent / "realtime_results"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 所有中间结果写入日志。终端只输出当前区间最终识别出的频率。
        self.log_file = self.output_dir / "runtime.log"
        self.running = True
        self._collecting = False
        self._samples: list[np.ndarray] = []
        self._processed_segments: list[np.ndarray] = []
        self._fbcca_scores: list[np.ndarray] = []
        self._fbcca_prediction_indexes: list[int] = []
        self._fbcca_prediction_frequencies: list[float] = []

    def run(self) -> None:
        while self.running and not self.txt_file.exists():
            time.sleep(self.poll_interval)
        if not self.running:
            return

        with self.txt_file.open("r", encoding="utf-8") as txt:
            if self.start_at_end:
                txt.seek(0, 2)

            while self.running:
                line = txt.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    continue
                self._consume_line(line)

    def stop(self) -> None:
        self.running = False

    def _consume_line(self, line: str) -> None:
        # blink 标记：清空上一次缓存，开始收集一个新的独立数据区间。
        if "[MARKER] blink" in line:
            self._collecting = True
            self._samples = []
        # pause 标记：当前区间结束，立即触发预处理和 FBCCA。
        elif "[MARKER] pause" in line:
            if self._collecting:
                self._process_current_segment()
            self._collecting = False
            self._samples = []
            return

        # 只收集 blink 和 pause 标记之间的 EEG 行。
        if self._collecting:
            eeg = parse_eeg_line(line)
            if eeg is not None:
                self._samples.append(eeg)

    def _process_current_segment(self) -> None:
        try:
            processed = process_blink_segment(self._samples)
        except ValueError as error:
            self.log(f"Skipped blink segment: {error}")
            return

        # 单段处理边界：每个 blink -> pause 区间在这里独立完成处理。
        # processed 是当前单段经过带通滤波、50Hz 陷波、归一化和 PCA
        # 后的数据，形状为 (PCA 分量数, 当前区间采样点数)。
        #
        # FBCCA 也在这里对当前单段独立处理：
        # 1. 截取延迟 0.2 秒后的 1.5 秒数据。
        # 2. 分别计算 15、11、13、9 Hz 的 FBCCA 分数。
        # 3. 保存当前单段的四个分数和最高分对应的预测频率。
        try:
            scores, prediction_index, prediction_frequency = process_fbcca_segment(
                processed
            )
        except ValueError as error:
            self.log(f"Skipped FBCCA for blink segment: {error}")
            return

        self._processed_segments.append(processed)
        self._fbcca_scores.append(scores)
        self._fbcca_prediction_indexes.append(prediction_index)
        self._fbcca_prediction_frequencies.append(prediction_frequency)
        self._save_results()
        # 终端运行时只输出最终预测结果，例如：15 Hz
        print(f"{prediction_frequency:g} Hz")
        self.on_segment(processed)

    def _save_results(self) -> None:
        # 汇总保存边界：磁盘上只更新汇总文件，不为每个区间单独建文件。
        # 各区间采样点数量可能有轻微差异，因此统一截取到当前最短长度。
        min_timepoints = min(segment.shape[1] for segment in self._processed_segments)
        eeg_segments = np.stack(
            [segment[:, :min_timepoints] for segment in self._processed_segments]
        )
        segments_file = self.output_dir / "eeg_segments.npy"
        np.save(segments_file, eeg_segments)

        # FBCCA 汇总文件：每一行都对应一个独立的 blink -> pause 区间。
        # fbcca_scores.npy:
        #   形状为 (已处理区间数, 4)。
        #   每行依次保存当前区间对 [15, 11, 13, 9] Hz 的四个分数。
        # fbcca_prediction_indexes.npy:
        #   形状为 (已处理区间数,)。
        #   每项保存当前区间预测结果在 TARGET_FREQS 中的索引。
        # fbcca_prediction_frequencies.npy:
        #   形状为 (已处理区间数,)。
        #   每项保存当前区间最终预测出的实际频率，单位 Hz。
        scores_file = self.output_dir / "fbcca_scores.npy"
        indexes_file = self.output_dir / "fbcca_prediction_indexes.npy"
        frequencies_file = self.output_dir / "fbcca_prediction_frequencies.npy"
        np.save(scores_file, np.asarray(self._fbcca_scores))
        np.save(indexes_file, np.asarray(self._fbcca_prediction_indexes))
        np.save(frequencies_file, np.asarray(self._fbcca_prediction_frequencies))

        # eeg_segments.npy 保存全部已处理区间，形状为：
        # (已完成区间数, PCA 分量数, 统一后的采样点数)。
        # notebook 使用每 4 个频率组成一个 block。只有凑齐完整 block 后，
        # 才更新 eeg_trials.npy、eeg_labels.npy 和 MAT 汇总文件。
        complete_segments = len(eeg_segments) // NUM_FREQS * NUM_FREQS
        if complete_segments:
            eeg_trials = eeg_segments[:complete_segments].reshape(
                -1, NUM_FREQS, NUM_CHANNELS, min_timepoints
            )
            eeg_labels = np.tile(np.arange(NUM_FREQS), (len(eeg_trials), 1))
            np.save(self.output_dir / "eeg_trials.npy", eeg_trials)
            np.save(self.output_dir / "eeg_labels.npy", eeg_labels)
            savemat(
                self.output_dir / f"eeg_data_{self.txt_file.stem}.mat",
                {"eeg": eeg_trials.transpose(1, 2, 3, 0)},
            )

        self.log(
            f"Processed blink segment {len(self._processed_segments)}: "
            f"shape={self._processed_segments[-1].shape}; "
            f"FBCCA scores={self._fbcca_scores[-1]}, "
            f"prediction={self._fbcca_prediction_frequencies[-1]} Hz; "
            f"updated aggregate file={segments_file}"
        )

    def log(self, message: str) -> None:
        """将中间信息写入 runtime.log，不在终端输出。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.log_file.open("a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] {message}\n")

    @staticmethod
    def _print_segment(processed: np.ndarray) -> None:
        # 默认回调不输出内容，避免终端出现预测频率以外的中间信息。
        return
