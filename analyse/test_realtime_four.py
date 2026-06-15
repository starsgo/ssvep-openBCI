import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from analyse.test_four import BlinkSegmentProcessor, TARGET_FREQS

# ============================================================
# 测试输入文件
# ============================================================
# 本脚本只用于离线测试实时处理线程。
# 它不会连接串口，也不会启动 GUI。
# 线程会直接从项目根目录读取已经存在的 four_61_ssvep.txt。
TXT_FILE = Path("four_73_ssvep.txt")

# 测试结果集中保存在 analyse/realtime_test_segments/<运行时间>/。
# 不会为每个 blink -> pause 区间单独创建文件。
OUTPUT_ROOT = Path(__file__).parent / "realtime_test_segments"

# 最长等待时间。现成 TXT 文件读取完成后，线程应该在此时间内处理完毕。
TIMEOUT_SECONDS = 120.0


def count_complete_segments(txt_file: Path) -> int:
    """统计 TXT 文件中完整的 blink -> pause 区间数量。"""
    collecting = False
    segment_count = 0

    with txt_file.open("r", encoding="utf-8") as txt:
        for line in txt:
            # 遇到 blink：开始统计一个新的数据区间。
            if "[MARKER] blink" in line:
                collecting = True
            # 遇到 pause：当前 blink -> pause 区间完整结束。
            elif "[MARKER] pause" in line and collecting:
                segment_count += 1
                collecting = False

    return segment_count


def main() -> None:
    if not TXT_FILE.is_file():
        raise FileNotFoundError(f"TXT file does not exist: {TXT_FILE.resolve()}")

    # 先统计现成文件中应该处理多少个完整区间。
    # 后续用这个数量判断实时处理线程是否正确处理完全部数据。
    expected_segments = count_complete_segments(TXT_FILE)
    if expected_segments == 0:
        raise RuntimeError("No complete blink -> pause segment found.")

    # 每次测试使用单独的时间戳目录，避免覆盖上一次测试结果。
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_id

    processed_segments = []
    all_completed = threading.Event()

    def on_segment(processed: np.ndarray) -> None:
        """每完成一个独立 blink -> pause 区间后，由处理线程调用一次。"""
        processed_segments.append(processed)

        # 所有完整区间都处理完后，通知主测试线程结束等待。
        if len(processed_segments) >= expected_segments:
            all_completed.set()

    # ============================================================
    # 只启动 BlinkSegmentProcessor 线程
    # ============================================================
    # start_at_end=False：
    #   正式运行 main.py 时，线程只读取 TXT 后续新增的数据。
    #   本测试读取现成 TXT，所以必须从文件开头开始回放。
    processor = BlinkSegmentProcessor(
        TXT_FILE,
        on_segment=on_segment,
        start_at_end=False,
        output_dir=output_dir,
    )
    processor.log(f"Replay TXT: {TXT_FILE.resolve()}")
    processor.log(f"Expected complete blink -> pause segments: {expected_segments}")
    processor.log(f"Test output directory: {output_dir.resolve()}")
    processor.start()

    finished = all_completed.wait(timeout=TIMEOUT_SECONDS)
    processor.stop()
    processor.join(timeout=2.0)

    if not finished:
        raise TimeoutError(
            f"replay did not finish within {TIMEOUT_SECONDS} seconds: "
            f"processed {len(processed_segments)}/{expected_segments} segments"
        )

    # ============================================================
    # 检查线程集中保存的结果
    # ============================================================
    # eeg_segments.npy：
    #   每一项对应一个独立 blink -> pause 区间经过预处理后的 EEG。
    # fbcca_scores.npy：
    #   每一行对应一个独立区间，四列依次为 [15, 11, 13, 9] Hz 的分数。
    # fbcca_prediction_indexes.npy：
    #   每一项是当前区间预测频率在 [15, 11, 13, 9] 中的索引。
    # fbcca_prediction_frequencies.npy：
    #   每一项是当前区间最终预测的实际频率，单位 Hz。
    segments_file = output_dir / "eeg_segments.npy"
    scores_file = output_dir / "fbcca_scores.npy"
    indexes_file = output_dir / "fbcca_prediction_indexes.npy"
    frequencies_file = output_dir / "fbcca_prediction_frequencies.npy"

    eeg_segments = np.load(segments_file)
    fbcca_scores = np.load(scores_file)
    fbcca_indexes = np.load(indexes_file)
    fbcca_frequencies = np.load(frequencies_file)

    if eeg_segments.shape[0] != expected_segments:
        raise RuntimeError("aggregate segment count does not match processed segment count")
    if fbcca_scores.shape != (expected_segments, 4):
        raise RuntimeError("FBCCA score shape does not match processed segment count")
    if fbcca_indexes.shape != (expected_segments,):
        raise RuntimeError("FBCCA prediction index count does not match segment count")
    if fbcca_frequencies.shape != (expected_segments,):
        raise RuntimeError("FBCCA prediction frequency count does not match segment count")

    # ============================================================
    # 生成最终准确率总结文本
    # ============================================================
    # 测试文件中的目标顺序按 [15, 11, 13, 9] Hz 循环。
    # 如果后续实验改变刺激顺序，只需修改 test_four.py 中的 TARGET_FREQS。
    expected_frequencies = np.resize(TARGET_FREQS, expected_segments)
    correct = fbcca_frequencies == expected_frequencies
    accuracy = float(np.mean(correct))
    summary_file = output_dir / "accuracy_summary.txt"
    summary_lines = [
        f"TXT file: {TXT_FILE.resolve()}",
        f"Processed segments: {expected_segments}",
        f"Target frequency order: {TARGET_FREQS.tolist()} Hz",
        f"Correct predictions: {int(np.sum(correct))}/{expected_segments}",
        f"Final accuracy: {accuracy * 100:.2f}%",
        "",
        "segment, expected_hz, predicted_hz, correct",
    ]
    for index, (expected, predicted, is_correct) in enumerate(
        zip(expected_frequencies, fbcca_frequencies, correct), start=1
    ):
        summary_lines.append(
            f"{index}, {expected:g}, {predicted:g}, {'yes' if is_correct else 'no'}"
        )
    summary_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    processor.log(f"Final accuracy: {accuracy * 100:.2f}%")
    processor.log(f"Accuracy summary file: {summary_file.resolve()}")


if __name__ == "__main__":
    main()
