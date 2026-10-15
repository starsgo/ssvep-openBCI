"""Run the real-time SSVEP experiment after connecting gaze and pressing Start."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from analyse.prediction_results import PredictionResultRecorder
from analyse.test_four import BlinkSegmentProcessor, PredictionResult
from experiment import (
    TargetSpec,
    generate_target_indexes,
    load_targets,
    unique_target_frequencies,
)
from gaze import GazeTracker, Region
from gaze.api import ApiGazeTracker
from ui.start_screen import show_start_screen
from mySerial import SerialReader
from ui.display_psy import (
    ObjAdder,
    VisualObj,
    build_visual_objects,
    display,
    format_prediction_text,
    partition_visual_objects,
)


COMMAND_BLINK = "blink"
COMMAND_PAUSE = "pause"
WINDOW_SIZE = (1920, 1080)
CONFIG_FILE = Path(__file__).with_name("object.json")


def sleep_precise(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while True:
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.001:
            time.sleep(remaining - 0.0005)


class TimerThread(threading.Thread):
    """Drive the fixed rest, prompt, gaze, and flicker experiment phases."""

    def __init__(
        self,
        adder: ObjAdder,
        serial: SerialReader,
        gaze_tracker: GazeTracker,
        targets: list[TargetSpec],
        visual_objects: list[VisualObj],
        target_indexes: list[int],
        *,
        render_ready: threading.Event | None = None,
        blink_duration: float = 1.5,
        pause_duration: float = 2.5,
        cue_duration: float = 0.5,
        gaze_duration: float = 0.5,
        prediction_results_file: str | Path | None = None,
    ) -> None:
        super().__init__(daemon=True)
        if len(targets) != len(visual_objects):
            raise ValueError("target specs and visual objects must have the same length")
        if not target_indexes:
            raise ValueError("at least one target index is required")
        self.adder = adder
        self.serial = serial
        self.gaze_tracker = gaze_tracker
        self.targets = targets
        self.visual_objects = visual_objects
        self.target_indexes = target_indexes
        self.render_ready = render_ready
        self.blink_duration = float(blink_duration)
        self.pause_duration = float(pause_duration)
        self.cue_duration = float(cue_duration)
        self.gaze_duration = float(gaze_duration)
        self.results = PredictionResultRecorder(targets, prediction_results_file)
        self._stop_event = threading.Event()
        self._prediction_lock = threading.Lock()
        self._predictions: dict[int, PredictionResult] = {}
        self._active_rest_segment: int | None = None

    @property
    def running(self) -> bool:
        return not self._stop_event.is_set()

    def _wait(self, seconds: float) -> bool:
        return not self._stop_event.wait(max(0.0, seconds))

    def _collect_gaze(self) -> tuple[int, int] | None:
        """Keep the latest measured point seen during this target's gaze phase."""
        deadline = time.monotonic() + max(0.0, self.gaze_duration)
        position = None
        while self.running:
            current = self.gaze_tracker.get_current_position()
            if current is not None:
                position = current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return position
            if not self._wait(min(0.01, remaining)):
                break
        return None

    def _register_trial_metadata(
        self,
        segment_index: int,
        actual_index: int,
        gaze_position: tuple[int, int],
        gaze_side: Region,
    ) -> None:
        with self._prediction_lock:
            self.results.register(
                segment_index, actual_index, gaze_position, gaze_side
            )
            prediction = self._predictions.get(segment_index)
            if prediction is not None:
                self.results.record(prediction)

    def on_prediction(self, prediction: PredictionResult) -> None:
        """Receive classifier results without calling PsychoPy from this thread."""
        with self._prediction_lock:
            self._predictions[prediction.segment_index] = prediction
            metadata = self.results.metadata_for(prediction.segment_index)
            self.results.record(prediction)
            if self._active_rest_segment == prediction.segment_index and self.running:
                self.adder.set_status_text.emit(
                    format_prediction_text(
                        prediction.prediction_frequency,
                        None if metadata is None else metadata.gaze_side,
                    )
                )

    def _begin_rest(self, segment_index: int | None) -> None:
        self.adder.set_static_targets.emit(self.visual_objects)
        self.adder.delete_visual_point.emit()
        self.adder.set_prompt_text.emit(None)
        with self._prediction_lock:
            self._active_rest_segment = segment_index
            prediction = self._predictions.get(segment_index) if segment_index else None
            if segment_index is None:
                self.adder.set_status_text.emit(None)
            elif prediction is None:
                self.adder.set_status_text.emit("predit:processing...")
            else:
                metadata = self.results.metadata_for(segment_index)
                self.adder.set_status_text.emit(
                    format_prediction_text(
                        prediction.prediction_frequency,
                        None if metadata is None else metadata.gaze_side,
                    )
                )

    def _end_rest(self) -> None:
        with self._prediction_lock:
            self._active_rest_segment = None
            self.adder.set_status_text.emit(None)

    def run(self) -> None:
        while self.running and self.render_ready is not None and not self.render_ready.wait(0.05):
            pass
        if not self.running:
            return
        while self.running and not self.gaze_tracker.ready_event.wait(0.05):
            pass
        if not self.running:
            return
        trial_count = 0
        stopped_scene_shown = False
        while self.running:
            if self.serial.status == "s":
                if not stopped_scene_shown:
                    self.adder.set_static_targets.emit(self.visual_objects)
                    self.adder.delete_visual_point.emit()
                    self.adder.set_status_text.emit(None)
                    self.adder.set_prompt_text.emit(None)
                    stopped_scene_shown = True
                if not self._wait(0.1):
                    break
                continue
            stopped_scene_shown = False

            # Rest: all targets black; asynchronously show the preceding result.
            preceding_segment = trial_count if trial_count > 0 else None
            self._begin_rest(preceding_segment)
            if not self._wait(self.pause_duration):
                break
            self._end_rest()

            # Prompt: keep all targets black and show the requested object INDEX.
            target_index = self.target_indexes[trial_count % len(self.target_indexes)]
            self.adder.set_static_targets.emit(self.visual_objects)
            self.adder.set_prompt_text.emit(f"注视目标{target_index}")
            if not self._wait(self.cue_duration):
                break

            # Gaze: continuously collect measurements during the 0.5 s phase.
            cue_x, cue_y = self.targets[target_index].center
            self.adder.set_prompt_text.emit(None)
            self.adder.add_visual_point.emit(cue_x, cue_y)
            position = self._collect_gaze()
            if not self.running:
                break
            if position is None:
                # Keep this target visible until a measured coordinate arrives;
                # do not restart the rest/cue phases on every missing sample.
                reason = getattr(self.gaze_tracker, 'last_error', None) or '尚无新的注视坐标'
                self.adder.set_status_text.emit('等待眼动坐标')
                print(f'[gaze] 等待坐标，目标 {target_index}: {reason}', flush=True)
                while self.running and self.serial.status != 's' and position is None:
                    if not self._wait(0.01):
                        break
                    position = self.gaze_tracker.get_current_position()
                if not self.running:
                    break
            if position is None or self.serial.status == 's':
                continue

            region = self.gaze_tracker.region_for(position)
            segment_index = trial_count + 1
            self._register_trial_metadata(
                segment_index, target_index, position, region
            )
            active_objects, inactive_objects = partition_visual_objects(
                self.targets, self.visual_objects, region
            )
            self.adder.delete_visual_point.emit()
            self.adder.set_status_text.emit(None)

            # The renderer sends markers on actual flips, not worker-thread time.
            trial_done = threading.Event()
            self.adder.start_trial.emit(
                active_objects,
                self.blink_duration,
                lambda: self.serial.send_event(COMMAND_BLINK),
                lambda: self.serial.send_event(COMMAND_PAUSE),
                trial_done,
                inactive_objects,
            )
            while self.running and not trial_done.wait(0.05):
                pass
            if not self.running:
                break
            trial_count += 1

    def stop(self) -> None:
        self._stop_event.set()


class InputThread(threading.Thread):
    def __init__(self, serial_reader: SerialReader, on_exit=None) -> None:
        super().__init__(daemon=True)
        self.serial_reader = serial_reader
        self.on_exit = on_exit
        self.running = True

    def run(self) -> None:
        try:
            sleep_precise(1.0)
            self.serial_reader.send_command("b\r\n")
            while self.running:
                user_input = input("Command: ")
                if user_input.lower() == "exit":
                    break
                self.serial_reader.send_command(user_input + "\r\n")
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            self.serial_reader.stop()
            if self.on_exit is not None:
                self.on_exit()
            print("Serial reader stopped.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SSVEP experiment")
    parser.add_argument(
        "--four",
        action="store_true",
        help="classify using the first four EEG channels instead of all eight",
    )
    parser.add_argument(
        "--ar-green",
        action="store_true",
        help="flicker rectangle targets using only the green channel",
    )
    parser.add_argument("--gaze-source", choices=("mouse", "api"), default="api",
                        help="use mouse simulation or measured coordinates from the eye API")
    parser.add_argument("--gaze-api-url", default="http://127.0.0.1:8765/gaze")
    args = parser.parse_args(argv)
    selection = show_start_screen(args.gaze_source, args.gaze_api_url)
    if selection is None:
        return 0
    args.gaze_source = selection.source
    args.connected_gaze_tracker = selection.tracker
    try:
        return run_experiment(args)
    finally:
        if selection.tracker is not None:
            selection.tracker.stop()
            selection.tracker.join(timeout=1.)


def run_experiment(args) -> int:
    """Open devices and run the original flow only after the start button."""
    channel_count = 4 if args.four else 8

    targets = load_targets(CONFIG_FILE, window_size=WINDOW_SIZE)
    target_indexes = generate_target_indexes(len(targets), count=40)
    # Left and right screen targets may reuse the same frequencies. The UI keeps
    # every object, while FBCCA only needs one score per distinct frequency.
    target_frequencies = unique_target_frequencies(targets)
    print(f"INDEX: {target_indexes}")
    print(f"Classifier channels: {channel_count}")
    print(f"Target color mode: {'green' if args.ar_green else 'gray'}")

    output_file = "test_40_ssvep.txt"
    serial_reader = SerialReader("COM3", 115200, output_file)
    serial_reader.start()
    Path(f"labels_{output_file}").write_text(
        "".join(f"{index}\n" for index in target_indexes), encoding="utf-8"
    )

    window = display(
        background=0.0,
        size=WINDOW_SIZE,
        fullscr=True,
        screen=0,
    )
    adder = ObjAdder()
    adder.add_obj.connect(window.add_visual_obj)
    adder.remove_all_obj.connect(window.clear_visual_objs)
    adder.add_visual_point.connect(window.add_visual_point)
    adder.delete_visual_point.connect(window.delete_visual_point)
    adder.set_static_targets.connect(window.set_static_targets)
    adder.set_status_text.connect(window.set_status_text)
    adder.set_prompt_text.connect(window.set_prompt_text)
    adder.start_trial.connect(window.start_visual_trial)
    adder.close_requested.connect(window.request_close)

    visual_objects = build_visual_objects(
        targets, color_mode="green" if args.ar_green else "gray"
    )
    # Show the rest scene immediately, including during display warm-up. This
    # makes startup visibly distinct from a stalled, empty gray window.
    window.set_static_targets(visual_objects)
    if args.gaze_source == "api":
        gaze_tracker = getattr(args, "connected_gaze_tracker", None) or ApiGazeTracker(
            args.gaze_api_url, window_size=WINDOW_SIZE)
        window.set_gaze_callback(None)
        window.set_gaze_provider(gaze_tracker.get_current_position)
        window.set_status_text("实时眼动已连接")
        print(f"Gaze input: {args.gaze_api_url}; gaze phase remains 0.5 seconds")
    else:
        gaze_tracker = GazeTracker(window_size=WINDOW_SIZE)
        window.set_gaze_callback(gaze_tracker.update_position)
    if not gaze_tracker.is_alive():
        gaze_tracker.start()
    timer = TimerThread(
        adder,
        serial_reader,
        gaze_tracker,
        targets,
        visual_objects,
        target_indexes,
        render_ready=window.ready_event,
        prediction_results_file=window.record_dir / "prediction_results.csv",
    )
    segment_processor = BlinkSegmentProcessor(
        output_file,
        target_frequencies=target_frequencies,
        on_prediction=timer.on_prediction,
        channel_count=channel_count,
    )
    segment_processor.start()
    input_thread = InputThread(serial_reader, on_exit=adder.close_requested.emit)
    input_thread.start()
    timer.start()

    try:
        window.setWindowTitle("PsychoPy SSVEP display")
        window.show()
        return window.run(adder)
    finally:
        print("GUI closed, stopping threads...")
        timer.stop()
        timer.join(timeout=2.0)
        gaze_tracker.stop()
        gaze_tracker.join(timeout=1.0)
        input_thread.running = False
        input_thread.join(timeout=1.0)
        serial_reader.stop()
        segment_processor.stop()
        segment_processor.join(timeout=2.0)
        record_dir = window.close()
        print(f"Timing records saved to: {record_dir.resolve()}")
        print("All stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
