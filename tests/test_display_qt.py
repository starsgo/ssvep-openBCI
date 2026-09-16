import tempfile
import unittest
from pathlib import Path

from ui.display import VisualObj


class QtBlinkLogTests(unittest.TestCase):
    def test_blink_log_is_written_inside_timestamped_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260914_153000"
            obj = VisualObj(
                (0, 0),
                (10, 10),
                15.0,
                name="visualObj[1]",
                log_blink=True,
            )

            obj.set_record_dir(run_dir)
            obj.update_blink(obj.start + 0.55 / obj.blink_frequency)

            log_path = run_dir / "visualObj[1]_log.txt"
            self.assertTrue(log_path.exists())
            contents = log_path.read_text(encoding="utf-8")
            self.assertIn("visualObj[1] initialized", contents)
            self.assertIn("visualObj[1] blink", contents)

    def test_reattaching_same_run_does_not_duplicate_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260914_153000"
            obj = VisualObj((0, 0), (10, 10), log_blink=True)

            obj.set_record_dir(run_dir)
            obj.set_record_dir(run_dir)

            contents = (run_dir / "visualObj_log.txt").read_text(encoding="utf-8")
            self.assertEqual(contents.count("initialized"), 1)


if __name__ == "__main__":
    unittest.main()
