"""Gaze is optional; the display retains the last measured coordinate."""
from contextlib import ExitStack
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main
from ui.start_screen import StartScreen
from gaze import GazePointProvider


class StartScreenTests(unittest.TestCase):
    @patch('main.run_experiment', return_value=0)
    @patch('main.show_start_screen', return_value=None)
    def test_closing_start_screen_does_not_start_experiment(self, screen, run):
        self.assertEqual(main.main([]), 0)
        screen.assert_called_once_with('api', 'http://127.0.0.1:8765/gaze')
        run.assert_not_called()

    @patch('main.run_experiment', return_value=0)
    @patch('main.show_start_screen')
    def test_start_passes_connected_tracker_and_preserves_existing_flags(self, screen, run):
        tracker = Mock()
        screen.return_value = SimpleNamespace(source='api', tracker=tracker)
        self.assertEqual(main.main(['--four', '--ar-green', '--nogaze']), 0)
        args = run.call_args.args[0]
        self.assertTrue(args.four)
        self.assertTrue(args.ar_green)
        self.assertTrue(args.nogaze)
        self.assertEqual(args.gaze_source, 'api')
        self.assertIs(args.connected_gaze_tracker, tracker)
        tracker.stop.assert_called_once()
        tracker.join.assert_called_once_with(timeout=1.)

    def make_screen(self, stack):
        for name in ('Tk', 'Label', 'Canvas', 'Frame', 'Button'):
            stack.enter_context(patch('tkinter.' + name))
        return StartScreen()

    def test_start_without_gaze_uses_default_point_and_does_not_auto_connect(self):
        with ExitStack() as stack:
            factory = stack.enter_context(patch('ui.start_screen.ApiGazeTracker'))
            screen = self.make_screen(stack)
            factory.assert_not_called()
            screen.start()
            self.assertIsNone(screen.selection.tracker)
            self.assertEqual(screen.selection.gaze_point.get_current_position(), (0, 0))
            self.assertFalse(screen.selection.gaze_point.has_measurement)
            screen.root.quit.assert_called_once()
            factory.assert_not_called()

    def test_get_gaze_starts_receiver_and_draws_real_coordinates_then_transfers_connection(self):
        with ExitStack() as stack:
            factory = stack.enter_context(patch('ui.start_screen.ApiGazeTracker'))
            screen = self.make_screen(stack)
            tracker = factory.return_value
            tracker.window_size = (1920, 1080)
            tracker.get_current_position.return_value = (1440, 270)
            screen.fetch_gaze()
            factory.assert_called_once_with('http://127.0.0.1:8765/gaze')
            tracker.start.assert_called_once()
            screen.preview.create_line.reset_mock()
            screen._poll_gaze()
            self.assertEqual(screen.preview.create_line.call_count, 2)
            self.assertEqual(screen.preview.create_line.call_args_list[0].args, (322., 45., 338., 45.))
            screen.start_button.configure.assert_not_called()
            screen.root.mainloop.side_effect = screen.start
            selection = screen.show()
            self.assertIs(selection.tracker, tracker)
            self.assertEqual(selection.source, 'api')
            tracker.stop.assert_not_called()
            screen.root.destroy.assert_called_once()

    def test_invalid_sample_shows_default_point_and_cancel_stops_receiver(self):
        with ExitStack() as stack:
            screen = self.make_screen(stack)
            screen.tracker = Mock()
            screen.tracker.get_current_position.return_value = None
            screen.tracker.api_connected = False
            screen.gaze_point.source = screen.tracker
            screen.preview.create_line.reset_mock()
            screen._poll_gaze()
            screen.preview.delete.assert_called_with('gaze')
            self.assertEqual(screen.preview.create_line.call_count, 2)
            self.assertEqual(screen.gaze_point.get_current_position(), (0, 0))
            screen.start_button.configure.assert_not_called()
            screen.root.mainloop.side_effect = screen.cancel
            self.assertIsNone(screen.show())
            screen.tracker.stop.assert_called_once()
            screen.tracker.join.assert_called_once_with(timeout=1.)

    def test_start_during_failed_connection_transfers_receiver_and_retained_point(self):
        with ExitStack() as stack:
            factory = stack.enter_context(patch('ui.start_screen.ApiGazeTracker'))
            screen = self.make_screen(stack)
            tracker = factory.return_value
            tracker.get_current_position.return_value = (1440, 270)
            screen.fetch_gaze()
            screen._poll_gaze()
            tracker.get_current_position.return_value = None
            tracker.api_connected = False
            screen._poll_gaze()
            self.assertEqual(screen.gaze_point.get_current_position(), (1440, 270))
            screen.start()
            self.assertIs(screen.selection.tracker, tracker)
            self.assertIs(screen.selection.gaze_point, screen.gaze_point)
            screen.root.quit.assert_called_once()

    def test_run_without_connection_never_creates_receiver_and_cleans_up(self):
        with ExitStack() as stack, TemporaryDirectory() as directory:
            api_factory = stack.enter_context(patch('ui.start_screen.ApiGazeTracker'))
            mouse_factory = stack.enter_context(patch('main.GazeTracker'))
            for name in ('SerialReader', 'InputThread', 'BlinkSegmentProcessor', 'ObjAdder'):
                stack.enter_context(patch('main.' + name))
            timer = stack.enter_context(patch('main.TimerThread'))
            window = stack.enter_context(patch('main.display')).return_value
            stack.enter_context(patch('main.Path.write_text'))
            window.record_dir = Path(directory)
            window.close.return_value = Path(directory)
            window.run.return_value = 0
            args = SimpleNamespace(four=False, ar_green=False, nogaze=True, gaze_source='api',
                                   gaze_api_url='http://127.0.0.1:8765/gaze')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main.run_experiment(args), 0)
            api_factory.assert_not_called()
            mouse_factory.assert_not_called()
            point = timer.call_args.args[2]
            self.assertIsInstance(point, GazePointProvider)
            self.assertEqual(point.get_current_position(), (0, 0))
            self.assertTrue(timer.call_args.kwargs['nogaze'])
            self.assertEqual(window.set_gaze_provider.call_args.args[0](), (0, 0))
            timer.return_value.stop.assert_called_once()
            window.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
