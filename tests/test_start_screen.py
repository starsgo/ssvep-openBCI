"""Live gaze is connected explicitly before the experiment starts."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main
from ui.start_screen import StartScreen


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
        self.assertEqual(main.main(['--four', '--ar-green']), 0)
        args = run.call_args.args[0]
        self.assertTrue(args.four)
        self.assertTrue(args.ar_green)
        self.assertEqual(args.gaze_source, 'api')
        self.assertIs(args.connected_gaze_tracker, tracker)
        tracker.stop.assert_called_once()
        tracker.join.assert_called_once_with(timeout=1.)

    def make_screen(self, stack):
        for name in ('Tk', 'Label', 'Canvas', 'Frame', 'Button'):
            stack.enter_context(patch('tkinter.' + name))
        return StartScreen()

    def test_start_requires_live_coordinate_and_does_not_auto_connect(self):
        with ExitStack() as stack:
            factory = stack.enter_context(patch('ui.start_screen.ApiGazeTracker'))
            screen = self.make_screen(stack)
            factory.assert_not_called()
            screen.start()
            self.assertIsNone(screen.selection)
            screen.root.quit.assert_not_called()

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
            screen._poll_gaze()
            self.assertEqual(screen.preview.create_line.call_count, 2)
            self.assertEqual(screen.preview.create_line.call_args_list[0].args, (322., 45., 338., 45.))
            screen.start_button.configure.assert_called_with(state='normal')
            screen.root.mainloop.side_effect = screen.start
            selection = screen.show()
            self.assertIs(selection.tracker, tracker)
            self.assertEqual(selection.source, 'api')
            tracker.stop.assert_not_called()
            screen.root.destroy.assert_called_once()

    def test_invalid_sample_hides_preview_and_cancel_stops_receiver(self):
        with ExitStack() as stack:
            screen = self.make_screen(stack)
            screen.tracker = Mock()
            screen.tracker.get_current_position.return_value = None
            screen.tracker.api_connected = False
            screen._poll_gaze()
            screen.preview.delete.assert_called_with('gaze')
            screen.preview.create_line.assert_not_called()
            screen.start_button.configure.assert_called_with(state='disabled')
            screen.root.mainloop.side_effect = screen.cancel
            self.assertIsNone(screen.show())
            screen.tracker.stop.assert_called_once()
            screen.tracker.join.assert_called_once_with(timeout=1.)


if __name__ == '__main__':
    unittest.main()
