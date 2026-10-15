"""Measured gaze overlay is independent of the instructional target cue."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ui.display_psy import display_psy


class GazeDisplayTests(unittest.TestCase):
    def renderer(self):
        value = display_psy.__new__(display_psy)
        value.win = SimpleNamespace(size=(1920, 1080), mouseVisible=True)
        value._visual = Mock()
        value._mouse = Mock()
        value._gaze_cross = None
        return value

    def test_real_api_coordinates_draw_red_cross_and_update_each_frame(self):
        renderer = self.renderer()
        lines = [Mock(), Mock()]
        renderer._visual.Line.side_effect = lines
        provider = Mock(return_value=(1440, 270))
        renderer.set_gaze_provider(provider)
        self.assertFalse(renderer.win.mouseVisible)
        renderer._draw_live_gaze()
        self.assertEqual(lines[0].start, (468., 270.))
        self.assertEqual(lines[0].end, (492., 270.))
        self.assertEqual(lines[1].start, (480., 258.))
        self.assertEqual(renderer._visual.Line.call_args.kwargs['lineColor'], 'red')
        provider.return_value = (480, 810)
        renderer._draw_live_gaze()
        self.assertEqual(lines[0].start, (-492., -270.))
        self.assertEqual(renderer._visual.Line.call_count, 2)
        self.assertEqual(lines[0].draw.call_count, 2)
        provider.return_value = None
        renderer._draw_live_gaze()
        self.assertEqual(lines[0].draw.call_count, 2)

    def test_gaze_cross_is_drawn_alongside_separate_target_cue(self):
        renderer = self.renderer()
        renderer._cue_stim = Mock()
        renderer._draw_live_gaze = Mock()
        renderer._draw_cue()
        renderer._cue_stim.draw.assert_called_once()
        renderer._draw_live_gaze.assert_called_once()
