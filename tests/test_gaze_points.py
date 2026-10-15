"""Display fallback coordinates must not masquerade as measured eye frames."""
import unittest
from unittest.mock import Mock

from gaze import GazePointProvider


class GazePointTests(unittest.TestCase):
    def test_default_origin_is_displayable_but_is_not_a_measurement(self):
        point = GazePointProvider()
        self.assertEqual(point.get_current_position(), (0, 0))
        self.assertIsNone(point.get_current_measurement())
        self.assertFalse(point.has_measurement)
        self.assertFalse(point.current_is_valid)

    def test_invalid_and_disconnected_samples_hold_last_point_then_recover(self):
        source = Mock()
        source.get_current_position.return_value = (1500, 300)
        point = GazePointProvider(source)
        self.assertEqual(point.get_current_position(), (1500, 300))
        source.get_current_position.return_value = None
        self.assertIsNone(point.get_current_measurement())
        self.assertEqual(point.get_current_position(), (1500, 300))
        self.assertTrue(point.has_measurement)
        self.assertFalse(point.current_is_valid)
        source.get_current_position.return_value = (480, 810)
        self.assertEqual(point.get_current_position(), (480, 810))
        self.assertTrue(point.current_is_valid)

    def test_measured_origin_is_a_valid_measurement(self):
        source = Mock()
        source.get_current_position.return_value = (0, 0)
        point = GazePointProvider(source)
        self.assertEqual(point.get_current_measurement(), (0, 0))
        self.assertTrue(point.has_measurement)
        self.assertTrue(point.current_is_valid)
