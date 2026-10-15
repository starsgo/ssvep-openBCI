"""API gaze freshness, coordinate conversion and reconnect handling."""
import json
import threading
import time
from unittest.mock import patch
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gaze.api import ApiGazeTracker


class ApiTrackerTests(unittest.TestCase):
    @staticmethod
    def packet(u=.2, v=.3, **extra):
        return {'schema': 'slippage.gaze.v1', 'coordinate_space': 'normalized_top_left',
                'valid': True, 'u': u, 'v': v, 'age_ms': 10., 'frame_id': 1, **extra}

    def test_normalized_actual_eye_coordinates_map_to_window_pixels_and_side(self):
        tracker = ApiGazeTracker(window_size=(1920, 1080))
        with patch('gaze.api.time.monotonic', return_value=10.):
            tracker.update_from_api(self.packet(.75, .25), received_at=10.)
            self.assertEqual(tracker.get_current_position(), (1440, 270))
            self.assertEqual(tracker.get_current_region(), 'right')
            tracker.update_from_api(self.packet(.5, 1.), received_at=10.)
            self.assertEqual(tracker.get_current_position(), (960, 1079))
            self.assertEqual(tracker.get_current_region(), 'right')

    def test_stale_or_invalid_packets_clear_prior_gaze(self):
        tracker = ApiGazeTracker()
        for packet in (self.packet(valid=False), self.packet(age_ms=210.),
                       self.packet(u=float('nan')), self.packet(schema='other'),
                       self.packet(coordinate_space='bottom_left'), self.packet(v=float('inf'))):
            with self.subTest(packet=packet), patch('gaze.api.time.monotonic', return_value=10.):
                self.assertTrue(tracker.update_from_api(self.packet(), received_at=10.))
                self.assertFalse(tracker.update_from_api(packet, received_at=10.01))
                self.assertIsNone(tracker.get_current_position())

    def test_offscreen_coordinates_clamp_to_edges_and_keep_gaze_side(self):
        tracker = ApiGazeTracker(window_size=(1920, 1080))
        for point, expected, side in (
            ((-.1, .5), (0, 540), 'left'),
            ((1.1, .5), (1919, 540), 'right'),
            ((.25, -.2), (480, 0), 'left'),
            ((.75, 1.2), (1440, 1079), 'right'),
            ((-1e308, 1e308), (0, 1079), 'left'),
            ((1e308, -1e308), (1919, 0), 'right'),
        ):
            with self.subTest(point=point), patch('gaze.api.time.monotonic', return_value=10.):
                self.assertTrue(tracker.update_from_api(self.packet(*point), received_at=10.))
                self.assertEqual(tracker.get_current_position(), expected)
                self.assertEqual(tracker.get_current_region(), side)

    def test_samples_expire_even_when_api_stops_responding_and_rtt_counts_toward_age(self):
        tracker = ApiGazeTracker()
        tracker.update_from_api(self.packet(age_ms=80.), received_at=10., request_elapsed=.02)
        with patch('gaze.api.time.monotonic', return_value=10.09):
            self.assertIsNotNone(tracker.get_current_position())
        with patch('gaze.api.time.monotonic', return_value=10.11):
            self.assertIsNone(tracker.get_current_position())
        self.assertFalse(tracker.update_from_api(self.packet(age_ms=180.),
                                                received_at=11., request_elapsed=.03))

    def test_real_http_receiver_follows_new_position_and_clears_invalid_sample(self):
        payload = [self.packet()]
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def do_GET(self):
                body = json.dumps(payload[0]).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        tracker = ApiGazeTracker(f'http://127.0.0.1:{server.server_port}/gaze', poll_interval=.005)
        try:
            tracker.start()
            self.assertTrue(tracker.ready_event.wait(1.))
            self.assertEqual(tracker.get_current_region(), 'left')
            payload[0] = self.packet(.8, .3, frame_id=2)
            deadline = time.monotonic() + 1.
            while tracker.get_current_region() != 'right' and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(tracker.get_current_position(), (1536, 324))
            payload[0] = self.packet(-.1, 1.2, frame_id=3)
            deadline = time.monotonic() + 1.
            while tracker.frame_id != 3 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(tracker.get_current_position(), (0, 1079))
            self.assertEqual(tracker.get_current_region(), 'left')
            payload[0] = self.packet(valid=False, reason='no_measurement')
            deadline = time.monotonic() + 1.
            while tracker.get_current_position() is not None and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertIsNone(tracker.get_current_position())
        finally:
            tracker.stop()
            tracker.join(timeout=1.)
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.)
        self.assertFalse(tracker.is_alive())


if __name__ == '__main__':
    unittest.main()
