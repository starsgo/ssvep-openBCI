import unittest

from mySerial.input import SerialReader


def make_frame(frame_number=1):
    return bytes([0xA0, frame_number]) + bytes(30) + bytes([0xC0])


class SerialFrameAlignmentTests(unittest.TestCase):
    def make_reader(self, data):
        reader = SerialReader.__new__(SerialReader)
        reader.buffer = bytearray(data)
        return reader

    def test_discards_prefix_and_recovers_frame_boundary(self):
        reader = self.make_reader(b"noise" + make_frame(17))

        raw_frame = reader._pop_next_frame()
        parsed = reader.parse_data(raw_frame)

        self.assertEqual(parsed["fn"], 17)
        self.assertEqual(parsed["eeg"], [0] * 8)

    def test_skips_false_start_byte_before_valid_frame(self):
        invalid = bytes([0xA0]) + bytes(32)
        reader = self.make_reader(invalid + make_frame(23))

        raw_frame = reader._pop_next_frame()

        self.assertEqual(reader.parse_data(raw_frame)["fn"], 23)

    def test_waits_for_complete_frame(self):
        reader = self.make_reader(make_frame()[:20])

        self.assertIsNone(reader._pop_next_frame())
        self.assertEqual(len(reader.buffer), 20)


if __name__ == "__main__":
    unittest.main()
