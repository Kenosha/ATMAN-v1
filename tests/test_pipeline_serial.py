"""Serial execution must preserve feature results and avoid process pools."""
import os
import tempfile
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np

from ATMANpy.types.Features import SFList, StreamFeature, StreamFeatures
from ATMANpy.types.Streams import Stream, apply_async


def double(value):
    result = value * 2
    return result


def whole(feature, increment=0):
    return StreamFeature({frame: value + increment for frame, value in feature.items()}, name=feature.name)


def mean_pixel(gFrm):
    mean = float(gFrm.array.mean())
    return mean


class SerialPipelineTest(unittest.TestCase):
    def test_interpolated_audio_frame_numbers_merge_with_integer_video_frames(self):
        import pandas as pd
        audio_frame = pd.DataFrame({'hit': [0.1, 0.7], 'bounce': [0.2, 0.3]}, index=[12.0, 13.0])
        audio = StreamFeatures.from_dataframe(audio_frame)
        video = StreamFeatures({13: {'top_oth': 0.5}, 14: {'top_oth': 0.9}})
        merged = video.merge(audio, how='outer')
        self.assertTrue(all(isinstance(frame, int) for frame in audio.get_frames()))
        self.assertEqual(merged[13], {'top_oth': 0.5, 'hit': 0.7, 'bounce': 0.3})
        self.assertEqual(set(merged.get_frames()), {12, 13, 14})
        np.testing.assert_array_equal(audio.to_dataframe().values, audio_frame.values)

    def test_saved_chunk_keeps_match_frame_numbers_and_last_decodable_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'from_853_to_857.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            for value in [10, 40, 90, 140, 210]:
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()
            stream = Stream.from_file(path, target_color_space='grey')
            try:
                self.assertEqual((stream.start_frame, stream.stop_frame, stream.length()), (853, 857, 5))
                self.assertAlmostEqual(stream.get(853).array.mean(), 10, delta=1)
                self.assertAlmostEqual(stream.get(857).array.mean(), 210, delta=1)
                self.assertEqual(stream.decap().frames.length(), 5)
                widened = stream.widen(2)
                self.assertEqual((widened.start_frame, widened.stop_frame), (853, 857))
                full = Stream(stream.cap, 0, 5)
                self.assertEqual((full.start_frame, full.stop_frame), (0, 4))
            finally:
                stream.cap.release()

    def test_video_reads_preserve_forward_backward_and_repeated_frames(self):
        from ATMANpy.funcs.cv_funcs import get_frame, get_framelist
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'clip.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            for value in [10, 40, 90, 140, 210]:
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()
            order = [0, 1, 2, 2, 4, 0, 3]
            cap = cv2.VideoCapture(path)
            expected = []
            for index in order:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                self.assertTrue(ok)
                expected.append(frame)
            cap.release()
            for batch in (False, True):
                cap = cv2.VideoCapture(path)
                wrapped = Mock(wraps=cap)
                try:
                    actual = (get_framelist(wrapped, order) if batch else
                              [get_frame(wrapped, index) for index in order])
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(wrapped.set.call_count, 4)
                finally:
                    cap.release()

    def test_feature_list_serial_results_and_return_types(self):
        features = SFList([StreamFeature({1: 3, 2: 7}, name='value'),
                           StreamFeature({5: 11}, name='value')])
        with patch('multiprocessing.pool.Pool', side_effect=AssertionError('Unexpected worker pool')):
            scaled = features.apply_mp(double, 1)
            shifted = features.apply_whole_mp(whole, 1, increment=4)
            paired = features.apply_whole_mp_starmapped(whole, 1, [2, 9])
        self.assertEqual([dict(f.items()) for f in scaled], [{1: 6, 2: 14}, {5: 22}])
        self.assertEqual([dict(f.items()) for f in shifted], [{1: 7, 2: 11}, {5: 15}])
        self.assertEqual([dict(f.items()) for f in paired], [{1: 5, 2: 9}, {5: 20}])

    def test_stream_serial_extraction_equals_regular_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'clip.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            for value in [10, 40, 90, 140, 210]:
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()
            cap = cv2.VideoCapture(path)
            try:
                stream = Stream(cap, 0, 4, target_color_space='grey')
                expected = stream.extract(mean_pixel)
                with patch('ATMANpy.types.Streams.Pool', side_effect=AssertionError('Unexpected worker pool')):
                    actual = stream.extract_mp(mean_pixel, 1)
                self.assertEqual(dict(actual.items()), dict(expected.items()))
            finally:
                cap.release()

    def test_chunk_dispatch_keeps_order_without_pool(self):
        class Chunk:
            def __init__(self, value):
                self.value = value
            def apply(self, function, offset=0):
                return function(self.value) + offset
        def forbidden(**kwargs):
            raise AssertionError('Unexpected worker pool')
        result = apply_async(forbidden, iter([Chunk(2), Chunk(9)]), 1, double,
                             metaFunc='apply', kwargs={'offset': 3})
        self.assertEqual(result, [7, 21])


if __name__ == '__main__':
    unittest.main()
