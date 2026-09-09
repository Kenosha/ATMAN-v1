"""Recognition CLI contracts: bounded local work and honest no-court output."""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from ATMANpy.cli import recognize


class ArgumentsTest(unittest.TestCase):
    def test_duration_and_start_cannot_escape_the_processing_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, 'local.mp4')
            with open(video, 'wb') as handle:
                handle.write(b'local video placeholder')
            for value in ['nan', 'inf', '-1', '0', '30.01']:
                with self.subTest(duration=value):
                    args = recognize.parser().parse_args([video, '--duration', value])
                    with self.assertRaises(ValueError):
                        recognize.validate_args(args)
            for value in ['nan', 'inf', '-1']:
                with self.subTest(start=value):
                    args = recognize.parser().parse_args([video, '--start', value])
                    with self.assertRaises(ValueError):
                        recognize.validate_args(args)

    def test_local_input_and_existing_outputs_are_preserved(self):
        args = recognize.parser().parse_args(['https://example.invalid/video.mp4'])
        with self.assertRaisesRegex(ValueError, 'existing local file'):
            recognize.validate_args(args)
        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, 'local.mp4')
            with open(video, 'wb') as handle:
                handle.write(b'unchanged input')
            args = recognize.parser().parse_args([video, '--output', directory])
            with self.assertRaisesRegex(ValueError, 'existing outputs are preserved'):
                recognize.validate_args(args)
            with open(video, 'rb') as handle:
                self.assertEqual(handle.read(), b'unchanged input')


class OptionalModelsTest(unittest.TestCase):
    def test_missing_torch_is_reported_without_fabricating_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, 'weights'))
            with open(os.path.join(directory, 'weights', recognize.SOUND_MODEL), 'wb') as handle:
                handle.write(b'placeholder; model must not be loaded without torch')
            args = recognize.parser().parse_args(['unused.mp4', '--sound-model-dir', directory])
            report = {'stages': {}}
            with patch.dict(sys.modules, {'torch': None}):
                result = recognize.recognize_audio(args, directory, report)
            self.assertIsNone(result)
            self.assertEqual(report['stages']['sound_cnn']['status'], 'failed')
            self.assertIn('torch', report['stages']['sound_cnn']['error'])
            self.assertFalse(os.path.exists(os.path.join(directory, 'sound_probabilities.csv')))


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for the end-to-end video check')
class NoCourtVideoTest(unittest.TestCase):
    def test_blank_video_is_not_reported_as_tennis(self):
        import cv2 as cv
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, 'blank.mp4')
            writer = cv.VideoWriter(video, cv.VideoWriter_fourcc(*'mp4v'), 25, (160, 90))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(25):
                    writer.write(np.zeros((90, 160, 3), dtype='uint8'))
            finally:
                writer.release()
            with open(video, 'rb') as handle:
                original_bytes = handle.read()
            output = os.path.join(directory, 'results')
            code = recognize.main([video, '--duration', '0.8', '--calibration-samples', '2',
                                   '--output', output, '--data-root', directory, '--no-audio'])
            self.assertEqual(code, 2, 'A usable artifact is produced, but a no-court attempt must not claim success.')
            with open(os.path.join(output, 'report.json'), encoding='utf-8') as handle:
                report = json.load(handle)
            self.assertEqual(report['stages']['court_calibration']['status'], 'not_detected')
            self.assertNotIn('homography', report['stages']['court_calibration'])
            self.assertEqual(report['stages']['foreground_and_motion']['court_frames'], 0)
            self.assertEqual(report['stages']['ball_svm']['status'], 'skipped')
            self.assertEqual(report['decoded_frames'], 20)
            with open(os.path.join(output, 'frame_features.csv'), newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(row['is_court'] == 'False' for row in rows))
            self.assertTrue(all(row['ball_svm_candidates'] == '0' for row in rows))
            cap = cv.VideoCapture(os.path.join(output, 'recognition.mp4'))
            try:
                self.assertEqual(int(cap.get(cv.CAP_PROP_FRAME_COUNT)), 20)
                self.assertTrue(cap.read()[0])
            finally:
                cap.release()
            for filename in ['preview.png', 'recognition_analysis.png']:
                self.assertIsNotNone(cv.imread(os.path.join(output, filename)))
            with open(video, 'rb') as handle:
                self.assertEqual(handle.read(), original_bytes)


if __name__ == '__main__':
    unittest.main()
