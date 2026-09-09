"""Check the public sample and optional stored match against the original ATMAN results."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['ATMAN_HOME'] = ROOT
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from ATMANpy.cli import demo


def expected(name):
    with open(os.path.join(ROOT, 'tests', 'fixtures', name + '_expected.json')) as handle:
        return json.load(handle)


class SampleDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, cls.points = demo.load_points(ROOT, sample=True)

    def test_sample_matches_original_numeric_results(self):
        actual, rows = demo.summarize(self.table, self.points)
        self.assertEqual(actual, expected('sample'))
        self.assertEqual(len(rows), 65)

    def test_filters_form_the_expected_subsets(self):
        shots = self.points.flatten()
        serves = shots.filter('is_serve')
        first = shots.filter('is_first_serve')
        second = shots.filter('is_second_serve')
        lets = shots.filter('is_let')
        self.assertEqual((len(serves), len(first), len(second), len(lets)), (12, 8, 3, 1))
        self.assertEqual(len(shots.filter('get_initial_speed', is_above=150)), 9)
        self.assertTrue(all(shot.is_serve() for shot in first))
        self.assertEqual(len(self.points.get_returns()), 9)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for animation export')
    def test_sample_produces_real_charts_and_changing_video_frames(self):
        import cv2
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                status = demo.main(['--sample', '--data-root', ROOT,
                                    '--output', directory, '--max-frames', '12'])
            self.assertEqual(status, 0)
            for name in ['serve_placement', 'serve_density', 'speed_spin', 'trajectory']:
                array = cv2.imread(os.path.join(directory, name + '.png'))
                self.assertIsNotNone(array)
                self.assertGreater(np.count_nonzero(array), 500)
            capture = cv2.VideoCapture(os.path.join(directory, 'trajectory.mp4'))
            frames = []
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 25)
                while True:
                    ok, array = capture.read()
                    if not ok:
                        break
                    frames.append(array)
            finally:
                capture.release()
            self.assertEqual(len(frames), 12)
            self.assertFalse(np.array_equal(frames[0], frames[-1]))
            yellow = (frames[-1][:, :, 1] > 180) & (frames[-1][:, :, 0] < 130)
            self.assertGreater(np.count_nonzero(yellow), 50)
            with Image.open(os.path.join(directory, 'trajectory.gif')) as animation:
                self.assertGreater(animation.n_frames, 1)


@unittest.skipUnless(os.path.isdir(os.path.join(ROOT, 'features_stored', demo.MATCH)),
                     'Optional full match features are not distributed with this repository')
class FullMatchTest(unittest.TestCase):
    def test_full_store_matches_original_numeric_results(self):
        from ATMANpy.funcs import util_funcs
        with patch.object(util_funcs, 'mainDir', ROOT):
            table, points = demo.load_points(ROOT)
        actual, _ = demo.summarize(table, points)
        self.assertEqual(actual, expected('full_match'))


class LocalVideoTest(unittest.TestCase):
    def test_local_path_and_match_id_open_the_same_video(self):
        import cv2
        import numpy as np
        from ATMANpy.funcs import util_funcs

        with tempfile.TemporaryDirectory() as directory:
            folder = os.path.join(directory, 'video_files', 'example')
            os.makedirs(folder)
            path = os.path.join(folder, 'match.mp4')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
            writer.release()
            with patch.object(util_funcs, 'mainDir', directory):
                for identifier in [path, 'example']:
                    capture = util_funcs.get_video(identifier)
                    try:
                        self.assertTrue(capture.read()[0])
                    finally:
                        capture.release()
                with self.assertRaises(FileNotFoundError):
                    util_funcs.get_video('https://example.invalid/missing.mp4')


if __name__ == '__main__':
    unittest.main()
