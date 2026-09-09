"""Small codec fixtures exercise actual full-report media validation."""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'scripts', 'report_full_pipeline.py')
SPEC = importlib.util.spec_from_file_location('full_pipeline_report', SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required to create tiny encoded fixtures')
class ReportMediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='atman-report-media-')
        self.output = self.temporary.name
        self.conf = {'width': 64, 'heigth': 48}
        self.frames = []
        for index in range(6):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[8:36, 5 + index * 5:20 + index * 5] = (50, 180, 240)
            self.frames.append(frame)
        self.encode('trajectory.mp4', self.frames)
        shutil.copyfile(os.path.join(self.output, 'trajectory.mp4'), os.path.join(self.output, 'broadcast_overlay.mp4'))
        for name in ('speed_spin.png', 'serve_placement.png', 'ball_candidates.png',
                     'trajectory.png', 'broadcast_overlay.png'):
            self.assertTrue(cv2.imwrite(os.path.join(self.output, name), self.frames[-1]))

    def tearDown(self):
        self.temporary.cleanup()

    def encode(self, name, frames, fps=25):
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-filter_threads', '1',
                        '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '64x48',
                        '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                        '-threads', '1', '-preset', 'ultrafast', '-crf', '0',
                        '-pix_fmt', 'yuv420p', os.path.join(self.output, name)],
                       input=b''.join(frame.tobytes() for frame in frames), check=True)

    def verify(self, expected=6):
        record = {'checks': [], 'rendering': {'source_frames': list(range(expected))}}
        REPORT.verify_media(self.output, self.conf, record)
        return record

    def test_decodes_every_frame_and_png(self):
        record = self.verify()
        self.assertEqual(len(record['checks']), 7)
        self.assertTrue(all(check['passed'] for check in record['checks']))
        for name in ('trajectory.mp4', 'broadcast_overlay.mp4'):
            media = record['media_verification'][name]
            self.assertEqual(media['decoded_frames'], 6)
            self.assertTrue(media['decoded_to_eof'])
            self.assertEqual(media['fps'], 25)
            self.assertEqual(media['nonblank_frames'], 6)
            self.assertEqual(media['unique_decoded_frames'], 6)
            self.assertGreater(media['changing_frame_pairs'], 0)

    def test_wrong_frame_count_and_fps_fail(self):
        self.encode('trajectory.mp4', self.frames[:-1])
        self.encode('broadcast_overlay.mp4', self.frames, fps=24)
        failed = {check['name'] for check in self.verify()['checks'] if not check['passed']}
        self.assertEqual(failed, {'decoded_media:trajectory.mp4', 'decoded_media:broadcast_overlay.mp4'})

    def test_static_and_blank_videos_fail(self):
        self.encode('trajectory.mp4', [self.frames[0]] * 6)
        self.encode('broadcast_overlay.mp4', [np.zeros_like(self.frames[0])] * 6)
        record = self.verify()
        failed = {check['name'] for check in record['checks'] if not check['passed']}
        self.assertEqual(failed, {'decoded_media:trajectory.mp4', 'decoded_media:broadcast_overlay.mp4'})
        self.assertEqual(record['media_verification']['trajectory.mp4']['changing_frame_pairs'], 0)
        self.assertEqual(record['media_verification']['broadcast_overlay.mp4']['nonblank_frames'], 0)

    def test_corrupt_png_fails(self):
        with open(os.path.join(self.output, 'speed_spin.png'), 'wb') as handle:
            handle.write(b'not an image')
        failed = {check['name'] for check in self.verify()['checks'] if not check['passed']}
        self.assertEqual(failed, {'decoded_media:speed_spin.png'})

    def test_too_many_decoded_frames_fail_with_bounded_read(self):
        record = self.verify(expected=4)
        for name in ('trajectory.mp4', 'broadcast_overlay.mp4'):
            media = record['media_verification'][name]
            self.assertEqual(media['decoded_frames'], 5)
            self.assertFalse(media['decoded_to_eof'])
        self.assertFalse(all(check['passed'] for check in record['checks']))


if __name__ == '__main__':
    unittest.main()
