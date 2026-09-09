"""Projection audit follows ATMAN's actual Camera/View conventions."""

import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ATMANpy.types.Features import ConstFeature, StreamFeature
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Shapes3D import Point3D
from ATMANpy.types.Visual3D import Camera, Scene, View


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'scripts', 'report_full_pipeline.py')
SPEC = importlib.util.spec_from_file_location('full_pipeline_report_camera', SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ReportCameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera.pinhole_origin_aligned(100, 100, 32, 24, 64, 48)

    def test_projection_equivalence_accepts_positive_and_negative_scale(self):
        projection = self.camera.get_P()
        for scale in (7.0, -3.038255200143349e-6):
            with self.subTest(scale=scale):
                equivalent, info = REPORT.projection_equivalence(scale * projection, projection)
                self.assertTrue(equivalent)
                self.assertAlmostEqual(info['scale_to_saved_P'], scale)
                self.assertLess(info['relative_frobenius_error_after_scale'], 1e-12)

    def test_projection_equivalence_rejects_different_or_degenerate_maps(self):
        projection = self.camera.get_P()
        changed = projection.copy()
        changed[0, 3] += 100
        self.assertFalse(REPORT.projection_equivalence(changed, projection)[0])
        self.assertFalse(REPORT.projection_equivalence(np.zeros((3, 4)), projection)[0])

    def test_report_residuals_match_real_view_front_and_behind_visibility(self):
        coordinates = np.asarray([(0, 0, -10), (1, 0, -10), (0, 0, 10)], dtype=float)
        scene = Scene([Point3D(tuple(xyz)) for xyz in coordinates])
        view = View.make_without_single_pixel_pruning(self.camera, scene)
        # Visibility is decided by the actual View implementation, not a mock.
        self.assertEqual(len(view), 2)
        np.testing.assert_array_equal(view.ptArr, coordinates[:2])
        frames = [10, 11, 12]
        detections = StreamFeature({number: Point2D(tuple(xy)) for number, xy in
                                    zip(frames, self.camera.project_points(coordinates))})
        rallies = StreamFeature({number: True for number in range(10, 14)})
        table = pd.DataFrame(coordinates, columns=['x_t', 'y_t', 'z_t'], index=frames)
        table = table.assign(point_idx=0, shot_idx=0, event='HT', event_descriptor='fh_ts',
                             hit_name='one', rec_name='two')
        tables = {name: table.copy() for name in ('trajTable', 'colinearTable_fair', 'colinearTable_fault',
                                                  'dataTable', 'dataTable_colinear')}
        matrices = {'M_int': self.camera.M_int, 'M_ext': self.camera.M_ext,
                    'P': -3.038255200143349e-6 * self.camera.get_P()}
        load = lambda *args, featname: SimpleNamespace(feature=matrices[featname])
        record = {'checks': [], 'warnings': []}
        with patch.object(ConstFeature, 'from_store', side_effect=load):
            _, residuals = REPORT.trajectory_checks(tables, {'video_id': 'synthetic', 'store_name': 'fresh',
                                                            'width': 64, 'heigth': 48}, detections, record, rallies)
        for info in record['trajectories'].values():
            self.assertEqual(info['behind_camera_rows'], 1)
            self.assertEqual(info['projection_pairs'], len(view))
            self.assertEqual(info['median_residual_px'], 0)
        self.assertEqual(len(residuals), 4)  # two visible frames in each joined table
        self.assertEqual({row['frame'] for row in residuals}, {10, 11})
        matrix_check = next(check for check in record['checks'] if check['name'] == 'saved_projection_matches_camera')
        self.assertTrue(matrix_check['passed'])


if __name__ == '__main__':
    unittest.main()
