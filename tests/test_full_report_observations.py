"""Audit actual selected-ball support independently of fitted coordinates."""

import importlib.util
import os
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'scripts', 'report_full_pipeline.py')
SPEC = importlib.util.spec_from_file_location('full_pipeline_report_observations', SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ShotObservationTests(unittest.TestCase):
    def test_raw_shot_indices_are_separated_by_rally_and_endpoint_is_restored(self):
        table = pd.DataFrame({'shot_idx': [0] * 6, 'event_descriptor': ['fh_ts'] * 6},
                             index=[100, 101, 102, 200, 201, 202])
        rallies = {frame: True for frame in [100, 101, 102, 103, 200, 201, 202, 203]}
        ball = {frame: SimpleNamespace(coords=(10, 20)) for frame in [100, 103, 201, 202, 203]}
        rows = REPORT.shot_observation_counts(table, ball, rallies)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['rally_idx'] for row in rows], [0, 1])
        self.assertEqual([row['selected_ball_detections'] for row in rows], [2, 3])
        self.assertEqual([row['selected_ball_detections_on_saved_rows'] for row in rows], [1, 2])
        self.assertEqual([row['fewer_than_3_detections'] for row in rows], [True, False])
        self.assertTrue(all(row['endpoint_detection'] for row in rows))

    def test_joined_tables_use_point_and_shot_indices(self):
        table = pd.DataFrame({'point_idx': [7, 7, 8, 8], 'shot_idx': [0] * 4}, index=[10, 11, 13, 14])
        rallies = {frame: True for frame in range(10, 16)}
        rows = REPORT.shot_observation_counts(table, {}, rallies)
        self.assertEqual([row['point_idx'] for row in rows], [7, 8])
        self.assertEqual([row['start_frame'] for row in rows], [10, 13])

    def test_inferred_xy_with_nan_z_are_not_input_observations(self):
        table = pd.DataFrame({'shot_idx': [0] * 10, 'x_o': [1] * 10, 'y_o': [2] * 10,
                              'z_o': [np.nan] * 10}, index=range(1121, 1131))
        rallies = {frame: True for frame in range(1121, 1132)}
        ball = {1125: SimpleNamespace(coords=(50, 60)), 1126: None,
                1127: SimpleNamespace(coords=(np.nan, 3))}
        row = REPORT.shot_observation_counts(table, ball, rallies)[0]
        self.assertEqual(row['fit_stop_frame'], 1131)
        self.assertEqual(row['selected_ball_detections'], 1)
        self.assertTrue(row['fewer_than_3_detections'])


if __name__ == '__main__':
    unittest.main()
