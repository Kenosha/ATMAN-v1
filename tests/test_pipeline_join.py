"""Regression checks for joining short recordings and writing empty placeholders."""
import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.util_funcs as ut

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'ATMANpy', 'scripts', 'pipeline_brest_tightseg', 'join_traj_and_other_info.py')
spec = importlib.util.spec_from_file_location('pipeline_join_for_test', SOURCE)
join = importlib.util.module_from_spec(spec)
spec.loader.exec_module(join)


class PipelineJoinTests(unittest.TestCase):
    def test_numeric_and_missing_scores_form_one_zero_score(self):
        values = pd.Series([0, 0.0, np.int64(0), np.float32(0), np.nan, None, pd.NA], dtype=object)
        normalized = values.map(join.game_score_as_text)
        self.assertEqual(normalized.unique().tolist(), ['0'])
        self.assertEqual([[int(x) for x in score.split('-')] for score in normalized], [[0]] * 7)

    def test_multiset_score_text_is_preserved(self):
        values = pd.Series(['6-4-0', '4-6-0', '7-5', '0'])
        normalized = values.map(join.game_score_as_text)
        self.assertEqual(normalized.tolist(), values.tolist())
        self.assertEqual([[int(x) for x in score.split('-')] for score in normalized],
                         [[6, 4, 0], [4, 6, 0], [7, 5], [0]])

    def test_fractional_scores_are_not_silently_truncated(self):
        for value in (0.5, np.float64(2.7), np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'whole number'):
                join.game_score_as_text(value)

    def test_all_rallies_with_trajectories_write_readable_empty_placeholders(self):
        with tempfile.TemporaryDirectory(prefix='atman-join-test-') as root:
            os.makedirs(os.path.join(root, 'features_stored'))
            with patch.object(ut, 'mainDir', root):
                frame = pd.DataFrame({'event': ['STD', 'F'], 'point_idx': [0, 0]},
                                     index=pd.Index([308, 309], name='idx0'))
                join.save([frame], [frame.copy()], [], 'test-video', 'fresh')
                trajectories = ff.read_feature_frame('dataTable', 'test-video', 'fresh', keep_only='last')
                placeholders = ff.read_feature_frame('placeTable', 'test-video', 'fresh', keep_only='last')
                assert_frame_equal(trajectories, frame)
                self.assertTrue(placeholders.empty)
                self.assertEqual(placeholders.columns.tolist(),
                                 ['point_idx', 'lo_name', 'up_name', 'lo_score', 'up_score',
                                  'lo_games', 'up_games', 'lo_sets', 'up_sets'])

    def test_existing_placeholder_rows_are_preserved(self):
        captured = {}
        def capture(frame, name, video, store):
            captured[name] = frame
        frame = pd.DataFrame({'event': ['STD']}, index=[10])
        placeholder = pd.Series({'point_idx': 1, 'lo_name': 'A', 'up_name': 'B',
                                 'lo_score': 15, 'up_score': 0, 'lo_games': 0,
                                 'up_games': 0, 'lo_sets': 0, 'up_sets': 0})
        with patch.object(join.ff, 'save_feature_frame', side_effect=capture):
            join.save([frame], [frame], [placeholder], 'test-video', 'fresh')
        assert_frame_equal(captured['placeTable'], pd.concat([placeholder], axis=1).T)


if __name__ == '__main__':
    unittest.main()
