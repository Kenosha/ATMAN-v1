"""Track discovery order must not change temporal joining or lose samples."""
import unittest

import numpy as np
import pandas as pd

from ATMANpy.funcs.sf_funcs import iteratively_match_tracks, split_player_candidate_tracks
from ATMANpy.types.Contours import Contour, Contours
from ATMANpy.types.Features import StreamFeatures
from ATMANpy.types.Points import Point2D


class PlayerTrackOrderTest(unittest.TestCase):
    def test_contours_of_different_lengths_compare_without_broadcasting(self):
        points = np.arange(28, dtype=np.int32).reshape(14, 1, 2)
        original = Contour(points)
        self.assertEqual(original, Contour(points.copy()))
        self.assertNotEqual(original, Contour(points + 1))
        self.assertNotEqual(original, Contour(np.arange(32, dtype=np.int32).reshape(16, 1, 2)))

    def test_short_late_track_does_not_put_earlier_samples_last(self):
        # The backward traversal finds the short track before the long one.
        parents = {0: [[]], 5: [[0]], 10: [[], [0]], 15: [[0], [1]]}
        candidates = StreamFeatures({frame: {'player_parents': parents.get(frame)}
                                     for frame in range(16)})
        actual = split_player_candidate_tracks(candidates, frame_offset=5)
        expected = pd.DataFrame({0: [np.nan, np.nan, 0, 0], 1: [0, 0, 1, 1]},
                                index=[0, 5, 10, 15])
        pd.testing.assert_frame_equal(actual, expected)
        self.assertTrue(actual.index.is_unique)
        self.assertTrue(actual.index.is_monotonic_increasing)

    def test_unsorted_tracks_join_only_at_actual_temporal_boundaries(self):
        frames = [10, 15, 0, 5, 20, 25]
        contours = Contours([Contour.from_point(Point2D((10, 10)))])
        candidates = StreamFeatures({frame: {'player_candidates': contours} for frame in sorted(frames)})
        # Track 3 overlaps tracks 0 and 1. In row order it appears to start at
        # frame 10 and finish at frame 5, which used to make it join itself.
        tracks = pd.DataFrame({0: {0: 0, 5: 0}, 1: {10: 0, 15: 0},
                               2: {20: 0, 25: 0}, 3: {5: 0, 10: 0}}).reindex(frames)
        result = iteratively_match_tracks(candidates, tracks, frame_offset=5)
        ordered_result = iteratively_match_tracks(candidates, tracks.sort_index(), frame_offset=5)
        pd.testing.assert_frame_equal(result.sort_index(), ordered_result.sort_index())
        pd.testing.assert_frame_equal(result[tracks.columns], tracks, check_column_type=False)
        self.assertTrue(result.index.is_unique)
        self.assertNotIn('3_3', result.columns)
        self.assertIn('0_1_2', result.columns)
        np.testing.assert_array_equal(result['0_1_2'].sort_index(), np.zeros(len(frames)))


if __name__ == '__main__':
    unittest.main()
