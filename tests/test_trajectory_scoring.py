"""Bounded trajectory scoring must preserve the historical numerical result."""
import unittest

import numpy as np

from ATMANpy.models.BallTrajectory.flight_bounce_model import _trajectory_deviation_sums


class TrajectoryScoringTest(unittest.TestCase):
    def test_full_grid_deviations_match_broadcast_formula(self):
        rng = np.random.default_rng(7201)
        for dtype in [np.float16, np.float32, np.float64]:
            ground = rng.normal(size=(7, 4, 3, 5, 11)).astype(dtype)
            vertical = rng.normal(size=ground.shape).astype(dtype)
            measured_ground = rng.normal(size=(3, 11)).astype('float32')
            measured_vertical = rng.normal(size=(3, 11)).astype('float32')
            measured_ground[1, 2] = np.nan
            measured_vertical[2, 5] = np.nan
            ground[2, 1, 1, 0, 3] = np.nan
            reference_dtype = np.result_type(dtype, np.float32)
            dg = measured_ground[:, None, None, None, None] - ground.astype(reference_dtype)[None]
            dz = measured_vertical[:, None, None, None, None] - vertical.astype(reference_dtype)[None]
            for squared in [True, False]:
                expected = np.nansum((dg ** 2 + dz ** 2) ** 0.5 if squared
                                    else np.abs(dg) + np.abs(dz), axis=-1)
                for allowance in [0.00001, 0.001, 1]:
                    with self.subTest(dtype=dtype, squared=squared, allowance=allowance):
                        actual = _trajectory_deviation_sums(
                            ground, vertical, measured_ground, measured_vertical,
                            sum_of_squares=squared, ram_allowed_in_gb=allowance)
                        np.testing.assert_array_equal(actual, expected)

    def test_nonpositive_memory_allowance_is_rejected(self):
        bank = np.ones((1, 1, 1, 1, 3), dtype='float32')
        observations = np.ones((1, 3), dtype='float32')
        for allowance in [0, -1]:
            with self.subTest(allowance=allowance), self.assertRaises(ValueError):
                _trajectory_deviation_sums(bank, bank, observations, observations,
                                          ram_allowed_in_gb=allowance)


if __name__ == '__main__':
    unittest.main()
