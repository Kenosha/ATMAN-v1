"""Small generated banks exercise colinear scores, weighting, and exported heights."""
from itertools import product
from types import SimpleNamespace
import unittest

import numpy as np

from ATMANpy.funcs.sf_apply_funcs import (
    _colinear_trajectory_probabilities, _identical_trajectory_counts,
    calculate_trajectory_coordinates, fit_trajectory_colinear, get_highest_prob,
    make_empty_coord_and_plane_array, make_trajectory_arrays,
    trajectory_height_prob, trajectory_hori_prob, trajectory_spin_prob,
)


def small_bank():
    heights = np.array([0, 1, 2.5, 3.5, 4.5], dtype='float32')
    horizontal = np.array([5, 15, 25], dtype='float32')
    vertical = np.array([-3, 0, 3], dtype='float32')
    spins = np.array([-1000, 0, 1000], dtype='float32')
    t = np.arange(9, dtype='float32') / 25
    shape = (3, 3, 3, 5, len(t))
    ground = np.broadcast_to(horizontal[:, None, None, None, None] * t, shape).copy()
    z = np.broadcast_to(np.maximum(heights[None, None, None, :, None]
                                   + vertical[None, :, None, None, None] * t
                                   - 0.5 * 9.78 * t ** 2, 0), shape).copy()
    rpm = np.broadcast_to(spins[None, None, :, None, None], shape).copy()
    return ground, z, rpm, heights, horizontal, vertical, spins


def historical_probabilities(ground, z, rpm, ox, oy, dx, dy, bounce, net, hit, heights, speeds, spins):
    """Original unblocked probability expression, evaluated only on a tiny bank."""
    ground, z, rpm = [arr.astype('float32') for arr in [ground, z, rpm]]
    distances = np.abs(dx * (oy - z) - dy * (ox - ground))
    mean_distance = np.nanmean(distances, axis=-1)
    is_bounce = z == 0
    count = is_bounce.sum(axis=-1)
    p0, p1, p2 = (0.9, 0.09, 0.01) if net else bounce[0, 2:]
    c1 = np.prod(np.where(is_bounce, bounce[:, 0], 1), axis=-1)
    c2 = np.prod(np.where(is_bounce, bounce[:, 1], 1), axis=-1)
    prior = np.where(count == 0, p0, np.where(count == 1, p1, p2))
    conditional = np.where(count == 0, 1, np.where(count == 1, c1, c2))
    return ((1 / (1 + mean_distance)) * (prior * conditional)
            * trajectory_spin_prob(rpm, hit, spin_rate_space=spins)
            * trajectory_hori_prob(ground, hit, v_init_ground_space=speeds)
            * trajectory_height_prob(z, hit, height_space=heights))


class ColinearReconstructionTest(unittest.TestCase):
    def test_bounded_probabilities_match_original_expression(self):
        ground, z, rpm, heights, speeds, _, spins = small_bank()
        ox = np.linspace(10, 15, len(heights))[None, None, None, :, None]
        oy = np.linspace(12, 8, len(heights))[None, None, None, :, None]
        dx = np.linspace(0.8, 0.6, 9)[None, None, None, None]
        dy = np.linspace(-0.6, -0.8, 9)[None, None, None, None]
        dx[..., -1] = np.nan
        bounce = np.tile([1 / 9, 1 / 9, 0.1, 0.85, 0.05], (9, 1))
        for dtype, net, hit, allowance in product([np.float16, np.float32], [False, True],
                                                  ['bot_srv', 'top_fh_ts'], [0.00001, 1]):
            with self.subTest(dtype=dtype, net=net, hit=hit, allowance=allowance):
                arrays = [arr.astype(dtype) for arr in [ground, z, rpm]]
                expected = historical_probabilities(*arrays, ox, oy, dx, dy, bounce, net, hit,
                                                    heights, speeds, spins)
                actual = _colinear_trajectory_probabilities(
                    *arrays, ox, oy, dx, dy, bounce, net, hit, heights, speeds, spins,
                    ram_allowed_in_gb=allowance)
                np.testing.assert_array_equal(actual, expected)

    def test_exported_height_matches_selected_bank_trajectory(self):
        ground, z, rpm, heights, speeds, vertical, spins = small_bank()
        points = np.column_stack([np.full(9, 100), np.full(9, 400), np.linspace(90, 10, 9)])
        bounce = np.tile([1 / 9, 1 / 9, 0.1, 0.85, 0.05], (9, 1))
        camera = SimpleNamespace(get_cw=lambda: np.array([500., 0., 500.]))
        for hit, length in product(['bot_srv', 'top_fh_ts'], [7, 9]):
            with self.subTest(hit=hit, bank_length=length):
                fitted = fit_trajectory_colinear(
                    (200, 300), points, False, np.zeros(9, dtype=bool), hit, bounce,
                    matchCamera=camera, groundArr=ground[..., :length], vertArr=z[..., :length],
                    rpmArr=rpm[..., :length], height_space=heights, v_init_ground_space=speeds,
                    v_init_vert_space=vertical, spin_rate_space=spins, number_of_local_minima=5,
                    ram_allowed_in_gb=0.00001)
                parameters, fitted_ground, fitted_z, fitted_rpm, anchors, enough = fitted
                self.assertEqual(parameters.shape[-1], 5)
                height_indices = parameters[..., 3].astype(int)
                self.assertTrue(np.all(height_indices >= 2) if '_srv' in hit
                                else np.all(height_indices < 2))
                np.testing.assert_array_equal(heights[height_indices], fitted_z[..., 0])
                coordinates = calculate_trajectory_coordinates(fitted_ground, fitted_z, anchors, enough, False)
                observed, planes = make_empty_coord_and_plane_array(coordinates)
                selection = get_highest_prob(parameters, np.ones(parameters.shape[:2]), np.ones(parameters.shape[:2]))
                self.assertEqual(selection[1], int(np.argmax(parameters[0, :, -1])))
                out, columns = make_trajectory_arrays(
                    *selection, traj_coordArr=coordinates, coordArr=observed, planeArr=planes,
                    trajArr=parameters, groundArr_filt=fitted_ground, vertArr_filt=fitted_z,
                    rpmArr_filt=fitted_rpm, v_init_ground_space=speeds, v_init_vert_space=vertical,
                    spin_rate_space=spins, height_space=heights)
                expected_height = heights[height_indices[selection[0], selection[1]]]
                np.testing.assert_array_equal(out[:, columns.index('height0')], expected_height)
                np.testing.assert_allclose(out[0, columns.index('z_t')], expected_height * 8 / 0.3048, rtol=1e-6)

    def test_duplicate_weights_equal_pairwise_comparison(self):
        ground = np.array([[0, 1, 2], [0, 1, 2], [0, 2, 4], [0, np.nan, 2], [0, np.inf, 2]])
        z = np.tile([1, 2, 0], (len(ground), 1))
        with np.errstate(invalid='ignore'):
            equal_ground = np.abs(ground[:, None] - ground[None, :]).sum(axis=-1) == 0
            equal_z = np.abs(z[:, None] - z[None, :]).sum(axis=-1) == 0
        np.testing.assert_array_equal(_identical_trajectory_counts(ground, z), (equal_ground & equal_z).sum(axis=1))


if __name__ == '__main__':
    unittest.main()
