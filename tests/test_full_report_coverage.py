"""Coverage must include active frames absent from serialized features."""

import importlib.util
import os
import unittest


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'scripts', 'report_full_pipeline.py')
SPEC = importlib.util.spec_from_file_location('full_pipeline_report_coverage', SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ReportCoverageTests(unittest.TestCase):
    def test_dropping_none_rows_preserves_active_coverage(self):
        active = {10: True, 11: True, 12: True, 13: True, 14: False}
        original = {10: object(), 11: None, 13: object(), 14: object()}
        serialized = {frame: value for frame, value in original.items() if value is not None}
        dense = REPORT.presence_coverage(original, active, 'isRally')
        sparse = REPORT.presence_coverage(serialized, active, 'isRally')
        self.assertEqual(dense['stored_frames'], 4)
        self.assertEqual(sparse['stored_frames'], 3)
        self.assertEqual(sparse['stored_null_frames'], 0)
        for coverage in (dense, sparse):
            self.assertEqual(coverage['active_frames'], 4)
            self.assertEqual(coverage['present_active_frames'], 2)
            self.assertEqual(coverage['missing_active_frames'], 2)
            self.assertEqual(coverage['present_outside_active_frames'], 1)
            self.assertEqual(coverage['active_coverage_fraction'], 0.5)

    def test_player_edge_frames_are_counted_missing(self):
        active = {frame: True for frame in range(2166)}
        present = {frame: object() for frame in range(6, 2159)}
        coverage = REPORT.presence_coverage(present, active, 'isRally_sound')
        self.assertEqual(coverage['reference_feature'], 'isRally_sound')
        self.assertEqual(coverage['present_active_frames'], 2153)
        self.assertEqual(coverage['missing_active_frames'], 13)
        self.assertEqual(coverage['active_coverage_fraction'], 2153 / 2166)

    def test_no_active_frames_does_not_claim_complete_coverage(self):
        coverage = REPORT.presence_coverage({1: object()}, {1: False}, 'isRally')
        self.assertEqual(coverage['active_frames'], 0)
        self.assertEqual(coverage['missing_active_frames'], 0)
        self.assertEqual(coverage['present_outside_active_frames'], 1)
        self.assertIsNone(coverage['active_coverage_fraction'])


if __name__ == '__main__':
    unittest.main()
