import json
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'ATMANpy', 'cli', 'pipeline_annotations.py')
spec = importlib.util.spec_from_file_location('pipeline_annotations_for_test', SOURCE)
annotations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotations)
import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.util_funcs as ut
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature


class PipelineAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='atman-annotation-tests-')
        os.makedirs(os.path.join(self.tmp.name, 'features_stored'))
        self.patch = patch.object(ut, 'mainDir', self.tmp.name)
        self.patch.start()
        with open(os.path.join(ROOT, 'examples', 'brest_first_4min', 'human_review.json')) as f:
            self.human = json.load(f)
        self.conf = {'video_id': self.human['video_id'], 'store_name': 'fresh_test',
                     'start_frame': 0, 'stop_frame': 5999}

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def save(self, feature):
        feature.save(self.conf['video_id'], self.conf['store_name'], verbose=False)

    def test_warmup_preserves_new_boundaries(self):
        self.save(BoolFeature({309: True, 310: True, 312: False}, name='isRally'))
        annotations.apply_warmup_policy(self.conf, self.human)
        result = annotations._load_feature(self.conf, 'isRally_warmup_removed')
        self.assertEqual(result.streamDict, {309: True, 310: True, 312: False})

    def test_identity_uses_only_new_court_frames(self):
        self.save(BoolFeature({30: True, 31: False, 5209: True}, name='isCourt'))
        self.save(BoolFeature({30: True, 5209: True}, name='isRally_sound'))
        annotations.apply_player_identity(self.conf, self.human)
        result = annotations._load_feature(self.conf, 'top_name_Flag')
        self.assertEqual(result.streamDict, {30: 'Joao Sousa', 5209: 'Brandon Nakashima'})
        annotations.apply_no_tiebreak(self.conf, self.human)
        self.assertTrue(StreamFeatures.from_store(self.conf['video_id'], self.conf['store_name'],
                        featnameList=['top_name_tb_Flag','top_hand_tb_Flag','bot_name_tb_Flag','bot_hand_tb_Flag']).is_empty())

    def test_serve_labels_follow_fresh_detection_indices(self):
        self.save(BoolFeature({308: True, 2468: True}, name='isRally_warmup_removed'))
        frame = pd.DataFrame({'event': ['STD', 'STD']}, index=pd.Index([308,2468], name='idx0'))
        ff.save_feature_frame(frame, 'trajTable', self.conf['video_id'], self.conf['store_name'], verbose=False)
        report = annotations.apply_serve_annotations(self.conf, self.human)
        out = annotations._load_feature(self.conf, 'srvFlag')
        self.assertEqual(out.streamDict, {308:'first_serve',2468:'let'})
        self.assertNotIn(301, out.streamDict)
        self.assertEqual(len(report['undetected_review_references']),6)

    def test_unreviewed_extra_serve_fails(self):
        self.save(BoolFeature({5800: True}, name='isRally_warmup_removed'))
        frame = pd.DataFrame({'event':['STD']},index=pd.Index([5800],name='idx0'))
        ff.save_feature_frame(frame,'trajTable',self.conf['video_id'],self.conf['store_name'],verbose=False)
        with self.assertRaisesRegex(ValueError,'additional human classification'):
            annotations.apply_serve_annotations(self.conf,self.human)

    def test_colinear_selects_one_fresh_hypothesis(self):
        frame=pd.DataFrame({'x_t':[1.,2.]},index=pd.Index([308,309],name='idx0'))
        ff.save_feature_frame(frame,'colinearTable_fair',self.conf['video_id'],self.conf['store_name'],verbose=False)
        annotations.select_colinear_variant(self.conf)
        loaded=ff.read_feature_frame('colinearTable',self.conf['video_id'],self.conf['store_name'],keep_only='last')
        self.assertEqual(loaded.index.tolist(),[308,309]);self.assertEqual(loaded.x_t.tolist(),[1.,2.])

    def test_score_cleanup_handles_missing_without_reference_ocr(self):
        self.save(BoolFeature({10:True,11:True,30:True,31:True},name='isRally_warmup_removed'))
        # Use literal CSV to retain missing OCR rows (feature .save drops None).
        for side in ('upper','lower'):
            frame=pd.DataFrame({'sb_games_'+side+'_Flag':[None,None,'0','0']},index=pd.Index([10,11,30,31],name='idx0'))
            ff.save_feature_frame(frame,'sb_games_'+side+'_Flag',self.conf['video_id'],self.conf['store_name'],verbose=False)
        report=annotations.apply_score_cleanup(self.conf)
        out=annotations._load_feature(self.conf,'sb_games_upper_cleaned_Flag')
        self.assertEqual(set(out.streamDict),{10,11,30,31});self.assertEqual({str(x) for x in out.streamDict.values()},{'0'})
        self.assertEqual(report['missing_scores_normalized_to_zero'],2)


if __name__ == '__main__':
    unittest.main()
