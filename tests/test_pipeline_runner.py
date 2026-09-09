"""A resumed run must not silently use changed configuration or missing data."""
import contextlib
import io
import json
import os
import tempfile
import unittest

from ATMANpy.cli import pipeline


class PipelineRunnerTest(unittest.TestCase):
    def prepared_run(self, root):
        conf = dict(data_root=os.path.join(root, 'data'), video_id='example', store_name='fresh', seed=1)
        feature = os.path.join(conf['data_root'], 'features_stored', 'example', 'fresh', 'H')
        os.makedirs(feature)
        with open(os.path.join(feature, 'matrix.txt'), 'w') as handle:
            handle.write('abc')
        os.makedirs(os.path.join(root, 'stages'))
        pipeline.write_json(os.path.join(root, 'config.json'), conf)
        for stage in pipeline.STAGES:
            pipeline.write_json(os.path.join(root, 'stages', stage + '.json'),
                                dict(status='complete', configuration=conf,
                                     output_features=pipeline.files_in_store(conf)))
        return conf, os.path.join(feature, 'matrix.txt')

    def test_resume_accepts_intact_outputs_and_rejects_missing_feature(self):
        with tempfile.TemporaryDirectory() as root:
            conf, path = self.prepared_run(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(['--run-dir', root, '--all', '--resume']), 0)
            os.remove(path)
            with self.assertRaisesRegex(ValueError, 'missing after court_lines'):
                pipeline.main(['--run-dir', root, '--all', '--resume'])

    def test_changed_configuration_cannot_reuse_completed_stages(self):
        with tempfile.TemporaryDirectory() as root:
            conf, _ = self.prepared_run(root)
            conf['seed'] = 2
            pipeline.write_json(os.path.join(root, 'config.json'), conf)
            with self.assertRaisesRegex(ValueError, 'Configuration changed'):
                pipeline.main(['--run-dir', root, '--all', '--resume'])


if __name__ == '__main__':
    unittest.main()
