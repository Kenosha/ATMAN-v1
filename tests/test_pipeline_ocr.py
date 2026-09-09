"""OCR runtime selection and explicit local whitelist configuration."""
import importlib.util
import os
import shlex
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

annotations = load_source('pipeline_annotations_ocr_test', 'ATMANpy/cli/pipeline_annotations.py')
score_ocr = load_source('pipeline_score_ocr_test', 'ATMANpy/scripts/pipeline_brest_tightseg/score_detection_testing.py')


class PipelineOCRTests(unittest.TestCase):
    def fake_pytesseract(self):
        return types.SimpleNamespace(pytesseract=types.SimpleNamespace(tesseract_cmd='unchanged'),
                                     get_tesseract_version=lambda: '5.test')

    def test_explicit_config_path_is_one_argument_even_with_spaces(self):
        path = os.path.join('/tmp', 'ocr configs', 'digits_tennis')
        config = score_ocr.get_ocr_config('digits_tennis', {'ocr_configs': {'digits_tennis': path}})
        self.assertEqual(shlex.split(config), ['--psm', '6', path])
        self.assertEqual(score_ocr.get_ocr_config('digits_tennis', {}), '--psm 6 digits_tennis')

    def test_system_fallback_keeps_system_data_and_creates_local_configs(self):
        fake = self.fake_pytesseract()
        with tempfile.TemporaryDirectory(prefix='atman-ocr-test-') as root:
            with patch.dict(sys.modules, {'pytesseract': fake}), \
                 patch.object(annotations.shutil, 'which', return_value='/usr/bin/tesseract'), \
                 patch.dict(os.environ, {'TESSDATA_PREFIX': '/existing/tessdata', 'LD_LIBRARY_PATH': '/existing/lib'}):
                result = annotations.configure_ocr(root)
                self.assertEqual(result['mode'], 'system')
                self.assertEqual(fake.pytesseract.tesseract_cmd, '/usr/bin/tesseract')
                self.assertEqual(os.environ['TESSDATA_PREFIX'], '/existing/tessdata')
                self.assertEqual(os.environ['LD_LIBRARY_PATH'], '/existing/lib')
                with open(result['configs']['digits_tennis']) as handle:
                    self.assertEqual(handle.read(), 'tessedit_char_whitelist 0123456789AD\n')
                self.assertTrue(all(os.path.commonpath([root, value]) == root for value in result['configs'].values()))

    def test_portable_runtime_is_used_without_looking_for_system_binary(self):
        fake = self.fake_pytesseract()
        with tempfile.TemporaryDirectory(prefix='atman-ocr-test-') as root:
            base = os.path.join(root, 'root')
            binary = os.path.join(base, 'usr', 'bin', 'tesseract')
            data = os.path.join(base, 'usr', 'share', 'tesseract-ocr', '5', 'tessdata')
            os.makedirs(os.path.dirname(binary)); os.makedirs(data)
            open(binary, 'w').close(); open(os.path.join(data, 'eng.traineddata'), 'w').close()
            with patch.dict(sys.modules, {'pytesseract': fake}), \
                 patch.object(annotations.shutil, 'which') as which, patch.dict(os.environ, {}, clear=True):
                result = annotations.configure_ocr(root)
                which.assert_not_called()
                self.assertEqual(result['mode'], 'portable')
                self.assertEqual(fake.pytesseract.tesseract_cmd, binary)
                self.assertEqual(os.environ['TESSDATA_PREFIX'], data)

    def test_incomplete_portable_runtime_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory(prefix='atman-ocr-test-') as root:
            os.makedirs(os.path.join(root, 'root'))
            with patch.dict(sys.modules, {'pytesseract': self.fake_pytesseract()}), \
                 patch.object(annotations.shutil, 'which') as which:
                with self.assertRaisesRegex(FileNotFoundError, 'Incomplete portable'):
                    annotations.configure_ocr(root)
                which.assert_not_called()


if __name__ == '__main__':
    unittest.main()
