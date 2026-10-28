"""ATMAN delegates algorithms while retaining project-specific adapters."""
import importlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
try:
    from kenosha_hmm import GaussianHMM
except ModuleNotFoundError as error:
    if error.name != 'kenosha_hmm':
        raise
    GaussianHMM = None


@unittest.skipIf(GaussianHMM is None, 'Install ATMANpy[hmm] to test the optional HMM integration')
class HMMPackageIntegration(unittest.TestCase):
    def test_historical_namespaces_delegate(self):
        for name in ['HInit', 'HRest', 'HERest', 'HVite', 'high_level']:
            core = importlib.import_module('kenosha_hmm.' + name)
            for prefix in ['ATMANpy.models.HMM.', 'ATMANpy.funcs.HMM.']:
                self.assertIs(importlib.import_module(prefix + name), core)
        adapter = importlib.import_module('ATMANpy.models.HMM.TokenPass')
        self.assertIs(importlib.import_module('ATMANpy.funcs.HMM.TokenPass'), adapter)
        self.assertIs(adapter.infer, importlib.import_module('kenosha_hmm.TokenPass').infer)

    def test_tennis_postprocessing_stays_in_atman(self):
        module = importlib.import_module('ATMANpy.models.HMM.HPost')
        self.assertIs(importlib.import_module('ATMANpy.funcs.HMM.HPost'), module)
        self.assertTrue(hasattr(module, 'narrow_down_instantaneous_events'))

    def test_named_and_explicit_grammars(self):
        from ATMANpy.funcs.HMM.TokenPass import read_grammar
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'backus_naur_grammars';folder.mkdir()
            path = folder/'sample.txt';path.write_text('all ::= low high\n')
            with patch('ATMANpy.funcs.util_funcs.get_folder_struct', return_value={'auxil': tmp}):
                self.assertEqual(read_grammar('sample'), {'all': 'low high'})
                self.assertEqual(read_grammar(path), {'all': 'low high'})

    def test_original_csv_model_layout_round_trips(self):
        from ATMANpy.funcs.HMM.utils import save_hmm_params, read_hmm_params
        model = GaussianHMM.from_arrays([[0,1,0],[0,.8,.2],[0,0,0]],[[2]],[[[.5]]])
        with tempfile.TemporaryDirectory() as tmp:
            with patch('ATMANpy.funcs.util_funcs.get_folder_struct', return_value={'saved_models': tmp}):
                save_hmm_params({'example': model.as_legacy()}, 'test')
                restored = read_hmm_params('test')['example']
            np.testing.assert_array_equal(restored['A'], model.A)
            pd.testing.assert_frame_equal(restored['avg'], model.avg)
            pd.testing.assert_frame_equal(restored['cov'], model.cov)


if __name__ == '__main__':
    unittest.main()
