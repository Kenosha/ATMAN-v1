"""Standalone HMM decoding with ATMAN's historical grammar-name adapter."""
import os
from kenosha_hmm.TokenPass import *
from kenosha_hmm.TokenPass import read_grammar as _read_grammar


def read_grammar(grammarStr):
    """Accept an explicit path, or resolve an ATMAN bundled grammar by name."""
    candidate = os.fspath(grammarStr)
    if os.path.isfile(candidate):
        return _read_grammar(candidate)
    from ATMANpy.funcs.util_funcs import get_folder_struct
    folder = os.path.join(get_folder_struct('ATMANpy')['auxil'], 'backus_naur_grammars')
    return _read_grammar(os.path.join(folder, candidate + '.txt'))
