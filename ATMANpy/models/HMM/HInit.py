"""Compatibility alias for the standalone kenosha-hmm implementation."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('kenosha_hmm.HInit')
