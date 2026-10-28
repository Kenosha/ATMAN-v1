"""Preserve the historical ATMAN HMM import location."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('ATMANpy.models.HMM.HVite')
