# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:11:20 2024

@author: gianl
"""

import os
import importlib

_directory = os.path.dirname(__file__)
_name = os.path.basename(_directory)
_scripts = [file for file in os.listdir(_directory) if not file.startswith('__') and file.endswith('.py')]
_other_loads = ['parameter_files', 'config_files']

for script in _scripts:
    script_name = script[:-3]
    globals()[script_name] = importlib.import_module('.' + script_name, __package__)

for load in _other_loads:
    globals()[load] = importlib.import_module('.' + load, __package__)

del script, script_name, load
del os, importlib
