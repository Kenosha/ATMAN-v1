# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 02:41:55 2024

@author: gianl
"""


import os
import importlib

_directory = os.path.dirname(__file__)
_name = os.path.basename(_directory)
_scripts = [file for file in os.listdir(_directory) if not file.startswith('__') and file.endswith('.py')]

for script in _scripts:
    script_name = script[:-3]
    globals()[script_name] = importlib.import_module('.' + script_name, __package__)

del script, script_name
del os, importlib
