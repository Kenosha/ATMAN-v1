# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:11:20 2024

@author: gianl
"""

import os
cwd = os.getcwd() ### current directory of this very __init__.py file

scripts = [file for file in os.listdir(cwd) if not file.startswith('__')]

for script in scripts:
    

from ATMANpy.scripts.pipeline_brest import (
    court_lines,
    court_roi,
    )
