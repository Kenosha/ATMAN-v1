# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 20:20:19 2024

@author: gianl
"""


import ATMANpy.scripts.pipeline_brest as ppl
import ATMANpy.funcs.pipeline_funcs as ppl_fs

from ATMANpy.funcs.util_funcs import get_return_names

loadDict, saveDict = ppl_fs.get_load_and_save_dicts(ppl)
depDict = ppl_fs.get_dependency_dict(loadDict, saveDict)

####### Make a Dependency Graph #########
seq_orderDict = ppl_fs.get_sequential_ordering(depDict)
lat_orderDict = ppl_fs.get_lateral_ordering(seq_orderDict, depDict)

### Join Sequential and Lateral Ordering for the Positions
posDict = {s: (lat_orderDict[s], -seq_orderDict[s]) for s in lat_orderDict.keys()}

ppl_fs.make_digraph(depDict, posDict, saveName = ppl._name)








