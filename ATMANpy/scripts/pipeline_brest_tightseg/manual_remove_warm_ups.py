
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:34:47 2024

@author: gianl
"""

import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures


def load(vid, sto):
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally') ### use this as a way of marking changeovers
    return segBF

def save(vid, sto, outBF):
    outBF.rename('isRally_warmup_removed').save(vid, sto)

def main(conf, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = conf['cap']

    segBF = load(vid, sto)

    segBFs = segBF.filter(segBF).split()

    enhanced_rally_dict = {k: v for k, v in segBF.streamDict.items()}
    
    is_warmup = True
    idx_to_show = 0
    while is_warmup:
        bf = segBFs[idx_to_show]
        Stream.from_SF(bf, cap).show()
        inp = input('Is this still warm-up? y/n: ')

        if inp == 'y':
            updDict = {k: False for k in bf.streamDict}
            enhanced_rally_dict.update(updDict)
            is_warmup = True
            idx_to_show += 1
        elif inp == 'n':
            is_warmup = False
        else:
            continue
        
    outBF = BoolFeature(enhanced_rally_dict, name = 'isRally_warmup_removed')
    outBFs = outBF.filter(outBF).split()

    num_removed = segBFs.length() - outBFs.length()
    print(f'Removed {num_removed} warm-up rallys!')
    save(vid, sto, outBF)


if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[1])

    name = ppl._name
    main(conf, name = name)