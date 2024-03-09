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

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import get_width_height

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Visual3D import Scene, Camera, View

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D

def load(vid, sto):
    segBF = StreamFeature.from_store(vid, sto, featname = 'isValidFore') ### use this as a way of marking changeovers
    srvSF = StreamFeature.from_store(vid, sto, featname = 'srvFlag')
    return segBF, srvSF

def main(conf):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    width, height = get_width_height(cap)
    
    #### READ IN ####
    segBF, srvSF = load(vid, sto)

    srvSFs = srvSF.conform(segBF).filter(segBF).split()
    srvSFs = srvSFs.dropna()

    #### CHECK FOR INTEGRITY ###
    for seg_idx, sf in enumerate(srvSFs):
        serve_flags = list(sf.streamDict.values())
        ### split into consecutive serves
        serve_splits = ut.split_into_contiguous(serve_flags, offset = '')
        ### check if there are more than two consecutive serves from the same side.
        weird_splits = [(idx, flags) for idx, flags in enumerate(serve_splits) if len(flags) > 2]
        
        if len(weird_splits) > 0:
            print(f'Segment Index {seg_idx}: ')
            for rally_idx, flags in weird_splits:
                print(f'    Rally Index {rally_idx}: {len(flags)} consecutive {flags[0]}')

    srvSFs[3].streamDict
    
    Stream(cap, 41000, 41950).show(frame_counter = True)


if __name__ == '__main__':
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    # main(conf)


