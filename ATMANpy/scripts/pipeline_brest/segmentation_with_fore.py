# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:26:10 2023

@author: gianl
"""


import sys
import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import join_video_captures

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

def load(vid, sto):
    burnStms = StreamList.from_store('foregroundMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask')
    courtroiSF = ConstFeature.from_store(vid, sto, featname = 'courtroiReg', featType = Region)
    return burnStms, courtroiSF

def save(outBF, vid, sto):
    outBF = outBF.rename('isValidFore')
    outBF.save(vid, sto)

def main(conf, threshold_params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']

    burnStms, courtroiSF = load(vid, sto)
    
    courtROI = courtroiSF.feature
    litSFs = burnStms.extract_chunked_mp(ex.black_comparison, nwo, nch, ROI = courtROI)

    # a) kill below threshold
    validBFs = litSFs.apply(sa.below_threshold, threshold = threshold_params['ratio_threshold']).to_bool()
    
    # b) keep the longer ones
    outBFs = []
    for validBF in validBFs:
        subBFs = validBF.filter(validBF).split() 
        longBFs = SFList([bf for bf in subBFs if bf.length() > threshold_params['length_threshold']])
        longBF = longBFs.concatenate()
        outBFs.append(longBF)
    outBF = SFList(outBFs).concatenate_and_plug().to_bool().fillna(False)
    
    save(outBF, vid, sto)

if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import read_config   
    conf = read_config('brest', config_name = '_config_pipe2')
    
    threshold_params = {'ratio_threshold': 0.2, 
                        'length_threshold': 100}
    
    main(conf, threshold_params)
        
        