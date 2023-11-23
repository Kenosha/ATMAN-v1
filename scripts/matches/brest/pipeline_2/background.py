# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:17:42 2023

@author: gianl
"""


import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

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

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame, MaskFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

if __name__ == '__main__':
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    
    ### Use the Frames where we did find lines to calculate a background frame!
    # the nice thing is: Since the homography is constant, there is no need to warp!!!
    # We basically construct a NCDecapStream from where a homography has been found,
    # and calculate mean and std over all constituent frames.
    # This will constitute our background (for object detection).
    
    ### Get the Homography
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    H = HSF.feature
    
    ### Get the Segmenting Boolean
    courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
    courtBFs = courtBF.filter(courtBF).split()
    courtStms = StreamList.from_SFList(courtBFs, cap)
    
    ### Build an NCStream which holds all frames from which we can sample a valid court frame
    ### for construction of background frames for each member of courtStms
    # TODO: Maybe widen the segmentation for this, so we sample frames only from what is truly play, and not
    # sequences when the court is being shown but the lights are different and there is no play? (weird changeovers? check frame 24000)
    # Stream(cap, START_FRAME, STOP_FRAME).get(24000).show()
    TRUNC_N = 1000 ### how many frames to truncate before and after, addressing the above issue      
    SAMPLE_N = 1000 ### frames to sample for each background frame
    
    for courtStm in courtStms:
        tightStm = courtStm.widen(-TRUNC_N) 
    
        if tightStm.length() < SAMPLE_N:
            print('Too few frames in courtStm, consider reducing TRUNC_N')
            continue
        
        sampledStm = tightStm.sample(SAMPLE_N)
        sampledFrms = sampledStm.decap().frames
        
        medFrm, stdFrm = sampledFrms.median(), sampledFrms.std()
        medFrm, stdFrm = medFrm.rename('median_backFrm'), stdFrm.rename('std_backFrm')
        
        medFrm.save_like_feature(vid, sto, start_frame = courtStm.start_frame, stop_frame = courtStm.stop_frame)
        stdFrm.save_like_feature(vid, sto, start_frame = courtStm.start_frame, stop_frame = courtStm.stop_frame)
    
    
    