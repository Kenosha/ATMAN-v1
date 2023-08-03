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

START_FRAME = 0
STOP_FRAME = 100000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'svm_run'
NUM_WORKERS = 24
CHUNKSIZE = 1000

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE

if __name__ == '__main__':
    ### Use the Frames where we did find lines to calculate a background frame!
    # the nice thing is: Since the homography is constant, there is no need to warp!!!
    # We basically construct a NCDecapStream from where a homography has been found,
    # and calculate mean and std over all constituent frames.
    # This will constitute our background (for object detection).
    frameList = list(range(START_FRAME, STOP_FRAME + 1))
    
    ### Get the Homography
    HSF = ConstFeatures.from_store(VIDEO_ID, STORE_NAME, ['H'], frameList = frameList).squeeze()
    H = HSF.feature
    
    ### Get the Segmenting Boolean
    courtBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['isCourt']).squeeze()
    courtBFs = courtBF.filter(courtBF).split()
    courtStms = StreamList.from_SFList(courtBFs, cap)
    
    ### Build an NCStream which holds all frames from which we can sample a valid court frame
    ### for construction of background frames for each member of courtStms
    # TODO: Maybe widen the segmentation for this, so we sample frames only from what is truly play, and not
    # sequences when the court is being shown but the lights are different and there is no play? (weird changeovers? check frame 24000)
    # Stream(cap, START_FRAME, STOP_FRAME).get(24000).show()
    TRUNC_N = 1000 ### how many frames to truncate before and after, addressing the above issue      
    sampleStm = courtStms.widen(-TRUNC_N).join() 
    
    ### Now, for each courtStm, sample and calculate median and std
    WING_N = 2000 ### how many frames before start and after stop (respectively)
                    # can admissibly be sampled.
    SAMPLE_N = 1000 ### frames to sample for each background frame
    
    for courtStm in courtStms:
        ### construct subsample stream
        wideStm = courtStm.widen(WING_N)
        admissible_frames = [frm for frm in wideStm.get_framelist() if frm in sampleStm.frmNums]
        assert len(admissible_frames) >= SAMPLE_N, f'Too few frames in widened {courtStm}, consider increasing WING_N'
        subsampleStm = sampleStm.conform(admissible_frames).sample(SAMPLE_N)
        
        ### 
        subsampleDecStm = subsampleStm.decap()
        medFrm, stdFrm = subsampleDecStm.frames.median(), subsampleDecStm.frames.std()
        medFrm, stdFrm = medFrm.rename('median_backFrm'), stdFrm.rename('std_backFrm')
        
        # medFrm.show()
        # stdFrm.show()
        
        medFrm.save_like_feature(VIDEO_ID, STORE_NAME, start_frame = courtStm.start_frame, stop_frame = courtStm.stop_frame)
        stdFrm.save_like_feature(VIDEO_ID, STORE_NAME, start_frame = courtStm.start_frame, stop_frame = courtStm.stop_frame)
    
    