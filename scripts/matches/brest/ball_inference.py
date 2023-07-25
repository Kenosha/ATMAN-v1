# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:26:01 2023

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

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

START_FRAME = 0
STOP_FRAME = 100000
VIDEO_ID = '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano'
STORE_NAME = 'svm_run'
NUM_WORKERS = 24
CHUNKSIZE = 5000

cap = get_video(VIDEO_ID)

width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
NATURAL_SCALE = width / 1920
ADD_SCALE = 1 
SCALE_FACTOR = NATURAL_SCALE * ADD_SCALE

if __name__ == '__main__':
    
    cntSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['movCnts']).squeeze()
    ballkidBF = BoolFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['ballkidBool']).squeeze()
    playBF = ballkidBF.neg()
    
    cntSFs = cntSF.conform(playBF.filter(playBF)).fillna(Contours([])).split()
    
    ### Get the BallTable, for SVM inference on contour features
    frmSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['ballTable'])
    if frmSF.is_empty():
        hStms = StreamList.from_SFList(cntSFs, cap, target_color_space = 'hsv')
        gStms = StreamList.from_SFList(cntSFs, cap, target_color_space = 'grey')

        mulStms = StreamList.combine(hStms, gStms).attach(cntSFs)
        
        params = {'zoom_factor': round(10 / SCALE_FACTOR), 
                  'pixel_margin': max(round(3 * SCALE_FACTOR), 1),
                  'ksize': 3}
        frmSFs = mulStms.extract_chunked_mp(ex.get_feature_frames, NUM_WORKERS, CHUNKSIZE, **params)   
        
        frmSFs.save(VIDEO_ID, STORE_NAME)
    else:
        frmSFs = frmSF.conform(playBF.filter(playBF)).split()
    
    ### Load an SVM to run inference
    svm_model_name = VIDEO_ID
    svm_dict_name = 'svm_proba_dict_2023-07-24'

    ballDict = ut.get_folder_struct('ball_svm')
    fitPath = os.path.join(ballDict['main'], svm_model_name, 'fitted_svms')
    
    svmPath = os.path.join(fitPath, svm_dict_name + '.pickle')

    import pickle

    with open(svmPath, 'rb') as f:
        svmDict = pickle.load(f)
        
    ### Find Ball Candidates
    jntSFs = frmSFs.merge(cntSFs)
    
    params = {'svmDict': svmDict,
              'use_training_moments': True,
              'probability_threshold': 0.05}
    
    ### Testing
    locals().update(params)
    contSF = jntSFs[0]
    
    candSF = sf.find_ball_candidates(contSF, **params)
    Stream.from_SF(candSF, cap).show()
    
    
    candSFs = jntSFs.apply_whole(sf.find_ball_candidates, **params)

    candStms = StreamList.from_SFList(candSFs, cap)
    candStms[0].show()
    
    
    
    
    
    candSFs = jntSFs.apply_whole(sf.find_ball_candidates, **params)

    candStms = StreamList.from_SFList(candSFs, cap)
    candStms[0].show()

    cntStms = StreamList.from_SFList(cntSFs, cap)
    cntStms[0].show(ms_between_frames = 50)

    ### 
    contSF = jntSFs[0]
    locals().update(params)


    candStms[0].get(300, put = False).show()
