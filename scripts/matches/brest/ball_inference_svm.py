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


if __name__ == '__main__':

    conf = ut.read_config('brest')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']

    ### Load SVM for Inference ###
    svm_model_name = vid
    svm_dict_name = 'svm_proba_dict_2023-08-03'

    ballDict = ut.get_folder_struct('ball_svm')
    fitPath = os.path.join(ballDict['main'], svm_model_name, 'fitted_svms')
    
    svmPath = os.path.join(fitPath, svm_dict_name + '.pickle')

    import pickle

    with open(svmPath, 'rb') as f:
        svmDict = pickle.load(f)
    ##############
        
    ### Load Movement Contours ###
    cntSF = StreamFeatures.from_store(vid, sto, featnameList = ['movCnts']).squeeze()
    playBF = BoolFeatures.from_store(vid, sto, featnameList = ['isPlay']).squeeze()
    cntSFs = cntSF.conform(playBF.filter(playBF)).fillna(Contours([])).split()
    
    ### Make Contour Features into BallTable ###
    frmSF = StreamFeatures.from_store(vid, sto, featnameList = ['ballTable'])
    if frmSF.is_empty():
        hStms = StreamList.from_SFList(cntSFs, cap, target_color_space = 'hsv')
        gStms = StreamList.from_SFList(cntSFs, cap, target_color_space = 'grey')

        mulStms = StreamList.combine(hStms, gStms).attach(cntSFs)
        
        params = svmDict['feature_params']
        frmSFs = mulStms.extract_chunked_mp(ex.get_feature_frames, nwo, nch, **params)   
        
        frmSFs.save(vid, sto)
    else:
        frmSFs = frmSF.conform(playBF.filter(playBF)).split()
    
    ### Find Ball Candidates
    jntSFs = frmSFs.merge(cntSFs)
    
    params = {'svmDict': svmDict,
              'use_training_moments': True,
              'probability_threshold': None}
    
    candSFs = jntSFs.apply_whole_mp(sf.find_ball_candidates, nwo, **params)
    candSFs.save(vid, sto, compression = conf['compression_ball'])

    # Stream.from_SF(candSFs[0], cap).show()

