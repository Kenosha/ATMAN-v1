# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:45:37 2023

@author: gianl
"""

import os
import time

import numpy as np

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex

from ATMANpy.types.Streams import Stream, CachedStream, NCDecapStream, StreamList
from ATMANpy.types.Features import StreamFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

# from ATMANpy.types.Lines import CourtLines
# from ATMANpy.funcs.line_funcs import fit_frame_to_lines
# schemaLines = CourtLines.make().add_singles_netline()
# schemaFrm = fit_frame_to_lines(schemaLines)

if __name__ == '__main__':
    ### PARAMETERS AND GLOBALS
    VIDEO_ID = 'KTCDxjJvs2U'
    cap = get_video(VIDEO_ID)
    
    START_FRAME = 800
    # RUN_LENGTH = 290000 ### TODO: Generalize this to take into account length of stream
    RUN_LENGTH = 80000
    # RUN_LENGTH = 4000
    # START_FRAME = 28000
    # RUN_LENGTH = 600
    stop_frame = START_FRAME + RUN_LENGTH
    
    NUM_WORKERS = 24
    CHUNKSIZE = 100
    
    STORE_NAME = 'test'
    
    ### Make folders for feature storage
    featPath = ut.get_folder_struct('features_stored')['main']
    feat_vidPath = os.path.join(featPath, VIDEO_ID)
    try:
        os.mkdir(feat_vidPath)
    except FileExistsError:
        pass
    
    storePath = os.path.join(feat_vidPath, STORE_NAME)
    try:
        os.mkdir(storePath)
    except FileExistsError:
        pass
    
    ### Make folders for burn in data
    burnPath = ut.get_folder_struct('burn_in')['main']
    burn_vidPath = os.path.join(burnPath, VIDEO_ID)
    try:
        os.mkdir(burn_vidPath)
    except FileExistsError:
        pass
    
    
    ### Read in features which exist
    HSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['H']).squeeze()
    trackSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['trackFlag']).squeeze()
    boardSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['scoreBoard']).squeeze()
    
    # HSF = StreamFeatures({})
    # trackSF = StreamFeatures({})
    
    if HSF.is_empty() and trackSF.is_empty():
        ### Initialize Empty Streams
        emptyCtm = CachedStream(cap, START_FRAME, stop_frame)
        emptyStm = Stream(cap, START_FRAME, stop_frame)
        
        ### Track the courtlines
        courtSF = emptyCtm.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.track_courtlines)
        
        ### tighten the frame at which the match likely starts:
        notBF = courtSF['trackFlag'] != 'Not Found'
        first_frame, _ = courtSF.filter(notBF).first()
        
        ### reestablish the starting frame
        emptyCtm = CachedStream(cap, first_frame, stop_frame)
        emptyStm = Stream(cap, first_frame, stop_frame)
        courtSF = courtSF.conform(list(range(first_frame, stop_frame + 1)))
        
        print('Situation before first redo...')
        print(courtSF['trackFlag'].value_counts())
        
        ### redo frames that were not found
        redoBF = sf.get_redo_frames(courtSF)
        print(f'Re-attempting {redoBF.true()} frames...')
        redoNCD = NCDecapStream.from_bool(redoBF, cap, color_space = 'grey')
        redoNCD = redoNCD.attach(courtSF)
        redoSF = redoNCD.extract_mp(NUM_WORKERS, ex.redo_not_found)
        
        courtSF = courtSF.replace(redoSF)
        
        print('Situation before failure fix...')
        print(courtSF['trackFlag'].value_counts())
        
        ### try to fix failures
        redoBF = courtSF['trackFlag'] == 'Failed'
        print(f'Re-attempting {redoBF.true()} frames...')
        redoNCD = NCDecapStream.from_bool(redoBF, cap, color_space = 'grey')
        redoNCD = redoNCD.attach(courtSF)
        redoSF = redoNCD.extract(ex.iterative_redo)
        
        courtSF = courtSF.replace(redoSF)
        
        print('Situation before hole plugging...')
        print(courtSF['trackFlag'].value_counts())
        
        ### find holes in homography
        notBF = courtSF['trackFlag'] != 'Not Found'
        redoBF = sf.get_hole_frames(courtSF.filter(notBF))
        print(f'Re-attempting {redoBF.true()} frames...')
        redoNCD = NCDecapStream.from_bool(redoBF, cap, color_space = 'grey')
        redoNCD = redoNCD.attach(courtSF)
        redoSF = redoNCD.extract(ex.redo_not_found)        

        courtSF = courtSF.replace(redoSF)
    
        smallSF = courtSF[['H', 'courtLines']].dropna()
        sfList = smallSF.split()
        ### Correct Homographies
        cornerList = sfList.apply(sa.get_court_corner_points)
        regionList = cornerList.apply_whole(sf.median_corner_regions)
        mergeList = cornerList.merge(regionList)
    
        tlList = mergeList.apply(sa.correct_point, assignArgs = ['tl', 'tlReg'], assignVals = ['tl', 'tl_corr'])
        trList = mergeList.apply(sa.correct_point, assignArgs = ['tr', 'trReg'], assignVals = ['tr', 'tr_corr'])
        blList = mergeList.apply(sa.correct_point, assignArgs = ['bl', 'blReg'], assignVals = ['bl', 'bl_corr'])
        brList = mergeList.apply(sa.correct_point, assignArgs = ['br', 'brReg'], assignVals = ['br', 'br_corr'])
        correctList = tlList.merge(trList).merge(blList).merge(brList)
    
        ### calculate corrected homography from corner points
        correctSF = correctList.concatenate()
        HSF = correctSF.apply(sa.calculate_homography)
        
        ### update the track flag
        correctBool = correctSF[['tl_corr', 'tr_corr', 'bl_corr', 'br_corr']].to_bool().any()
        correctBool = correctBool.rename('corners_corrected')
        correctBool = correctBool.conform(courtSF).fillna(False)
        def replace_trackflag(trackFlag):
            trackFlag = 'Corner Corrected'
            return trackFlag
        trackSF = courtSF['trackFlag'].apply(replace_trackflag, where = correctBool)
        
        ### Save
        HSF.save(VIDEO_ID, STORE_NAME)
        trackSF.save(VIDEO_ID, STORE_NAME)

    ### Now that we have courtlines, we can work on segments
    courtSF = HSF.merge(HSF.apply(ex.find_courtlines))
    courtSFs = courtSF.split()
    grey_streamList = StreamList.from_SFList(courtSFs, cap, color_space = 'grey')

    if boardSF.is_empty():
            
        ### Find Scoreboard
        params = {'sobel_threshold': 30,
                  'slope_threshold': 0.0001,
                  'merge_lines': False,
                  'min_line_length_after_merge': 10,
                  'min_lines': 1}
        
        boardSFs = grey_streamList.extract_chunked_mp(NUM_WORKERS, ex.find_scoreboard, **params)
        boardSF = boardSFs.concatenate(axis = 0)
        boardSF.save(VIDEO_ID, STORE_NAME)
        
        mySFs = courtSFs.merge(boardSFs)
        
        streamList = StreamList.from_SFList(mySFs, cap)
        
    mySFs = courtSFs.merge(boardSF.split())
    
    try:
        medFrm = HSVFrame.from_csv(burn_vidPath, name = 'median')
        stdFrm = HSVFrame.from_csv(burn_vidPath, name = 'std')
    except OSError:
        ### BURN IN
        # collect a set of courtlines and the corresponding frames (HSV!)
        # separate out the 'weird' homographies (i think we might have done this already with the corner correction)
        # warp the frame back onto the schema
        # calculate median and pixel deviation across these warped frames for every channel
        # save median and deviation pixel frames.
        normBool = sf.get_normal_bool(courtSF) ### apparently, the burn in algo is much stricter than our corners...
        chosenBool = normBool.filter(normBool).random_choice(how_many = 1000) ### choose 1000 out of the 'normal' frames
        hsvNCD = NCDecapStream.from_bool(chosenBool, cap, color_space = 'hsv')
        hsvNCD = hsvNCD.attach(courtSF)
        wrpSF = hsvNCD.extract_mp(NUM_WORKERS, ex.courtlines_to_schema)
        
        medFrm, stdFrm = sf.get_median_and_deviation_frame(wrpSF)
        
        medFrm.to_csv(burn_vidPath)
        stdFrm.to_csv(burn_vidPath)
    
    ### Get Burn-In Mask
    featureName = 'foregroundMask'
    burnStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, color_space = 'grey')
    
    if burnStms.length() == 0:
        ### Calculate and Save Burn-In Mask
        params = {'medFrm': medFrm,
                  'stdFrm': stdFrm,
                  'minimum_proportion': 0.0125,
                  'minimum_step_gain': 0.002}
        
        hsv_streamList = StreamList.from_SFList(mySFs, cap, color_space = 'hsv')
        burnSFs = hsv_streamList.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.get_burn_in_mask_iterative, **params)
        
        burnSFs.save(VIDEO_ID, STORE_NAME)
        burnStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, color_space = 'grey')
    
    # # fileName = 'from_1024_to_1791.avi'
    # # featureName = 'foregroundMask'
    # # videoName = VIDEO_ID
    # # storeName = STORE_NAME
    
    # # burnStm = Stream.from_store(fileName, featureName, videoName, storeName, color_space = 'grey')
    
    # # burnStm.show()
    
    # # burnDec = burnStm.decap()
    
    # # burnStm[1792]
    # burnSF = hsv_
    
    # burnSFs = hsv_streamList[:1].extract_chunked_threaded(NUM_WORKERS, ex.get_burn_in_mask_iterative, **params)
    
    # burnSF = hsv_streamList.extract_chunked_mp(NUM_WORKERS, ex.get_burn_in_mask_iterative)
    # burnSF = hsv_streamList[0].extract(ex.get_burn_in_mask_iterative, **params)
    
    ###### Next Up ######
    
    ### (line_normalization.py)
    # -> i think with our corner correction, we are good here!    
    
    ### burn_in.py
    # Burn In (medFrm, stdFrm), Shadow
    
    ### (feature_saving_big_run.py)
    # Scoreboard Detection
    # Player Tracking with RoIs
    # Ball Contours
    # Track Ball Particles
    # Smoothe Particles
    # Choose Ball Contour
    
    ### (ballboy_detection_big_run/multiframe.py)
    # ballboy detection 
    # ballboy segmentation
    
    ### (join_features_and_labels_after_redos.py)
    # smart_interpolate_ball_contours
    
    ### (testing data evaluation II)
    # make inference on points.
    # find 3D players, 3D ball
    
    