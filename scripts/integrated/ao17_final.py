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
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCDecapStream, StreamList, MulStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

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
    grey_streamList = StreamList.from_SFList(courtSFs, cap, target_color_space = 'grey')

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
    del courtSF, courtSFs, boardSF
    
    try:
        medStms = StreamList.from_store('median_backFrm', VIDEO_ID, STORE_NAME, 
                                        cap_color_space = 'hsv', target_color_space = 'hsv')
        stdStms = StreamList.from_store('std_backFrm', VIDEO_ID, STORE_NAME, 
                                        cap_color_space = 'hsv', target_color_space = 'hsv')
    except FileNotFoundError:
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
        
        # medFrm = HSVFrame.from_csv(burn_vidPath, name = 'median_backFrm')
        # stdFrm = HSVFrame.from_csv(burn_vidPath, name = 'std_backFrm')
        
        for someSF in mySFs:
            start, stop = someSF.get_frames()[0], someSF.get_frames()[-1]
            medFrm.save_like_feature(VIDEO_ID, STORE_NAME, start_frame = start, stop_frame = stop)
            stdFrm.save_like_feature(VIDEO_ID, STORE_NAME, start_frame = start, stop_frame = stop)
            
        # ### save these frames as a stream! 
        # # naturally, right now, this is useless and makes things slower.
        # # however, we make it so that all frames that go into functions are read as Streams, which makes things more elegant.
        # # Also, down the line, we can have the background adapt and change 
        # # (like for example to reflect moving shadows of the roof or the stands)
        # for someSF in mySFs:
        #     medSF = StreamFeature({frame: medFrm for frame in someSF.get_frames()}, name = 'median_backFrm')
        #     stdSF = StreamFeature({frame: stdFrm for frame in someSF.get_frames()}, name = 'std_backFrm')
        #     medSF.save(VIDEO_ID, STORE_NAME, lossless = True) ### need lossless for these, since the margins are small
        #     stdSF.save(VIDEO_ID, STORE_NAME, lossless = True) ### need lossless for these, since the margins are small
        #     ### TODO: Lossless saving with FFV1 codec takes ages... try HEVC (aka H265) --> fourcc = HVC1
        #     ### TODO: Even better: have a PseudoStream class which, instead of a cap, has a single frame!
        #         ### to initialize it: save a single picture frame lossless, and read it as a stream.
        #         ### -> we will then automatically get a pseudostream.
    
        medStms = StreamList.from_store('median_backFrm', VIDEO_ID, STORE_NAME, 
                                        cap_color_space = 'hsv', target_color_space = 'hsv')
        stdStms = StreamList.from_store('std_backFrm', VIDEO_ID, STORE_NAME, 
                                        cap_color_space = 'hsv', target_color_space = 'hsv')
        

    ### Get Burn-In Mask
    featureName = 'foregroundMask'
    try:
        burnStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, 
                                         cap_color_space = 'bool', target_color_space = 'bool')
    except FileNotFoundError:
        hsvStms = StreamList.from_SFList(mySFs, cap, 
                                         cap_color_space = 'bgr', target_color_space = 'hsv')
        mulStms = StreamList.combine(hsvStms, medStms, stdStms)
        
        params = {'minimum_proportion': 0.0125,
                  'minimum_step_gain': 0.002}
        burnSFs = mulStms.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.get_burn_in_mask_iterative, **params)
    
        burnSFs.save(VIDEO_ID, STORE_NAME)
        burnStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, 
                                         cap_color_space = 'bool', target_color_space = 'bool')
    
    ### Get Shadow Mask
    featureName = 'shadowMask'
    try: 
        shadStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, 
                                         cap_color_space = 'bool', target_color_space = 'bool')
    except FileNotFoundError:
        hsvStms = StreamList.from_SFList(mySFs, cap, 
                                         cap_color_space = 'bgr', target_color_space = 'hsv')
        mulStms = StreamList.combine(hsvStms, medStms)
        
        params = {'h_threshold': 5,
                  's_threshold': 10,
                  'v_threshold_lower': 0.75,
                  'v_threshold_upper': 0.97}
        shadSFs = mulStms.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.get_shadow_mask, **params)
    
        shadSFs.save(VIDEO_ID, STORE_NAME)
        shadStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, 
                                         cap_color_space = 'bool', target_color_space = 'bool')
    
    
    # a) Get some constant shapes: the initial ROIs...
    params = {'width_factor': 1,
              'depth_factor': 1,
              'add_depth_factor_bottom': 0.3}
    from ATMANpy.funcs.player_funcs import get_service_ROIs
    initial_ROI_lower, initial_ROI_upper = get_service_ROIs(schemaLines, **params)

    # ... and the bounding lines above and below.
    from ATMANpy.funcs.player_funcs import get_half_court_Lines
    halfcourtLine_lower, halfcourtLine_upper = get_half_court_Lines(schemaLines)

    # b) extract ROIs and players simultaneously, in cached fashion.
    params = {'player_tracking_scale_factor': 1.5,
              'bot_min_bounding_height': 200,
              'bot_min_bounding_width': 80,
              'top_min_bounding_height': 150,
              'top_min_bounding_width': 60,
              'initial_ROI_upper': initial_ROI_upper,
              'initial_ROI_lower': initial_ROI_lower,
              'halfcourtLine_upper': halfcourtLine_upper,
              'halfcourtLine_lower': halfcourtLine_lower}
    
    mySFs = mySFs.add('topPlayer', None).add('botPlayer', None)
    greyCtms = StreamList.from_SFList(mySFs, cap, cache_size = 1, name = 'gFrm',
                                     cap_color_space = 'bgr', target_color_space = 'grey')
    mulStms = StreamList.combine(greyCtms, burnStms, shadStms)
    
    plSFs = mulStms[2:4].extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.find_both_players_smart, **params)
    
    plSFs[1]['topROIMask'].save(VIDEO_ID, STORE_NAME)
    
    plSFs.save(VIDEO_ID, STORE_NAME)
    
    # Stream.from_SF(plSFs[5], cap).show()
    
    # CHUNKSIZE = 750
    
    # for el in mulStms[:4].chunk_generator(chunksize = CHUNKSIZE):
    #     print(el.length())
    
    # plSFs
    
    # testStm = mulStms[2]
    # # chkSFs = testStm.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.find_both_players_smart, **params)
    # chkSF = chkSFs[0]
    # Stream.from_SF(chkSF, cap).show()
    
    # # for 
    # # mulStms[2].stop_frame - mulStms[2].start_frame
    
    # testStm[1].size(unit = 'MB')
    # testStm[1].decap().size(unit = 'MB')
    
    # testStm[0].features
    
    
    # for el in testStm.chunk_generator():
    #     print(el[1].start_frame)
    #     print(el[1].stop_frame)
    #     print(el.features)
        
    # from ATMANpy.funcs.cv_funcs import show_continuous
    # for el in testStm.chunk_generator():
    #     for frame in el[1].frames:
    #         show_continuous(frame.array)
    
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
    
    