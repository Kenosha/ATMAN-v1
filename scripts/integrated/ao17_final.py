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

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream
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
    # RUN_LENGTH = 1000
    stop_frame = START_FRAME + RUN_LENGTH
    
    NUM_WORKERS = 24
    CHUNKSIZE = 1000
    
    STORE_NAME = 'test_downscale'
    
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
    
    ### Read in features which exist
    HSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['H']).squeeze()
    trackSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['trackFlag']).squeeze()
    # HSF = StreamFeatures({})
    # trackSF = StreamFeatures({})
    
    if HSF.is_empty() and trackSF.is_empty(): ### Create HSF and trackSF
        SCALE_FACTOR = 2/3 ### downscale from full hd to hd 

        ### Initialize Empty Streams
        emptyCtm = CachedStream(cap, START_FRAME, stop_frame, 
                                scale_factor = SCALE_FACTOR,
                                cap_color_space = 'bgr', target_color_space = 'grey')
        
        WIDTH_OF_LINE_AT_FULL_HD = 8
        WIDTH_OF_LINE = WIDTH_OF_LINE_AT_FULL_HD * SCALE_FACTOR
        params = {'width_of_line': WIDTH_OF_LINE, 
                  'int_diff_thresh': 30,
                  'white_thresh': 150,
                  'rho': 3, 'theta': np.pi / 180, 'min_votes': int(500 * SCALE_FACTOR),
                  'minLineLength': WIDTH_OF_LINE + 1, 'maxLineGap': 1,
                  'merge_threshold': 2 * WIDTH_OF_LINE, 
                  'filter_margin': 40 * SCALE_FACTOR,
                  'only_225': True}
        
        courtSF = emptyCtm.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.track_courtlines, **params)
        
        ### tighten the frame at which the match likely starts:
        notBF = courtSF['trackFlag'] != 'Not Found'
        first_frame, _ = courtSF.filter(notBF).first()
        
        ### reestablish the starting frame
        courtSF = courtSF.conform(list(range(first_frame, stop_frame + 1)))
        
        print('Situation before first redo...')
        print(courtSF['trackFlag'].value_counts())
        
        ### redo frames that were not found
        redoBF = sf.get_redo_frames(courtSF)
        print(f'Re-attempting {redoBF.true()} frames...')
        redoNCD = NCDecapStream.from_bool(redoBF, cap, 
                                          target_color_space = 'grey', scale_factor = SCALE_FACTOR, 
                                          name = 'gFrm')
        redoNCD = redoNCD.attach(courtSF)
        redoSF = redoNCD.extract_mp(NUM_WORKERS, ex.redo_not_found)
        
        courtSF = courtSF.replace(redoSF)
        
        print('Situation before failure fix...')
        print(courtSF['trackFlag'].value_counts())
        
        ### try to fix failures
        redoBF = courtSF['trackFlag'] == 'Failed'
        print(f'Re-attempting {redoBF.true()} frames...')
        redoNCD = NCDecapStream.from_bool(redoBF, cap, 
                                          target_color_space = 'grey', scale_factor = SCALE_FACTOR, 
                                          name = 'gFrm')
        redoNCD = redoNCD.attach(courtSF)
        params = {'min_votes': int(500 * SCALE_FACTOR),
                  'vote_threshold': int(100 * SCALE_FACTOR)}
        redoSF = redoNCD.extract(ex.iterative_redo, **params)
        
        courtSF = courtSF.replace(redoSF)
        
        print('Situation before hole plugging...')
        print(courtSF['trackFlag'].value_counts())
        
        ### find holes in homography
        notBF = courtSF['trackFlag'] != 'Not Found'
        redoBF = sf.get_hole_frames(courtSF.filter(notBF))
        print(f'Re-attempting {redoBF.true()} frames...')
        redoNCD = NCDecapStream.from_bool(redoBF, cap, 
                                          target_color_space = 'grey', scale_factor = SCALE_FACTOR, 
                                          name = 'gFrm')
        redoNCD = redoNCD.attach(courtSF)
        params = {'min_votes': int(500 * SCALE_FACTOR),
                  'vote_threshold': int(100 * SCALE_FACTOR)}
        redoSF = redoNCD.extract(ex.iterative_redo, **params)  

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
    
        
        correctSF = correctList.concatenate()
        # rescale to Full HD (TODO: needs to be done always before saving... how to enforce?)
        correctSF = correctSF.rescale(1 / SCALE_FACTOR)
        # calculate corrected homography from corner points
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
        
    boardSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['scoreBoard']).squeeze()

    if boardSF.is_empty():
        
        SCALE_FACTOR = 2/3 ### downscale from full hd to hd 
        
        courtSF = HSF.merge(HSF.apply(ex.find_courtlines))
        courtSF = courtSF['courtLines'].rescale(SCALE_FACTOR)
        courtSFs = courtSF.split()

        grey_streamList = StreamList.from_SFList(courtSFs, cap, target_color_space = 'grey',
                                                 name = 'gFrm',
                                                 scale_factor = SCALE_FACTOR)    
        
        ### Find Scoreboard
        params = {'sobel_threshold': 30,
                  'slope_threshold': 0.0001,
                  'outlier_percentile': 0.8,
                  'min_lines_for_filtering': 4}
        boardSFs = grey_streamList.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.find_scoreboard, **params)
        boardSF = boardSFs.concatenate(axis = 0)
        boardSF = boardSF.rescale(1 / SCALE_FACTOR)
        boardSF.save(VIDEO_ID, STORE_NAME)
    
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
        SCALE_FACTOR = 2/3 ### is downscaling worth it here? overhead of scaling down H versus faster warping...
        
        ### create scaled courtLines and H
        courtSF_scaled = HSF.apply(ex.find_courtlines).rescale(SCALE_FACTOR)
        HSF_scaled = courtSF_scaled.apply(sa.get_court_corner_points).apply(sa.calculate_homography)
        
        ### filter for 'normal' frames
        normBool = sf.get_normal_bool(courtSF_scaled) ### apparently, the burn in algo is much stricter than our corners...
        
        ### filter for frames which are not at the beginning of a sequence 
            # -> in these, the players are likely to be serving/receiving, which we don't want to burn in!
        # params = {'first_seconds_to_disqualify': 10}
        params = {'last_seconds_to_take': 3,
                  'min_length_of_sequence': 13}
        nonserveBool = courtSF_scaled.split().apply_whole(sf.get_play_bool, **params).concatenate()
        
        ### choose 1000 out of the 'normal' frames 
        chosenBool = normBool.filter(normBool).filter(nonserveBool).random_choice(how_many = 1000) 
        hsvNCD = NCDecapStream.from_bool(chosenBool, cap, name = 'hFrm',
                                         target_color_space = 'hsv', 
                                         scale_factor = SCALE_FACTOR)
        
        ### warp selected courtlines to schema
        hsvNCD = hsvNCD.attach(HSF_scaled)
        wrpSF = hsvNCD.extract_mp(NUM_WORKERS, ex.courtlines_to_schema)
        
        medFrm, stdFrm = sf.get_median_and_deviation_frame(wrpSF)
        # medFrm, stdFrm = medFrm.rescale(1 / SCALE_FACTOR), stdFrm.rescale(1 / SCALE_FACTOR)
        ### -> not necessary! the smaller frame is projected onto a constant size schemaFrm
        
        # medFrm = HSVFrame.from_csv(burn_vidPath, name = 'median_backFrm')
        # stdFrm = HSVFrame.from_csv(burn_vidPath, name = 'std_backFrm')
        
        for someSF in courtSF_scaled.split():
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
                                         cap_color_space = 'bool', target_color_space = 'mask')
    except FileNotFoundError:
        SCALE_FACTOR = 2/3
        
        ### create scaled courtLines and H
        courtSF_scaled = HSF.apply(ex.find_courtlines).rescale(SCALE_FACTOR)
        HSF_scaled = courtSF_scaled.apply(sa.get_court_corner_points).apply(sa.calculate_homography)
        
        hsvStms = StreamList.from_SFList(HSF_scaled.split(), cap, name = 'hFrm',
                                         cap_color_space = 'bgr', target_color_space = 'hsv',
                                         scale_factor = SCALE_FACTOR)
        mulStms = StreamList.combine(hsvStms, medStms, stdStms)
        
        params = {'minimum_proportion': 0.0125,
                  'minimum_step_gain': 0.002}
        burnSFs = mulStms.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.get_burn_in_mask_iterative, **params)
        burnSFs = burnSFs.rescale(1 / SCALE_FACTOR)
        
        burnSFs.save(VIDEO_ID, STORE_NAME, lossless = True)
        burnStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, 
                                         cap_color_space = 'bool', target_color_space = 'mask')
    
    ### Get Shadow Mask
    featureName = 'shadowMask'
    try: 
        shadStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, 
                                         cap_color_space = 'bool', target_color_space = 'mask')
    except FileNotFoundError:
        SCALE_FACTOR = 2/3
        
        ### create scaled courtLines and H
        courtSF_scaled = HSF.apply(ex.find_courtlines).rescale(SCALE_FACTOR)
        HSF_scaled = courtSF_scaled.apply(sa.get_court_corner_points).apply(sa.calculate_homography)
        
        hsvStms = StreamList.from_SFList(HSF_scaled.split(), cap, name = 'hFrm',
                                         cap_color_space = 'bgr', target_color_space = 'hsv',
                                         scale_factor = SCALE_FACTOR)
        mulStms = StreamList.combine(hsvStms, medStms)
        
        params = {'h_threshold': 5,
                  's_threshold': 10,
                  'v_threshold_lower': 0.75,
                  'v_threshold_upper': 0.97}
        shadSFs = mulStms.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.get_shadow_mask, **params)
        shadSFs = shadSFs.rescale(1 / SCALE_FACTOR)
    
        shadSFs.save(VIDEO_ID, STORE_NAME, lossless = True)
        shadStms = StreamList.from_store(featureName, VIDEO_ID, STORE_NAME, 
                                         cap_color_space = 'bool', target_color_space = 'mask')
    
    
    plSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['topPlayer', 'botPlayer'])
    plSFs = plSF.split()
    
    if plSF.is_empty():
        SCALE_FACTOR = 2/3
        
        ### create scaled courtLines and H
        courtSF_scaled = HSF.apply(ex.find_courtlines).rescale(SCALE_FACTOR)
        HSF_scaled = courtSF_scaled.apply(sa.get_court_corner_points).apply(sa.calculate_homography)
        mySFs = HSF_scaled.unsqueeze().add('topPlayer', None).add('botPlayer', None).split()
        
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
        
        greyCtms = StreamList.from_SFList(mySFs, cap, cache_size = 1, name = 'gFrm',
                                          cap_color_space = 'bgr', target_color_space = 'grey',
                                          scale_factor = SCALE_FACTOR)
        burnStms = burnStms.rescale(SCALE_FACTOR)
        shadStms = shadStms.rescale(SCALE_FACTOR)
        mulStms = StreamList.combine(greyCtms, burnStms, shadStms)
        
        params = {'player_tracking_scale_factor': 1.5,
                  'bot_min_bounding_height': 200,
                  'bot_min_bounding_width': 80,
                  'top_min_bounding_height': 150,
                  'top_min_bounding_width': 60,
                  'initial_ROI_upper': initial_ROI_upper,
                  'initial_ROI_lower': initial_ROI_lower,
                  'halfcourtLine_upper': halfcourtLine_upper,
                  'halfcourtLine_lower': halfcourtLine_lower}
        
        plSFs = mulStms.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.find_both_players_smart, **params)
        
        plSFs = plSFs.rescale(1 / SCALE_FACTOR)
        plSFs.save(VIDEO_ID, STORE_NAME)
    
    ########################
    ### Ball Recognition ###
    ########################
    contSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['movCnts', 'ballTable'])
    if contSF.is_empty():
        SCALE_FACTOR = 1
        
        #a) get the motion mask by frame differencing
        greyCtms = StreamList.from_SFList(plSFs.rescale(SCALE_FACTOR), cap, cache_size = 8, name = 'gFrm',
                                          cap_color_space = 'bgr', target_color_space = 'grey',
                                          scale_factor = SCALE_FACTOR)
        hsvCtms = StreamList.from_SFList(plSFs.rescale(SCALE_FACTOR), cap, cache_size = 8, name = 'hFrm',
                                          cap_color_space = 'bgr', target_color_space = 'hsv',
                                          scale_factor = SCALE_FACTOR)
        mulCtms = StreamList.combine(greyCtms, hsvCtms)
        
        params = {'intensity_threshold': 10, 
                  'relevant_indices': [4, 6, 8],
                  'dil_kernel_dims': (20 * SCALE_FACTOR, 10 * SCALE_FACTOR), 
                  'open_kernel_dims': (5 * SCALE_FACTOR, 5 * SCALE_FACTOR),
                  'max_number_of_contours': 60}
        contSFs = mulCtms[:3].extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.get_movement_contours_and_frame, **params)
        
        contSF = contSFs.concatenate().dropna()
        contSF.save(VIDEO_ID, STORE_NAME, compression = 24)
    
    
    candSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, featnameList = ['candCnts'])
    if candSF.is_empty():
        ### construct mean and std of movement frame features, to be able to normalize ball shapes
        # ballTable = ff.read_feature_frame('ballTable', VIDEO_ID, STORE_NAME)
        ballTable = contSF['ballTable'].to_frame()
        mean, std = ballTable.mean(), ballTable.std()
        
        #### READ IN THE BALL TRACKING SVM ######
        svm_model_name = 'svm1'
        svm_dict_name = 'svm_proba_dict_2023-06-17'
    
        ballDict = ut.get_folder_struct('ball_svm')
        fitPath = os.path.join(ballDict['main'], svm_model_name, 'fitted_svms')
        
        svmPath = os.path.join(fitPath, svm_dict_name + '.pickle')
    
        import pickle
    
        with open(svmPath, 'rb') as f:
            svmDict = pickle.load(f)
        #######################
        
        params = {'svmDict': svmDict,
                  'use_training_moments': False,
                  'probability_threshold': None}
        candSF = sf.find_ball_candidates(contSF, **params)
        candSF = candSF.dropna()
        
        candSF.save(VIDEO_ID, STORE_NAME, compression = 24)
    
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
    
    