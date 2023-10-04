# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:34:55 2023

In this file, we collect all labels, and 
- check which ones are over- and underrepresented
- crop and center the event sequences on/around the player
- prepare the symmetries used for training 
    - flip left and right
    - removal of background (?)
    - change brightness/contrast, etc. (?)

@author: gianl
"""

#### READ IN ######
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.sf_funcs as sf

from ATMANpy.types.Lines import Region, Regions
from ATMANpy.types.Features import BoolFeatures, StreamFeature, StreamFeatures, SFList, ConstFeatures
from ATMANpy.types.Streams import Stream, StillStream, CachedStream, MulStream, StreamList
from ATMANpy.funcs.label_funcs import group_events, check_group, check_other, fill_in_group, group_to_dataframe, other_to_dataframe

if __name__ == '__main__':
    ### vidDict maps the video id to the first and last isCourt segment we want to label.
    vidDict = {
                '2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa': (0, 2),
                '2021-7341-ms002-brest-brandon-nakashima-vs-alexandre-muller': (0, 1),
                '2021-7341-ms003-brest-joao-sousa-vs-maxime-janvier': (0, 2),
                '2021-7341-ms004-brest-alexandre-muller-vs-luca-van-assche': (0, 2),
                '2021-7341-ms005-brest-brandon-nakashima-vs-henri-laaksonen': (0, 2),
                '2021-7341-ms007-brest-joao-sousa-vs-federico-gaio': (0, 2),
                '2021-7341-ms008-brest-luca-van-assche-vs-manuel-guinard': (0, 1),
                '2021-7341-ms009-brest-alexandre-muller-vs-stefano-travaglia': (1, 1), ### has warm up as first isCourt segment!
                '2021-7341-ms010-brest-brandon-nakashima-vs-bernabe-zapata-miralles': (1, 2), ### has warm up as first isCourt segment!
                '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano': (0, 3)
               }
    sto = 'segmentation_and_labelling'
    
    zoom_factor = 2
    
    label_names = ['labelled_top_hit_Flag', 'labelled_bot_hit_Flag', 
                   'labelled_top_serve_Flag', 'labelled_bot_serve_Flag']
    class_names = ['labelled_top_hit_side_Flag', 'labelled_bot_hit_side_Flag',
                   'labelled_top_hit_spin_Flag', 'labelled_bot_hit_spin_Flag',
                   'labelled_top_hit_type_Flag', 'labelled_bot_hit_type_Flag',
                   'labelled_top_serve_side_Flag', 'labelled_bot_serve_side_Flag']
    player_names = ['topPlayer', 'botPlayer']
    
    
    FRAME_OFFSET = 5 ### for player finding
                
    
    ######################
    ###### READ IN #######
    ######################
    
    conf = ut.read_config('brest')
    nwo, nch = conf['num_workers'], conf['chunksize']
    
    ### Court Lines
    HPSF = ConstFeatures.from_store('2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano', 'svm_run_seg_lossless', 
                                    featnameList = ['H', 'P'])
    courtSF = HPSF['H'].apply(sa.find_courtlines)
    
    ### Court ROI
    courtroiSF = ConstFeatures.from_store('2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano', 'svm_run_seg_lossless', 
                                          featnameList = ['courtroiReg'], 
                                          featTypeList = [Region]).squeeze()
    courtROI = courtroiSF.feature
    
    ### Ballkid ROI 
    kidroiSF = ConstFeatures.from_store('2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano', 'svm_run_seg_lossless',  
                                        featnameList = ['ballkidROIs'], 
                                        featTypeList = [Regions]).squeeze()
    kidROIs = kidroiSF.feature
    
    ### Label the ROIs depending on position
    namedROIs = []
    cntPnts = kidROIs.centroids()
    
    court = courtSF.feature
    ordered_labelDict = {'Mid Left': court['Net Line'].leftPt,
                         'Mid Right': court['Net Line'].rightPt,
                         'Top Left': court['Top Baseline'].leftPt,
                         'Top Right': court['Top Baseline'].rightPt,
                         'Bot Left': court['Bottom Baseline'].leftPt,
                         'Bot Right': court['Bottom Baseline'].rightPt}
    for label, myPt in ordered_labelDict.items():
        nearest_idx = np.argmin([cntPt.distance_to(myPt) for cntPt in cntPnts])
        nearest_centroid = cntPnts[nearest_idx]
        nearest_region = [roi for roi in kidROIs if roi.centroid() == nearest_centroid][0]
        namedROIs.append(nearest_region.rename(label))
        cntPnts.pop(nearest_idx)
        
    namedRois = Regions(namedROIs)
    
    ##############################
    ###### FEATURE CREATIION #####
    ##############################
    
    for vid in vidDict:
        
        print('________')
        print(vid)
        print('________')
        
        cap = ut.get_video(vid)
        
        ### Read in What we already have...
        # labSF = StreamFeatures.from_store(vid, sto, featnameList = label_names)
        # clsSF = StreamFeatures.from_store(vid, sto, featnameList = class_names)
        plySF = StreamFeatures.from_store(vid, sto, featnameList = player_names) 
        segBF = BoolFeatures.from_store(vid, sto, featnameList = ['isCourt']).squeeze()
        plyBF = BoolFeatures.from_store(vid, sto, featnameList = ['isPlay']).squeeze()
                
        ### Create isCourt Segments of Interest
        fs, ls = vidDict[vid]
        segBFs = segBF.filter(segBF).split()[fs: ls + 1]
        segBF = segBFs.concatenate()
        
        ### Check for which we still have to collect the players
        jntSFs = segBF.merge(plySF).merge(plyBF).split()
        for idx, jntSF in enumerate(jntSFs):
            players_in_segment = jntSF.drop('isCourt').drop('isPlay').dropna()
            if players_in_segment.is_empty():
                
                courtBF = jntSF['isCourt']
                courtStm = Stream.from_SF(courtBF, cap)
                start, stop = courtStm.start_frame, courtStm.stop_frame

                burnCtm = CachedStream.from_store(f'from_{start}_to_{stop}.avi', 'foregroundMask', vid, sto, 
                                             cap_color_space = 'bool', target_color_space = 'mask',
                                             cache_size = FRAME_OFFSET, force_fill_cache = True)
                shadCtm = CachedStream.from_store(f'from_{start}_to_{stop}.avi', 'shadowMask', vid, sto, 
                                             cap_color_space = 'bool', target_color_space = 'mask',
                                             cache_size = FRAME_OFFSET, force_fill_cache = True)
                # motiStm = Stream.from_store(f'from_{start}_to_{stop}.avi', 'rough_motMask', vid, sto, 
                #                              cap_color_space = 'bool', target_color_space = 'mask')
                
                if burnCtm.length() == 0 or shadCtm.length() == 0:
                    ############################
                    ####### Backgrounds ########
                    ############################
                    # -> needs: H, isCourt
                    TRUNC_N = 250
                    tightStm = courtStm.widen(-TRUNC_N)
                    
                    SAMPLE_N = 1000
                    if tightStm.length() < SAMPLE_N:
                        print('Too few frames in courtStm, consider reducing TRUNC_N')
                        continue
                    
                    sampleStm = tightStm.sample(SAMPLE_N)
                    sampleDecStm = sampleStm.decap()
                    
                    medFrm, stdFrm = sampleDecStm.frames.median(), sampleDecStm.frames.std()
                    medFrm, stdFrm = medFrm.rename('median_backFrm'), stdFrm.rename('std_backFrm')
                    medFrm, stdFrm = medFrm.to_HSV(), stdFrm.to_HSV()
                    
                    ############################
                    ########## Masks ###########
                    ############################
                    # -> needs: isCourt, courtroiReg, median_backFrm, std_backFrm
                    medStm = StillStream(medFrm, courtStm.start_frame, courtStm.stop_frame)
                    stdStm = StillStream(stdFrm, courtStm.start_frame, courtStm.stop_frame)
                    hsvStm = courtStm.recolor('hsv')
                    
                    mulStm = MulStream([hsvStm, medStm, stdStm])
                    params =  {'ROI': courtROI,
                               'h_threshold': 16, 
                               'v_threshold': 16, 
                               'gaussian_kernel': (round(21 * conf['scale_factor']), round(21 * conf['scale_factor'])), 
                               'closing_kernel': (round(10 * conf['scale_factor']), round(18 * conf['scale_factor'])),
                               'verbose': False}
                    burnSF = mulStm.extract_mp(ex.get_burn_in_mask_simple, nwo, **params)
                    burnSF.save(vid, sto, lossless = conf['lossless'])
                            
                    ### Shadow Mask
                    mulStm = MulStream([hsvStm, medStm])
                    params = {'ROI': courtROI,
                              'h_threshold': 5,
                              's_threshold': None,
                              'v_threshold_lower': 0.75,
                              'v_threshold_upper': 0.97,
                              'opening_kernel': (round(15 * conf['scale_factor']), round(15 * conf['scale_factor'])),
                              'closing_kernel': (round(15 * conf['scale_factor']), round(15 * conf['scale_factor']))}
                    shadSF = mulStm.extract_mp(ex.get_shadow_mask_simple, nwo, **params)
                    shadSF.save(vid, sto, lossless = conf['lossless'])
                    
                    burnCtm = CachedStream.from_store(f'from_{start}_to_{stop}.avi', 'foregroundMask', vid, sto, 
                                                 cap_color_space = 'bool', target_color_space = 'mask',
                                                 cache_size = FRAME_OFFSET, force_fill_cache = True)
                    shadCtm = CachedStream.from_store(f'from_{start}_to_{stop}.avi', 'shadowMask', vid, sto, 
                                                 cap_color_space = 'bool', target_color_space = 'mask',
                                                 cache_size = FRAME_OFFSET, force_fill_cache = True)
            
                # if motiStm.length() == 0:
                #     ### Motion Mask
                #     gCtm = CachedStream.from_SF(courtBF, cap, cache_size = 8, 
                #                                 target_color_space = 'grey', force_fill_cache = True)
                    
                #     # b) a rougher mask, for ballkid detection
                #     params = {'ROI': courtROI,
                #               'intensity_threshold': 10,
                #               'relevant_indices': [4, 6, 8],
                #               'close_kernel': (17, 17),
                #               'open_kernel': (4, 4)}
                #     motSF = gCtm.extract(ex.find_motion_mask, **params)
                #     motSF = motSF.rename('rough_motMask')
                #     motSF.save(vid, sto, lossless = conf['lossless'])
                    
                #     motiStm = Stream.from_store(f'from_{start}_to_{stop}.avi', 'rough_motMask', vid, sto, 
                #                                  cap_color_space = 'bool', target_color_space = 'mask')
                    

                # ### Find motion contours
                # params = {'min_area_of_contours': 10,
                #           'ROI': None, ### 6x slower when specifying a ROI!
                #           'max_number_of_contours': None}
                # cntSF = motiStm.rename('intMask').extract_mp(ex.get_contours_from_mask, nwo, **params)
                # ballkidBF = sf.find_ballkid_sequences(cntSF, namedROIs)
                    
                # playBF = courtBF.merge(ballkidBF.neg().rename('isNotBallKid')).to_bool()
                # playBF = playBF.all().rename('isPlay')
                # playBF.save(vid, sto)
                
                # chkStms = StreamList.from_SFList(playBF.filter(playBF).split(), cap)
            
                ############################
                ######### isPlay ###########
                ############################
                # -> needs: courtroiReg, ballkidROIs, isCourt, rough_motMask
                # --> done with labelled data!!!
                playBF = jntSF['isPlay'].to_bool()
                
                ############################
                ### topPlayer, botPlayer ###
                ############################
                # -> needs: H, courtroiReg, foregroundMask, shadowMask, isPlay 
                
                mySFs = jntSF.add('topPlayer_candidates', None).add('botPlayer_candidates', None).filter(playBF).split()
                burnCtms = StreamList([burnCtm]).conform(mySFs).attach(mySFs)
                shadCtms = StreamList([shadCtm]).conform(mySFs).attach(mySFs)
                            
                ### Run Player Extraction
                params = {'min_player_size': 50, 'ROI': courtroiSF.feature,
                          'frame_offset': FRAME_OFFSET, 'frame_blur': 1}
                mulStms = StreamList.combine(burnCtms, shadCtms)
                
                trackSFs = mulStms.extract_chunked_mp(ex.find_player_candidate_tracks_joint_timeblur, nwo, nch, **params)
                            
                ### Split Candidate Contours with Links into Tracks 
                params = {'critical_number_of_tracks': 50000,
                          'frame_offset': FRAME_OFFSET,
                          'verbose': False,
                          'super_verbose': False}
                splitDFs = trackSFs.apply_whole_mp(sf.split_player_candidate_tracks, nwo, **params)
                splitDFs = splitDFs.sfList ### have to do this so that splitDFs is not an SFList type
                ### -> note that the outputs here will be dataframes!!!
            
                candSFs = trackSFs.concatenate().add('courtroiReg', courtroiSF.feature).add('courtLines', courtSF.feature).split()
                
                params = {'granularity': 10,
                          'area_outlier_iqr_multiple': 1.5,
                          'overlap_optimum': 0.9 ** FRAME_OFFSET,
                          'overlap_strictness': 5,
                          'metric_weights': {'dist': 5, 'ovlp': 2, 'area': 1, 'move': 2}}
                plSFs = candSFs.apply_whole_mp_starmapped(sf.evaluate_player_candidates, nwo, splitDFs, **params)
                
                plSFs.save(vid, sto, compression = conf['compression_players'])
                
                StreamList.from_SFList(plSFs, cap).show(ms_between_frames = 10)
                
                # StreamList.from_SFList(trackSFs, cap).show(ms_between_frames = 10)
                
                
                
                
                
                
                