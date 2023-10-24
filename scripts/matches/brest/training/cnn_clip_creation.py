# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:53:27 2023

@author: gianl
"""


#### READ IN ######
import random
import os

import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.sf_funcs as sf

from ATMANpy.types.Lines import Region, Regions
from ATMANpy.types.Features import BoolFeatures, StreamFeature, StreamFeatures, SFList, ConstFeatures
from ATMANpy.types.Streams import Stream, StillStream, CachedStream, MulStream, StreamList, DecapStream
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Contours import Contours

from ATMANpy.funcs.label_funcs import group_events, unambiguify_group, fill_in_group_2, group_map

def shorten_col_name(col):
    segs = col.split('_')
    segs = segs[2:-1]
    if len(segs) == 1:
        segs = [*segs, 'event']
    return '_'.join(segs)

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
    vidDict = {'2021-7341-ms002-brest-brandon-nakashima-vs-alexandre-muller': (0, 1)}
    sto = 'segmentation_and_labelling'
    
    ZOOM_FACTOR = 2
    T_LENGTH = 32
    T_STRIDE = 16 # T_MARGIN * 2 + 1 if you want no overlap at all!
    PLAYER = 'bot'
    SIZEDICT = {'top': (60, 60),
                'bot': (120, 120)}
    H, W = SIZEDICT[PLAYER]
    SAVE_DIR = f'{PLAYER}_player_labels_testing'
    
    ### FIGURE OUT WHERE TO SAVE
    savePath = ut.get_folder_struct('video_files')['labelled_data']
    savePath = os.path.join(savePath, SAVE_DIR)
    try:
        os.mkdir(savePath)
    except FileExistsError:
        pass
        
    ######## READ IN ###########
    hit_name = f'labelled_{PLAYER}_hit_Flag'
    srv_name = f'labelled_{PLAYER}_serve_Flag'
    hit_cls_names = [f'labelled_{PLAYER}_hit_side_Flag', f'labelled_{PLAYER}_hit_spin_Flag', f'labelled_{PLAYER}_hit_type_Flag']
    srv_cls_names = [f'labelled_{PLAYER}_serve_side_Flag']
    
    capDict = {}
    labDict = {}
    plyDict = {}
    segDict = {}
    chkDict = {}
    
    for vid in vidDict:
        
        cap = ut.get_video(vid)
        
        ### Read in What we already have...
        hitSF = StreamFeatures.from_store(vid, sto, featnameList = [hit_name]).squeeze()
        srvSF = StreamFeatures.from_store(vid, sto, featnameList = [srv_name]).squeeze()
        hit_clsSF = StreamFeatures.from_store(vid, sto, featnameList = hit_cls_names)
        srv_clsSF = StreamFeatures.from_store(vid, sto, featnameList = srv_cls_names)
        
        hit_groups = [fill_in_group_2(group_map(group, 'hit'), 'hit') for group in group_events(hitSF, 'hit')]
        srv_groups = [fill_in_group_2(group_map(group, 'serve'), 'serve') for group in group_events(srvSF, 'serve')]
        
        hitSF = StreamFeature({k: v  for hit_group in hit_groups for k, v in hit_group.items()}, name = hit_name)
        srvSF = StreamFeature({k: v  for srv_group in srv_groups for k, v in srv_group.items()}, name = srv_name)
        
        hitSF = hitSF.merge(hit_clsSF, how = 'left')
        srvSF = srvSF.merge(srv_clsSF, how = 'left')
        
        labSF = hitSF.merge(srvSF, how = 'outer')
        labSF = labSF.conform(list(range(0, 100000))) ### conform with something large, so that we ALWAYS get a full size label frame, later.
        
        plySF = StreamFeatures.from_store(vid, sto, featnameList = [f'{PLAYER}Player']).squeeze() 
        segBF = BoolFeatures.from_store(vid, sto, featnameList = ['isPlay']).squeeze()
        chkBF = BoolFeatures.from_store(vid, sto, featnameList = [f'{PLAYER}Player_fine']).squeeze()
        
        segBF_strong = segBF.merge(chkBF).to_bool().all() ### restrict the segmenter to only use situations where player track is fine!
        
        capDict[vid] = cap
        labDict[vid] = labSF
        plyDict[vid] = plySF
        segDict[vid] = segBF_strong
        chkDict[vid] = chkBF
        
    ###############################
     
    #%% SEQUENTIAL CHOICE
    
    for vid in vidDict:
        idx = 0
        
        cap = capDict[vid]
        fs_seq, ls_seq = vidDict[vid]
        
        plSF = plyDict[vid][f'{PLAYER}Player']
        labSF = labDict[vid]
        labDFs = []
        
        vidPath = os.path.join(savePath, vid)
        try:
            os.mkdir(vidPath)
        except FileExistsError:
            pass
        
        for seq in range(0, ls_seq - fs_seq + 1):
        
            seqBF = segDict[vid].split()[seq] ### this now is our isPlay segmenter for the chosen sequence
            pntBFs = seqBF.filter(seqBF).split()
            
            for pntBF in pntBFs:
                fs_frm, ls_frm = pntBF.get_frames()[0], pntBF.get_frames()[-1]
                
                left_frames = range(fs_frm, ls_frm - T_LENGTH, T_STRIDE)
                
                for left_frame in left_frames:                    
                    right_frame = left_frame + T_LENGTH
                    choice_frames = list(range(left_frame, right_frame))
                    
                    myFrms = BGRFrames.from_video(cap, left_frame, right_frame)
                                
                    ####### CROP ONTO PLAYER ############
                    plCnts = []
                    for f in choice_frames:
                        try:
                            plCnt = plSF[f]
                        except KeyError:
                            continue
                        plCnts.append(plCnt)
                    
                    plCnts = Contours(plCnts)
                    if plCnts.length() == 0:
                        continue ### in this case, we have an ok point for the player side (chkBF above), but no player contour in this moment.
                    
                    plReg = plCnts.enclosing_regions().cover()
                    # if plReg.height() > H or plReg.width() > W:
                    #     print(plReg.height(), plReg.width())
                    plReg = plReg.resize(to_height = H, to_width = W)
                    
                    cropFrms = myFrms.crop(plReg)
                    assert cropFrms.rows() == H and cropFrms.cols() == W
                    
                    cropFrms.save(vidPath, f'clip_{idx}', lossless = True)
                    
                    ######## GET LABELS ##################
                    labDF = labSF[choice_frames].to_dataframe().reset_index(drop = True)
                    labDF.index.name = 'frame'
                    labDF = ut.prepend_index_level(labDF, 'clip', idx)
                    labDFs.append(labDF)
                    
                    idx += 1
                    
        labDF = pd.concat(labDFs)
                
        labDF.columns = [shorten_col_name(col) for col in labDF.columns]
        labDF.to_csv(os.path.join(vidPath, 'labels.csv'))
                    
    
    #%% RANDOM CHOICE
    
    # ####### CHOSE A SEQUENCE ######
    # ## TODO: How much weight on hits, serves?
    # labDFs = []
    # for idx in range(300):
    #     # these are chosen sensibly, at random
    #     vid = random.choice(list(vidDict.keys()))
    #     fs_seq, ls_seq = vidDict[vid]
    #     seq = random.choice(range(fs_seq, ls_seq + 1)) - fs_seq
        
    #     cap = capDict[vid]
        
    #     seqBF = segDict[vid].split()[seq] ### this now is our isPlay segmenter for the chosen sequence
        
    #     pntBFs = seqBF.filter(seqBF).split() ### have to chose a random point, from this!
    #     pntBF = random.choice(pntBFs.sfList)
    #     fs_frm, ls_frm = pntBF.get_frames()[0], pntBF.get_frames()[-1]
        
    #     choice_frm = random.randrange(fs_frm + T_MARGIN, ls_frm - T_MARGIN + 1)
    #     choice_fs, choice_ls = choice_frm - T_MARGIN, choice_frm + T_MARGIN
    #     choice_frms = list(range(choice_fs, choice_ls + 1))
        
    #     myFrms = BGRFrames.from_video(cap, choice_fs, choice_ls)
        
    #     ####### CROP ONTO PLAYER ############
    #     plSF = plyDict[vid][f'{PLAYER}Player']
    #     plCnts = []
    #     for f in choice_frms:
    #         try:
    #             plCnt = plSF[f]
    #         except KeyError:
    #             continue
    #         plCnts.append(plCnt)
        
    #     plCnts = Contours(plCnts)
    #     if plCnts.length() == 0:
    #         continue ### in this case, we have an ok point for the player side (chkBF above), but no player contour in this moment.
        
    #     plReg = plCnts.enclosing_regions().cover()
    #     if plReg.height() > H or plReg.width() > W:
    #         print(plReg.height(), plReg.width())
    #     plReg = plReg.resize(to_height = H, to_width = W)
        
    #     cropFrms = myFrms.crop(plReg)
    #     cropFrms.save(savePath, f'clip_{idx}', lossless = True)
        
    #     ######## GET LABELS ##################
    #     labSF = labDict[vid]
    #     labDF = labSF[choice_frms].to_dataframe().reset_index(drop = True)
    #     labDF.index.name = 'frame'
    #     labDF = ut.prepend_index_level(labDF, 'clip', idx)
    #     labDFs.append(labDF)
        
    # labDF = pd.concat(labDFs)
    
    # labDF.columns = [shorten_col_name(col) for col in labDF.columns]
    # labDF.to_csv(os.path.join(vidPath, 'labels.csv'))

    

    ######## CREATE THE CORRECT MODEL OUTPUT VECTOR ####
    ## --> depending on how centered temporally the snippet is on a hit or serve, 
    # we give the model a vector for learning, with one entry for each category (forehand topspin, ... serve, ... other.) 
    
    
    
    ####### PERFORM AUGMENTATIONS #######
    
    ## -> mirroring
    ## -> changing bgr 
        # of whole scene?
        # of background only?
    ## -> de-centering
    
    ###### ADD SOME NON-PLAYER-DATA #####
    # ---> especially for serve segmentation, we will use the model to infer before knowing where players are!
    # -> it needs to be able to discern empty space, or clutter, or the netpost, or whatever, and say: not a serve.
    # -> BUT DOES IT? If we do region proposals using foreground blobs, we will always have something "player-like"
    
    
    
    