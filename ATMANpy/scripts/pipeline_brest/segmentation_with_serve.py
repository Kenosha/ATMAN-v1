# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:20:45 2023

@author: gianl
"""


import sys
import os
import time
import dill as pickle
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

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

import torch
from torch.utils.data import DataLoader
from ATMANpy.models.TemporalCNN.nets import SmallFullConvModel
from ATMANpy.models.TemporalCNN.data import PlayerInferenceDataset
from ATMANpy.models.TemporalCNN.utils import infer_with_index, load_cnn

def load(vid, sto):
    ### Read Courtline Segmenter
    courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt') ### still need this because masks are tied to it.
    segBF = BoolFeature.from_store(vid, sto, featname = 'isValidFore') # this is slightly tighter than isCourt.
    
    ### Read Player Candidate Contours
    foreSF = StreamFeature.from_store(vid, sto, featname = 'foreCnts')
    
    ### Read Serve ROIs
    topCF = ConstFeature.from_store(vid, sto, featname = 'topserveReg', featType = Region)
    botCF = ConstFeature.from_store(vid, sto, featname = 'botserveReg', featType = Region)
    
    ### H
    HCF = ConstFeature.from_store(vid, sto, featname = 'H')
    
    return courtBF, segBF, foreSF, topCF, botCF, HCF

def save(playBF_filtered, topSF, botSF, topBF, botBF, vid, sto):
    playBF_filtered.rename('isServeStart').save(vid, sto)
    topSF.rename('topPlayer_srvReg').save(vid, sto)
    botSF.rename('botPlayer_srvReg').save(vid, sto)
    topBF.rename('topPlayer_srvFlag').save(vid, sto)
    botBF.rename('botPlayer_srvFlag').save(vid, sto)

def main(conf, model_params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = get_video(vid)
    
    ### additional params which right now can/should not be changed
    loader_workers = 0
    device = 'cuda'
    report_frequency = 100
    
    courtBF, segBF, foreSF, topCF, botCF, HCF = load(vid, sto)
    
    H = HCF.feature
    schemaLines = CourtLines.make().add_singles_netline()
    courtLines = sa.find_courtlines(H, schemaLines = schemaLines)
    
    topROI = topCF.feature
    botROI = botCF.feature
    
    ### Apply Criteria    
    topSF_1 = foreSF.apply(sa.keep_contours_in_roi, ROI = topROI)
    topSF_2 = topSF_1.apply(sa.keep_closest_contour_behind_baseline, which = 'Top', courtLines = courtLines)
    
    botSF_1 = foreSF.apply(sa.keep_contours_in_roi,  ROI = botROI)
    botSF_2 = botSF_1.apply(sa.keep_closest_contour_behind_baseline, which = 'Bottom', courtLines = courtLines)
    
    contours = {'top': topSF_2.dropna(), 'bot': botSF_2.dropna()}

    prbDict = {}
    labDict = {}
    prmDict = {}
    for player in ['top', 'bot']:
        model_name = model_params[player]
        candSF = contours[player]
        
        ### load model
        model, params = load_cnn(model_name, SmallFullConvModel, device = device)
        
        ### collect some params
        batch_size = params['BATCH_SIZE']
        D_SC = params['D_SCALE_SIZE']
        D = params['D_CROP_SIZE']
        T = params['T_CROP_SIZE']
        stride_t = model.stride_t
        
        ds = PlayerInferenceDataset(cap, vid, sto, courtBF, segBF, candSF,  
                                    stride_t = stride_t, T = T, D = D, D_SC = D_SC) 

        inf_loader = DataLoader(ds, batch_size = batch_size, num_workers = loader_workers, pin_memory = True)
        prbArr, labArr = infer_with_index(model, inf_loader, device = device, report_frequency = report_frequency)
        prbArr = prbArr.squeeze() ### kill T, H, W dims (which remain due to fully convolutional model)
        
        prbDict[player] = prbArr
        labDict[player] = labArr
        prmDict[player] = params
    
    ### Check inference for detected serves
    boolDict = {}
    regDict = {}
    for player in ['top', 'bot']:
        prbArr, labArr, params = prbDict[player].numpy(), labDict[player].numpy(), prmDict[player]
        candSF = contours[player]
        
        frm_index = labArr.mean(axis = 1)
        frm_index_int = frm_index.astype(int)
        
        labels = params['LABELS']
        srv_idxs = [idx for idx, lab in enumerate(labels) if lab[0] == 'any']
        srv_probs = prbArr[:, srv_idxs].sum(axis = 1)
        
        probSer = pd.Series(srv_probs, index = frm_index_int)
        # interpolate, so that holes from striding are killed.
        itplSer = probSer.reindex(segBF.get_frames())
        itplSer = itplSer.interpolate(method = 'index', limit_area = 'inside')
            
        # make boolean series by thresholding
        boolSer = itplSer > serve_params['serve_prob_threshold']
        boolSer = boolSer.rename(f'{player}Player_srvFlag')
        
        boolSF = BoolFeature.from_series(boolSer)
        
        # save also the region of the serving player, so we can initialize ball particles there
        regSF = candSF.filter(boolSF).apply(Contour.enclosing_region).rename(f'{player}Player_srvReg')
        regDict[player] = regSF

        # shift the boolean
            # right now serves = true occurs at the moment of the serve impact
            # we want it to be centered at the beginning of the serve, roughly.
        boolSer_shift = boolSer.shift(periods = -serve_params['serve_frame_shift']).dropna() # drop frames at the end, created by frameshift

        boolSF_shift = BoolFeature.from_series(boolSer_shift)
        boolDict[player] = boolSF_shift
        
    topBF, botBF = boolDict['top'], boolDict['bot']
    topSF, botSF = regDict['top'], regDict['bot']
        
    serveBF = topBF | botBF
    no_serveBF = serveBF.neg()
    
    ### Create a new segmenter for play sequences beginning with serve.
    playBF_naive = no_serveBF & segBF
    playBFs_naive = playBF_naive.filter(playBF_naive).split()    
    # Problem: boolean AND with courtBF causes some sequences at change overs not to begin with serve
    # idea: find all play segments, get first frame, check if the first frame - 1 is a serve (--> True in serveBF)
    begin_with_serve_bfs = []
    for bf in playBFs_naive:
        first_frame = bf.first()[0]
        is_serve = serveBF.maybe_get(first_frame - 1)
        if is_serve:
            begin_with_serve_bfs.append(bf)
        
    ### filter out sequences that are too short
    playBFs_filtered = SFList([bf for bf in begin_with_serve_bfs if bf.length() > serve_params['length_threshold']]) 
    playBF_filtered = playBFs_filtered.concatenate().conform(playBF_naive).fillna(False)

    print(f'Before length filtering: {len(begin_with_serve_bfs)} non-serve sequences, after filtering: {playBFs_filtered.length()} sequences')
            
    ### Save
    save(playBF_filtered, topSF, botSF, topBF, botBF, vid, sto)
    

if __name__ == '__main__':
        
    model_params = {'top': 'top_foremotion_T16_S8_randcrop5_dim56_small_fullconv_model_foremotion_20231121_235426',
                    'bot': 'bot_foremotion_T16_S8_randcrop5_dim88_small_fullconv_model_foremotion_20231122_095800'}
    
    serve_params = {'serve_prob_threshold': 0.5,
                    'serve_frame_shift': 50,
                    'serve_frames_to_pad_over': 10,
                    'length_threshold': 25}
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    
    main(conf, model_params, serve_params)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    