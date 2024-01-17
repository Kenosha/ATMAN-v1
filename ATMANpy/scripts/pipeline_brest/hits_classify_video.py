# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:18:36 2023

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

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

import torch
from torch.utils.data import DataLoader
from ATMANpy.models.TemporalCNN.nets import SmallFullConvModel
from ATMANpy.models.TemporalCNN.data import PlayerInferenceDataset
from ATMANpy.models.TemporalCNN.utils import infer_with_index, load_cnn, label_map, sum_over_distinct_labels

def load(vid, sto):
    HCF = ConstFeature.from_store(vid, sto, featname = 'H')
    courtroiCF = ConstFeature.from_store(vid, sto, featname = 'courtroiReg', featType = Region)
    
    ### Read Segmentation Boolean, which combines knowledge about courtlines and players serving.
    playBF = BoolFeature.from_store(vid, sto, featname = 'isServeStart')
    ### Read Courtline Segmenter
    courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
    
    playersSF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    
    return HCF, courtroiCF, playBF, courtBF, playersSF

def save(topFrm, botFrm, vid, sto):
    ff.save_feature_frame(topFrm, 'top_hitTable', vid, sto)
    ff.save_feature_frame(botFrm, 'bot_hitTable', vid, sto)

def main(conf, model_params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    
    ### additional params which right now can/should not be changed
    loader_workers = 0
    device = 'cuda'
    report_frequency = 100
    
    HCF, courtroiCF, playBF, courtBF, playersSF = load(vid, sto)
    
    H = HCF.feature
    schemaLines = CourtLines.make().add_singles_netline()
    courtLines = sa.find_courtlines(H, schemaLines = schemaLines)
    courtCF = ConstFeature('courtLines', courtLines)
    
    ### Run models on player contours
    prbDict = {}
    labDict = {}
    prmDict = {}
    for player in ['top', 'bot']:
        model_name = model_params[player]
        playerSF = playersSF[f'{player}PlayerCnt'].dropna()
        
        ### load model
        model, params = load_cnn(model_name, SmallFullConvModel, device = device)
        
        ### collect some params
        batch_size = params['BATCH_SIZE']
        D_SC = params['D_SCALE_SIZE']
        D = params['D_CROP_SIZE']
        T = params['T_CROP_SIZE']
        stride_t = model.stride_t
        
        ds = PlayerInferenceDataset(cap, vid, sto, courtBF, playBF, playerSF,  
                                    stride_t = stride_t, T = T, D = D, D_SC = D_SC) 
        
        inf_loader = DataLoader(ds, batch_size = batch_size, num_workers = loader_workers, pin_memory = True)
        prbArr, labArr = infer_with_index(model, inf_loader, device = device, report_frequency = report_frequency)
        prbArr = prbArr.squeeze() ### kill T, H, W dims (which remain due to fully convolutional model)
        
        prbDict[player] = prbArr
        labDict[player] = labArr
        prmDict[player] = params
        
    ### Aggregate probabilities of hits
    boolDict = {}
    regDict = {}
    frmDict = {}
    for player in ['top', 'bot']:
        prbArr, labArr, params = prbDict[player].numpy(), labDict[player].numpy(), prmDict[player]

        frm_index = labArr.mean(axis = 1)
        frm_index_int = frm_index.astype(int)
        
        labels = params['LABELS']
        labels_mapped = [label_map(lab) for lab in labels]
        
        prbArr_grp, unique_labels_mapped = sum_over_distinct_labels(prbArr, labels_mapped)
        
        unique_labels_mapped = [f'{player}_{label}' for label in unique_labels_mapped]
        
        prbFrm = pd.DataFrame(prbArr_grp, columns = unique_labels_mapped, index = frm_index_int)
        prbFrm = prbFrm.reindex(playBF.filter(playBF).get_frames())
        prbFrm = prbFrm.interpolate(method = 'index', limit_area = 'inside')
        prbFrm = prbFrm.dropna()
        
        frmDict[player] = prbFrm

    topFrm, botFrm = frmDict['top'], frmDict['bot']
    save(topFrm, botFrm, vid, sto)


if __name__ == '__main__':
    ### CNN parameters
    device = 'cuda'
    loader_workers = 0 ### TODO: Parallelize! (see todo_cnn)
    both_right_handed = True
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    
    model_params = {'top': 'top_foremotion_T16_S8_randcrop5_dim56_small_fullconv_model_foremotion_20231121_235426',
                    'bot': 'bot_foremotion_T16_S8_randcrop5_dim88_small_fullconv_model_foremotion_20231122_095800'}
    
    main(conf, model_params)
        
        
        
        