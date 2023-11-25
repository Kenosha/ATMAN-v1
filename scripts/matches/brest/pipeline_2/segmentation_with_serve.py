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
from ATMANpy.models.TemporalCNN.utils import infer_with_index

def load_cnn(model_name, device = 'cuda'):
    pathDict = ut.get_folder_struct('cnn')
    resPath, prmPath = pathDict['weights'], pathDict['params']
    
    res_read_where = os.path.join(resPath, model_name)
    prm_read_where = os.path.join(prmPath, model_name + '.pkl')
    
    ### Read Parameters 
    with open(prm_read_where, 'rb') as fp:
        params = pickle.load(fp)
        print('Read parameters: ')
        print(params)

    ### Read CNN
    T, D = params['T_CROP_SIZE'], params['D_SCALE_SIZE']
    d1, d2, d3, d4 = params['STRIDES_D']
    t1, t2, t3, t4 = params['STRIDES_T']
    
    model = SmallFullConvModel(labels = params['LABELS'], 
                               T = T, D = D,
                               stride_d1 = d1, stride_d2 = d2, stride_d3 = d3, stride_d4 = d4,
                               stride_t1 = t1, stride_t2 = t2, stride_t3 = t3, stride_t4 = t4,
                               share_other = 0.9)
    model.to(device)
    model.load_state_dict(torch.load(res_read_where))
    model.train(False)

    return model, params

### Define Criteria that a serving player contour has to fulfill.
# Criterion 1) center of shape is inside of the serve ROI
def keep_contours_in_roi(myCnts, which = 'Top'):
    assert which in ['Top', 'Bottom']
    
    if which == 'Top':
        serveReg = topROI
    elif which == 'Bottom':
        serveReg = botROI

    inCnts = Contours([cnt for cnt in myCnts if cnt.center().within(serveReg)])
    return inCnts
    
# Auxiliary function
def get_distance_to_baseline(myCnt, which = 'Top'):
    assert which in ['Top', 'Bottom']
    
    baseLine = courtLines[f'{which} Baseline']
    botPt = myCnt.enclosing_region()['Bottom'].center()
    linePt = botPt.project_vertically(baseLine)
    
    ### make it so that negative distances are not admissible -> they are in front of the line!
    ### then, the task is simply to choose the smallest positive distance.
    if which == 'Top':
        dist = linePt.y - botPt.y
    elif which == 'Bottom':
        dist = botPt.y - linePt.y
    
    return dist
    
 ### Criterion 2) bottom of shape is behind the respective baseline, keep the closest to baseline among these.
def keep_closest_contour_behind_baseline(myCnts, which = 'Top', margin = 5):
    assert which in ['Top', 'Bottom']
   
    bestCnt, best_dist = None, None
    for myCnt in myCnts:
        mrg_dist = get_distance_to_baseline(myCnt, which = which) + margin # to not exclude players at baseline, add a little margin
        if mrg_dist > 0: ### accept only contours whose bottom is behind the respective line
            if best_dist is None:
                bestCnt = myCnt
                best_dist = mrg_dist
            else:
                if mrg_dist < best_dist:
                    bestCnt = myCnt
                    best_dist = mrg_dist
    
    return bestCnt

def enclosing_region(myCnt):
    myReg = myCnt.enclosing_region()
    return myReg

if __name__ == '__main__':
        
    ### CNN parameters
    device = 'cuda'
    loader_workers = 0 ### TODO: Parallelize!
    serve_prob_threshold = 0.5
    serve_frame_shift = 50
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    ### Read Courtline Segmenter
    courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
    
    ### Read Player Candidate Contours
    foreSF = StreamFeature.from_store(vid, sto, featname = 'foreCnts')
    
    ### Read Serve ROIs
    topCF = ConstFeature.from_store(vid, sto, featname = 'topserveReg', featType = Region)
    botCF = ConstFeature.from_store(vid, sto, featname = 'botserveReg', featType = Region)
    topROI = topCF.feature
    botROI = botCF.feature
    
    ### Read H and make Courtlines
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    H = HSF.feature
    courtLines = sa.find_courtlines(H)
    
    ### Apply Criteria    
    topSF_1 = foreSF.apply(keep_contours_in_roi, which = 'Top')
    topSF_2 = topSF_1.apply(keep_closest_contour_behind_baseline, which = 'Top')
    
    botSF_1 = foreSF.apply(keep_contours_in_roi, which = 'Bottom')
    botSF_2 = botSF_1.apply(keep_closest_contour_behind_baseline, which = 'Bottom')
    
    contours = {'top': topSF_2.dropna(), 'bot': botSF_2.dropna()}
    
    ### Run Inference on Contours
    models = {'top': 'top_foremotion_T16_S8_randcrop5_dim56_small_fullconv_model_foremotion_20231121_235426',
              'bot': 'bot_foremotion_T16_S8_randcrop5_dim88_small_fullconv_model_foremotion_20231122_095800'}
    
    prbDict = {}
    labDict = {}
    prmDict = {}
    for player in ['top', 'bot']:
        model_name = models[player]
        candSF = contours[player]
        
        ### load model
        model, params = load_cnn(model_name, device = 'cuda')
        
        ### collect some params
        batch_size = params['BATCH_SIZE']
        D_SC = params['D_SCALE_SIZE']
        D = params['D_CROP_SIZE']
        T = params['T_CROP_SIZE']
        stride_t = model.stride_t
        
        ds = PlayerInferenceDataset(cap, vid, sto, courtBF, None, candSF,  
                                    stride_t = stride_t, T = T, D = D, D_SC = D_SC) 
        
        inf_loader = DataLoader(ds, batch_size = batch_size, num_workers = loader_workers, pin_memory = True)
        prbArr, labArr = infer_with_index(model, inf_loader, device = device, report_frequency = 100)
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
        
        labels = params['LABELS']
        srv_idxs = [idx for idx, lab in enumerate(labels) if lab[0] == 'any']
        srv_probs = prbArr[:, srv_idxs].sum(axis = 1)
        
        frm_index_int = frm_index.astype(int)
        probSer = pd.Series(srv_probs, index = frm_index_int)
        # interpolate, so that holes from striding are killed.
        itplSer = probSer.reindex(courtBF.get_frames())
        itplSer = itplSer.interpolate(method = 'index', limit_area = 'inside')
            
        # make boolean series by thresholding
        boolSer = itplSer > serve_prob_threshold
        boolSer = boolSer.rename(f'{player}Player_srvFlag')
        
        boolSF = BoolFeature.from_series(boolSer)
        
        # save also the region of the serving player, so we can initialize ball particles there
        regSF = candSF.filter(boolSF).apply(enclosing_region).rename(f'{player}Player_srvReg')
        regDict[player] = regSF

        # shift the boolean
            # right now serves = true occurs at the moment of the serve impact
            # we want it to be centered at the beginning of the serve, roughly.
        boolSer_shift = boolSer.shift(periods = -serve_frame_shift)
        
        boolSF_shift = BoolFeature.from_series(boolSer_shift)
        boolDict[player] = boolSF_shift
        
    topBF, botBF = boolDict['top'], boolDict['bot']
    topSF, botSF = regDict['top'], regDict['bot']
        
    serveBF = topBF | botBF
    no_serveBF = serveBF.neg()
    playBFs = no_serveBF.filter(no_serveBF).split()    
    
    # StreamList.from_SFList(playBFs, cap).show()
      
    topBF.save(vid, sto)
    botBF.save(vid, sto)
    topSF.save(vid, sto)
    botSF.save(vid, sto)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    