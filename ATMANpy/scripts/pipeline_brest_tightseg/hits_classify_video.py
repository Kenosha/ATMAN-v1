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
import ATMANpy.funcs.multi_extract_funcs as mx
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region

import torch
from torch.utils.data import DataLoader
from ATMANpy.models.TemporalCNN.nets import SmallFullConvModel
from ATMANpy.models.TemporalCNN.data import UncappedPlayerInferenceDataset
from ATMANpy.models.TemporalCNN.utils import infer_with_index, load_cnn, label_map, sum_over_distinct_labels

def check(vid, sto, name, topFrm, botFrm):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    import pandas as pd
    
    SAVE_NAME = 'hit_classification_video'
    topFrms, botFrms = ut.split(topFrm), ut.split(botFrm)

    base_graph_dir = ut.get_folder_struct('graphs')['pipeline_checks']
    store_dir = os.path.join(base_graph_dir, name, sto)
    
    fig_height = 24
    fig_width = 32
    
    ### establish how many random backgrounds we want to show
    grid_length = 6
    num_backgrounds = grid_length ** 2

    ### tighten the plot
    bleed = 0
    gridspec_kw = {'wspace': 0.05, 'hspace': 0.05}
    
    rng = np.random.default_rng()
    chosen_indices = rng.choice(np.arange(0, len(topFrms), dtype = int), size = num_backgrounds)
        
    fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
    
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for plot_idx, ax in enumerate(axes.ravel()):
        frame_idx = int(chosen_indices[plot_idx])
        top_subFrm, bot_subFrm = topFrms[frame_idx], botFrms[frame_idx]
        top_plotSer = 1 - top_subFrm['top_oth']
        bot_plotSer = 1 - bot_subFrm['bot_oth']
        top_plotSer.name = 'top_hit'
        bot_plotSer.name = 'bot_hit'

        plotFrm = pd.concat([top_plotSer, bot_plotSer], axis = 1)
        plotFrm.plot(ax = ax)

    store_name = SAVE_NAME + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    fig.savefig(store_path)
    plt.close()
    

def load(vid, sto):
    segBF = BoolFeature.from_store(vid, sto, featname = 'isRally_sound')
    foreStms = StreamList.from_store('foregroundMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask')
    motiStms = StreamList.from_store('fine_motMask', vid, sto, cap_color_space = 'bool', target_color_space = 'mask')
    playersSF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    
    return segBF, foreStms, motiStms, playersSF

def save(topFrm, botFrm, vid, sto):
    ff.save_feature_frame(topFrm, 'top_hitTable', vid, sto)
    ff.save_feature_frame(botFrm, 'bot_hitTable', vid, sto)

def main(conf, model_params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    
    ### additional params which right now can/should not be changed
    loader_workers = 0
    device = 'cuda'
    report_frequency = 100
    streams_to_decap_at_a_time = 80 ### to avoid RAM fuckery
    
    segBF, foreStms, motiStms, playersSF = load(vid, sto)
    
    segBFs = segBF.filter(segBF).split()
    cStms = StreamList.from_SFList(segBFs, cap)
    
    mulStms = StreamList.combine(cStms, foreStms, motiStms)
    streams_to_decap = mulStms.length()
    decap_iterations = streams_to_decap // streams_to_decap_at_a_time + 1
    
    ### Run models on player contours
    prbDict = {'top': [], 'bot': []}
    labDict = {'top': [], 'bot': []}
    prmDict = {'top': [], 'bot': []}
    
    for iter_idx in range(decap_iterations):
        left_iter_bound, right_iter_bound = iter_idx * streams_to_decap_at_a_time, (iter_idx + 1) * streams_to_decap_at_a_time 
        decapStms = mulStms[left_iter_bound: right_iter_bound].apply_chunked_mp(mx.get_masked_video, nwo, nch)

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
            
            ds = UncappedPlayerInferenceDataset(decapStms, playerSF,  
                                                stride_t = stride_t, T = T, D = D, D_SC = D_SC) 
            
            inf_loader = DataLoader(ds, batch_size = batch_size, num_workers = loader_workers, pin_memory = True)
            prbArr, labArr = infer_with_index(model, inf_loader, device = device, report_frequency = report_frequency)
            prbArr = prbArr.squeeze() ### kill T, H, W dims (which remain due to fully convolutional model)
            
            prbDict[player].append(prbArr)
            labDict[player].append(labArr)
            prmDict[player].append(params)
        
        del decapStms, ds, inf_loader ### free up RAM

    for player in ['top', 'bot']:
        prbDict[player] = torch.concat(prbDict[player])
        labDict[player] = torch.concat(labDict[player])
        prmDict[player] = prmDict[player][0]

    ### Aggregate probabilities of hits
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
        prbFrm = prbFrm.reindex(segBF.filter(segBF).get_frames())
        prbFrm = prbFrm.interpolate(method = 'index', limit_area = 'inside')
        prbFrm = prbFrm.dropna()
        
        frmDict[player] = prbFrm

    topFrm, botFrm = frmDict['top'], frmDict['bot']

    check(vid, sto, name, topFrm, botFrm)

    save(topFrm, botFrm, vid, sto)


if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])
    
    model_params = prm_mod.hits_classify_video.model_params
    
    name = ppl._name

    main(conf, model_params, name = name)
        
        
        
        