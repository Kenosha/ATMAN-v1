# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:26:01 2023

@author: gianl
"""


### GET VIDEO ###
import os

import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.extract_funcs as ex

from ATMANpy.types.Streams import Stream, StreamList
from ATMANpy.types.Features import StreamFeature, BoolFeature
from ATMANpy.types.Contours import Contours

from ATMANpy.models.BallSVM.utils import load_svm

def load(vid, sto):
    ### Load Movement Contours ###
    cntSF = StreamFeature.from_store(vid, sto, featname = 'movCnts')
    playBF = BoolFeature.from_store(vid, sto, featname = 'isRally_sound')
    return cntSF, playBF

def save(candSFs, vid, sto, compression = 24):
    candSFs['cand0Cnts'].rename('cand0Cnts').save(vid, sto, compression = compression)
    candSFs['cand1Cnts'].rename('cand1Cnts').save(vid, sto, compression = compression)
    candSFs['cand2Cnts'].rename('cand2Cnts').save(vid, sto, compression = compression)

def check(vid, sto, name, candSFs, cap):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    from ATMANpy.funcs.util_funcs import get_folder_struct
    from ATMANpy.types.Frames import GreyFrame
    
    SAVE_NAME = 'ball_chosen_contours'
    
    base_graph_dir = get_folder_struct('graphs')['pipeline_checks']
    store_dir = os.path.join(base_graph_dir, name, sto)
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    aspect_ratio = width / height
    fig_height = 42
    fig_width = fig_height * aspect_ratio
    
    ### establish how many random backgrounds we want to show
    grid_length = 10
    num_backgrounds = grid_length ** 2
    
    ### tighten the plot
    bleed = 0
    gridspec_kw = {'wspace': 0.0, 'hspace': 0.05}
    
    rng = np.random.default_rng()
    chosen_indices = rng.choice(np.arange(0, candSFs.length(), dtype = int), size = num_backgrounds)
        
    fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
    
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for plot_idx, ax in enumerate(axes.ravel()):
        sf_idx = int(chosen_indices[plot_idx])
        chosenSF = candSFs[sf_idx]
        frame = int(rng.choice(chosenSF.get_frames(), size = 1)[0])
        featDict = chosenSF[frame]

        showFrm = GreyFrame.from_video(cap, frame).to_BGR()
        showFrm = showFrm.put(featDict['cand0Cnts'], color = (0, 0, 255), thickness = 2)
        showFrm = showFrm.put(featDict['cand1Cnts'], color = (0, 255, 255), thickness = 2)
        showFrm = showFrm.put(featDict['cand2Cnts'], color = (0, 255, 0), thickness = 2)

        showFrm = showFrm.put(f'Rally #: {sf_idx}', color = (0, 0, 255))
        showFrm = np.flip(showFrm.array, axis = -1) 
        
        ax.imshow(showFrm)
        ax.set_axis_off()
    
    store_name = SAVE_NAME + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    fig.savefig(store_path)
    plt.close()

def main(conf, svm_params, inference_params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    svmDict = load_svm(svm_params['svm_name'])

    cntSF, playBF = load(vid, sto)
    
    cntSFs = cntSF.conform(playBF.filter(playBF)).fillna(Contours([])).split()

    hStms = StreamList.from_SFList(cntSFs, cap, target_color_space = 'hsv')
    gStms = StreamList.from_SFList(cntSFs, cap, target_color_space = 'grey')

    mulStms = StreamList.combine(hStms, gStms).attach(cntSFs)
    
    ### Extract the features from the ball contours
    frmSFs = mulStms.extract_chunked_mp(ex.get_feature_frames, nwo, nch, **svmDict['feature_params'])   
    frmSFs = frmSFs.rename('ballTable')

    ### Find Ball Candidates
    jntSFs = frmSFs.merge(cntSFs)
    candSFs = jntSFs.apply_whole_mp(sf.find_ball_candidates_tiered, nwo, **inference_params, svmDict = svmDict)
    
    check(vid, sto, name, candSFs, cap)

    save(candSFs, vid, sto, compression = conf['compression_ball'])
    
if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[8])
    
    svm_params = prm_mod.ball_inference_svm.svm_params    
    inference_params = prm_mod.ball_inference_svm.inference_params

    name = ppl._name

    main(conf, svm_params, inference_params, name = name)

