# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:17:42 2023

@author: gianl
"""


### GET VIDEO ###
import os
import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Streams import StreamList, NCStream
from ATMANpy.types.Features import ConstFeature, BoolFeature, SFList

def load(vid, sto):
    # courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
    segBF = BoolFeature.from_store(vid, sto, featname = 'isRally_sound')
    return segBF

def save(medSFs, stdSFs, vid, sto):
    medSFs.rename('median_backFrm').save(vid, sto)
    stdSFs.rename('std_backFrm').save(vid, sto)

def check(vid, sto, name, medSFs, stdSFs):
    import numpy as np
    import matplotlib.pyplot as plt
    
    SAVE_NAME = 'court_backgrounds'
    
    base_graph_dir = ut.get_folder_struct('graphs')['pipeline_checks']
    store_dir = os.path.join(base_graph_dir, name, sto)

    aspect_ratio = medSFs[0].feature.cols / medSFs[0].feature.rows
    fig_height = 24
    fig_width = fig_height * aspect_ratio
    
    ### establish how many random backgrounds we want to show
    grid_length = 6
    num_backgrounds = grid_length ** 2
    
    ### tighten the plot
    bleed = 0
    gridspec_kw = {'wspace': 0.05, 'hspace': 0.05}
    
    rng = np.random.default_rng()
    chosen_indices = rng.choice(np.arange(0, medSFs.length(), dtype = int), size = num_backgrounds)
    
    for which, sfList in enumerate([medSFs, stdSFs]):
        if which == 0:
            name_addon = 'median'
        elif which == 1:
            name_addon = 'std'
            
        fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
        
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))
    
        for plot_idx, ax in enumerate(axes.ravel()):
            sf_idx = int(chosen_indices[plot_idx])
            chosen_SF = sfList[sf_idx]
            
            sfFrm = chosen_SF.feature.put(f'Rally #: {sf_idx}')
            sfArr = np.flip(sfFrm.array, axis = -1) 
            
            ax.imshow(sfArr)
            ax.set_axis_off()
        
        store_name = SAVE_NAME + '_' + name_addon + '_' + vid + '.png'
        store_path = os.path.join(store_dir, store_name)
        
        fig.savefig(store_path)
        plt.close()
        
    
def main(conf, params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = conf['cap']
    
    segBF = load(vid, sto)
    
    segBFs = segBF.filter(segBF).split()
    
    ### Use the Frames where we did find lines to calculate a background frame!
    # the nice thing is: Since the homography is constant, there is no need to warp!!!
    # We basically construct a NCDecapStream from where a homography has been found,
    # and calculate mean and std over all constituent frames.
    # This will constitute our background (for object detection).
    
    ### Build an NCStream which holds all frames from which we can sample a valid court frame
    ### for construction of background frames for each member of courtStms
    # TODO: Maybe widen the segmentation for this, so we sample frames only from what is truly play, and not
    # sequences when the court is being shown but the lights are different and there is no play? (weird changeovers? check frame 24000)
    # Stream(cap, START_FRAME, STOP_FRAME).get(24000).show()
    ### trunc_n addresses the above issue.
    
    medList = []
    stdList = []
    for idx in range(segBFs.length()):
        ### get the leftmost and rightmost indices in scope: window_range points around the current point
        left_idx = max(0, idx - params['window_range'])
        right_idx = min(idx + params['window_range'] + 1, segBFs.length())
        windowBFs = segBFs[left_idx:right_idx]
        
        playStms = StreamList.from_SFList(windowBFs, cap).widen(params['widen_play_by_frames'])
        playStm = playStms.join()
        
        if playStm.length() < params['sample_frames']:
            print('Too few frames in courtStm, consider reducing TRUNC_N')
            continue
        
        sampledStm = playStm.sample(params['sample_frames'])
        sampledFrms = sampledStm.decap().frames
        
        medFrm, stdFrm = sampledFrms.median(), sampledFrms.std()
        medFrm, stdFrm = medFrm.rename('median_backFrm'), stdFrm.rename('std_backFrm')
        
        frames = segBFs[idx].get_frames()
        medCF = ConstFeature('median_backFrm', medFrm, frameList = frames)
        stdCF = ConstFeature('std_backFrm', stdFrm, frameList = frames)
        
        medList.append(medCF)
        stdList.append(stdCF)
    
    medSFs = SFList(medList)
    stdSFs = SFList(stdList)
    
    check(vid, sto, name, medSFs, stdSFs)
    
    save(medSFs, stdSFs, vid, sto)
    
    
if __name__ == '__main__':
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
   
    avail_vids = [f for f in ut.get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])
    
    params = prm_mod.court_background.params

    name = ppl._name

    main(conf, params, name = name)
    