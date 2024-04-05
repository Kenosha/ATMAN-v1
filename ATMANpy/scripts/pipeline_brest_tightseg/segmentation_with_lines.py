# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:16:32 2023

@author: gianl
"""

import os
import matplotlib.pyplot as plt

import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Streams import Stream
from ATMANpy.types.Features import ConstFeature, SFList

def load(vid, sto):
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    return HSF

def check(vid, sto, name, matchSF):
    base_graph_dir = ut.get_folder_struct('graphs')['pipeline_checks']
    store_dir = os.path.join(base_graph_dir, name, sto)
    store_name = 'court_match_rates' + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    fig, ax = plt.subplots(1, 1, figsize = (16, 10))
    matchSF.to_dataframe().plot(ax = ax) ### check this, to see where to set thresholds!!
    fig.savefig(store_path)
    plt.close()
    
def save(vid, sto, courtBF_without_short_sequences):
    courtBF_without_short_sequences.rename('isCourt').save(vid, sto)
    
def main(conf, court_match_params, threshold_params, filter_params, name = None):
    
    ### COLLECT CONF PARAMS ###
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo = conf['num_workers']
    cap = conf['cap']
    
    ### LOAD ###
    HSF = load(vid, sto)
    
    ### DO STUFF ###
    H = HSF.feature
    courtLines = sa.find_courtlines(H)
    courtlineSF = ConstFeature('courtLines', courtLines)
    
    gStm = Stream(cap, conf['start_frame'], conf['stop_frame'], target_color_space = 'grey').attach(courtlineSF)
    matchSF = gStm.extract_mp(ex.court_match_rates, nwo, **court_match_params)
    
    courtBF = matchSF.apply(sa.threshold_false_positives_negatives, **threshold_params).to_bool()
    
    ### filter out sequences which are too short
    courtBFs = courtBF.filter(courtBF).split()
    courtBFs = SFList([myBF for myBF in courtBFs if myBF.length() > filter_params['min_seq_length']])
    courtBF_without_short_sequences = courtBFs.concatenate_and_plug().to_bool().fillna(False)
    
    ### CHECK ###
    check(vid, sto, name, matchSF)
    
    ### SAVE ###
    save(vid, sto, courtBF_without_short_sequences)
    
    
if __name__ == '__main__':

    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
    
    conf = cnf_mod.conf
    court_match_params = prm_mod.segmentation_with_lines.court_match_params
    threshold_params = prm_mod.segmentation_with_lines.threshold_params
    filter_params = prm_mod.segmentation_with_lines.filter_params
    
    name = ppl._name
    
    main(conf, court_match_params, threshold_params, filter_params, name = name)
    