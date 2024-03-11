# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:16:32 2023

@author: gianl
"""


### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video

import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex

from ATMANpy.types.Streams import Stream
from ATMANpy.types.Features import ConstFeature, SFList

def load(vid, sto):
    HSF = ConstFeature.from_store(vid, sto, featname = 'H')
    return HSF

def check(matchSF):
    matchSF.to_dataframe().plot() ### check this, to see where to set thresholds!!
    
def save(vid, sto, courtBF_without_short_sequences):
    courtBF_without_short_sequences.rename('isCourt').save(vid, sto)
    
def main(conf, court_match_params, threshold_params, filter_params):
    
    ### COLLECT CONF PARAMS ###
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo = conf['num_workers']
    cap = get_video(vid)
    
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
    check(matchSF)
    
    ### SAVE ###
    save(vid, sto, courtBF_without_short_sequences)
    
    
if __name__ == '__main__':

    from ATMANpy.funcs.util_funcs import read_config    
    conf = read_config('brest', config_name = '_config_pipe2')
    
    WIDTH_OF_LINE_AT_FULL_HD = 8
    WIDTH_OF_LINE = WIDTH_OF_LINE_AT_FULL_HD * conf['scale_factor']
    court_match_params = {'width_of_line': WIDTH_OF_LINE, 
                          'int_diff_thresh': 10,
                          'white_thresh': 150,
                          'zoom_factor': 1.2,
                          'dil_kernel': (2, 2)}
    
    threshold_params = {'fp_threshold': None, 'fn_threshold': 0.3}
    
    filter_params = {'min_seq_length': 100}
    
    main(conf, court_match_params, threshold_params, filter_params)
    