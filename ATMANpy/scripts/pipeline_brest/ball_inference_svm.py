# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:26:01 2023

@author: gianl
"""


### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video

import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.extract_funcs as ex

from ATMANpy.types.Streams import StreamList
from ATMANpy.types.Features import StreamFeature, BoolFeature
from ATMANpy.types.Contours import Contours

from ATMANpy.models.BallSVM.utils import load_svm

def load(vid, sto):
    ### Load Movement Contours ###
    cntSF = StreamFeature.from_store(vid, sto, featname = 'movCnts')
    playBF = BoolFeature.from_store(vid, sto, featname = 'isRally')
    return cntSF, playBF

def save(candSFs, vid, sto):
    candSFs.rename('candCnts').save(vid, sto, compression = conf['compression_ball'])

def main(conf, svm_params, inference_params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    
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
    candSFs = jntSFs.apply_whole_mp(sf.find_ball_candidates, nwo, **inference_params, svmDict = svmDict)

    save(candSFs, vid, sto)
    
if __name__ == '__main__':

    from ATMANpy.funcs.util_funcs import read_config    
    
    conf = read_config('brest', config_name = '_config_pipe2')

    svm_params = {'svm_name': 'svm_proba_dict_r75_p60_2023-08-15'}    
    inference_params = {'use_training_moments': True,
                        'probability_threshold': None}

    main(conf, svm_params, inference_params)

