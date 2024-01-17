# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:17:42 2023

@author: gianl
"""


### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video

from ATMANpy.types.Streams import StreamList
from ATMANpy.types.Features import ConstFeature, BoolFeature, SFList

def load(vid, sto):
    courtBF = BoolFeature.from_store(vid, sto, featname = 'isCourt')
    return courtBF

def save(medSFs, stdSFs, vid, sto):
    medSFs.rename('median_backFrm').save(vid, sto)
    stdSFs.rename('std_backFrm').save(vid, sto)

def main(conf, params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = get_video(vid)
    
    courtBF = load(vid, sto)
    
    courtBFs = courtBF.filter(courtBF).split()
    courtStms = StreamList.from_SFList(courtBFs, cap)
    
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
    for courtStm in courtStms:
        tightStm = courtStm.widen(-params['trunc_n']) 
    
        if tightStm.length() < params['sample_n']:
            print('Too few frames in courtStm, consider reducing TRUNC_N')
            continue
        
        sampledStm = tightStm.sample(params['sample_n'])
        sampledFrms = sampledStm.decap().frames
        
        medFrm, stdFrm = sampledFrms.median(), sampledFrms.std()
        medFrm, stdFrm = medFrm.rename('median_backFrm'), stdFrm.rename('std_backFrm')
        
        frames = courtStm.get_framelist()
        medCF = ConstFeature('median_backFrm', medFrm, frameList = frames)
        stdCF = ConstFeature('std_backFrm', stdFrm, frameList = frames)
        
        medList.append(medCF)
        stdList.append(stdCF)
    
    medSFs = SFList(medList)
    stdSFs = SFList(stdList)
    
    save(medSFs, stdSFs, vid, sto)
    

if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import read_config
    
    conf = read_config('brest', config_name = '_config_pipe2')
    
    params = {'trunc_n': 1000, 'sample_n': 1000}

    main(conf, params)
    