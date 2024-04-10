# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:34:47 2024

@author: gianl
"""


import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Visual3D import Scene, Camera, View

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D

def fix_traj_frame(trajFrm):
    ### TODO: this can go as soon as ball_interpolation_plane is re-run 
    ### with overheads recognized as serves that have a serve event (SBA, SBD, STA, STD) at shot_idx > 0
    not_first = trajFrm.shot_idx > 0
    serve_recognized = trajFrm.event.isin(['SBA', 'SBD', 'STA', 'STD'])
    to_fix = not_first & serve_recognized
    print(f'Fixing {to_fix.sum()} instances of serves that are recognized during the rally.')
    trajFrm.event = np.where(not_first & serve_recognized, 'H' + trajFrm.event.str[1], trajFrm.event)
    return trajFrm

def load(vid, sto):
    crtBF = StreamFeature.from_store(vid, sto, featname = 'isCourt') ### use this as a way of marking changeovers
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally_warmup_removed') ### TODO: once we remove warmups earlier in the pipeline, this will not be necessary!
    trajFrm = ff.read_feature_frame('trajTable', vid, sto)
    return crtBF, segBF, trajFrm

def save(vid, sto, outSF):
    outSF.rename('srvFlag').save(vid, sto)

def main(conf, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = conf['cap']
    
    #### READ IN ####
    crtBF, segBF, trajFrm = load(vid, sto)
    trajFrm = fix_traj_frame(trajFrm)

    srvFrm = trajFrm[trajFrm.event.isin(['SBA', 'SBD', 'STA', 'STD'])]
    srvSF = StreamFeature.from_series(srvFrm['event'])
    srvSF = srvSF.filter(segBF) ### this amounts to removing the warm-up serves! TODO: will not be necessary once we remove warmups earlier.

    srvSFs = srvSF.conform(crtBF).filter(crtBF).split()
    srvSFs = srvSFs.dropna()

    ### How to check whether a serve is first or second serve:
    ### if there are no lets, and every serve is caught, then looking at consecutive serve events suffices
    ### we could mark lets by hand, or train a model on audio ('out', 'let')
    ### for the moment, we assume that the isRally (or maybe -> isRally_audio?) Flag catches all rallyes.
    
    def serve_map(srv_split):
        num_serves = len(srv_split)

        if num_serves == 1:
            return ['first_serve']
        elif num_serves == 2:
            return ['first_serve', 'second_serve']
        elif num_serves == 3:
            return ['fill_in'] * num_serves #['let', 'first_serve', 'second_serve']
        else:
            return ['fill_in'] * num_serves
            

    #### CHECK FOR INTEGRITY ###
    outDict = {}
    for seg_idx, sf in enumerate(srvSFs):
        serve_frames = list(sf.streamDict.keys())
        serve_flags = list(sf.streamDict.values())
        ### split into consecutive serves
        serve_splits = ut.split_into_contiguous(serve_flags, offset = '')
        frame_splits = ut.split_like(serve_frames, serve_splits)

        for frames, serves in zip(frame_splits, serve_splits):
            if len(frames) <= 2:
                ### normal -> mapped as above!
                serves_mapped = serve_map(serves)
                outDict.update(dict(zip(frames, serves_mapped)))
            else:
                serve_event = serves[0]
                print(f'    {len(frames)} consecutive {serve_event} at {frames}')

                ### FILL IN THE WEIRD SEGMENTS!
                filled_dict = {}
                while len(filled_dict) < len(frames):
                    Stream(cap, frames[0] - 25, frames[-1] + 25).show(frame_counter = True)
                    for idx, frame in enumerate(frames):
                        if frame in filled_dict:
                            continue

                        print(f'    Categorizing Serve Number {idx} at frame {frame} ...')
                        serve_event_key = input(f'      ... 1 for 1st, 2 for 2nd, 0 for let, x for not a serve, any other key for replay: ')
                        if serve_event_key == '1':
                            filled_dict[frame] = 'first_serve'
                        elif serve_event_key == '2':
                            filled_dict[frame] = 'second_serve'
                        elif serve_event_key == '0':
                            filled_dict[frame] = 'let'
                        elif serve_event_key == 'x':
                            filled_dict[frame] = 'not a serve'
                        else:
                            continue

                outDict.update(filled_dict)

    outSF = StreamFeature(outDict, name = 'srvFlag')

    save(vid, sto, outSF)

if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[1])

    name = ppl._name
    main(conf, name = name)



