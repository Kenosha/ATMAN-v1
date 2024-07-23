# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:34:47 2024

Idea: In the first isCourt segment, tag the bottom and the top player, in addition to his handedness
The script then automatically fills in the other isCourt segments (by switching top and bottom alternatingly).
This switching is then checked manually for confirmation.

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

def load(vid, sto):
    crtBF = StreamFeature.from_store(vid, sto, featname = 'isCourt') ### use this as a way of marking changeovers
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally_sound') ### EDIT: IMPORTANT: USED TO BE isRally_warmup_removed, but is circular!
    return crtBF, segBF

def save(vid, sto, topSF, botSF, tophandSF, bothandSF):
    topSF.rename('top_name_Flag').save(vid, sto)
    botSF.rename('bot_name_Flag').save(vid, sto)
    tophandSF.rename('top_hand_Flag').save(vid, sto)
    bothandSF.rename('bot_hand_Flag').save(vid, sto)

def main(conf, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = conf['cap']

    LEFT_HANDED_PLAYERS = ['Hugo Gaston', 
                           'Kenny De Schepper', 
                           'Liam Broady',
                           'Fernando Verdasco',
                           'Frederico Ferreira Silva']
    START_FRAME = 1200

    ### Figure out player names
    def fix_player(plStr):
        plStr = plStr.strip('-')
        name_segments = plStr.split('-')
        name_segments = [seg.capitalize() for seg in name_segments]
        name = ' '.join(name_segments)
        return name

    player_description = vid.split('brest')[1]
    left_player, right_player = player_description.split('vs')
    left_player, right_player = fix_player(left_player), fix_player(right_player)
    
    #### READ IN ####
    crtBF, segBF = load(vid, sto)
    crtBFs = crtBF.filter(crtBF).split()
    
    ### filter out warm up segments
    play_crtBFs = []
    for bf in crtBFs:
        non_warmup_play_BF = bf.filter(segBF.conform(bf).fillna(False))
        if not non_warmup_play_BF.is_empty():
            play_crtBFs.append(bf)

    play_crtBFs = SFList(play_crtBFs)

    ### Do annotation for every segment separately -> there could irregular side switching at set changes 
    outSFs = []
    for bf in play_crtBFs:
        is_decided = False
        while not is_decided:
            showBF = bf.conform(bf.get_frames()[START_FRAME:])
            Stream.from_SF(showBF, cap).show(ms_between_frames = 10)

            ### Tag top and bottom player
            out = input(f'Top Player is 1) {left_player} or 2) {right_player}? ')
            if out == '1':
                top_player = left_player
                bot_player = right_player
                is_decided = True
            elif out == '2':
                top_player = right_player
                bot_player = left_player
                is_decided = True

        ### Prep FeatDict
        top_hand = 'left' if top_player in LEFT_HANDED_PLAYERS else 'right'
        bot_hand = 'left' if bot_player in LEFT_HANDED_PLAYERS else 'right'

        feature_dict = {'top_name_Flag': top_player, 'top_hand_Flag': top_hand,  
                        'bot_name_Flag': bot_player, 'bot_hand_Flag': bot_hand}

        ### Append to List
        mySF = StreamFeatures({f: feature_dict for f in bf.get_frames()})
        outSFs.append(mySF)

    outSFs = SFList(outSFs)

    for sf in outSFs:
        showSF = sf.conform(sf.get_frames()[START_FRAME:])
        Stream.from_SF(showSF, cap).rescale(2).show(ms_between_frames = 3)

    outSF = outSFs.concatenate()

    topSF, botSF, tophandSF, bothandSF = outSF['top_name_Flag'], outSF['bot_name_Flag'], outSF['top_hand_Flag'], outSF['bot_hand_Flag']

    save(vid, sto, topSF, botSF, tophandSF, bothandSF)

if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[1])

    name = ppl._name
    main(conf, name = name)



