# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:34:47 2024

Idea: In manual_player_names_and_handedness, we assigned a player name (and his dominant hand) to each side (bot/top).
This is fine for all situations except tiebreaks, where sides can switch within an isCourt segment.

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
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally_warmup_removed')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['top_name_Flag', 'top_hand_Flag', 'bot_name_Flag', 'bot_hand_Flag'])
    return crtBF, segBF, plySF

def save(vid, sto, topSF, botSF, tophandSF, bothandSF):
    topSF.rename('top_name_tb_Flag').save(vid, sto)
    botSF.rename('bot_name_tb_Flag').save(vid, sto)
    tophandSF.rename('top_hand_tb_Flag').save(vid, sto)
    bothandSF.rename('bot_hand_tb_Flag').save(vid, sto)

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
    crtBF, segBF, plySF = load(vid, sto)
    crtBFs = crtBF.filter(crtBF).split()
    
    ### filter out warm up segments
    play_crtBFs = []
    for bf in crtBFs:
        non_warmup_play_BF = bf.filter(segBF.conform(bf).fillna(False))
        if not non_warmup_play_BF.is_empty():
            play_crtBFs.append(bf)

    play_crtBFs = SFList(play_crtBFs)

    ### Do annotation for every segment separately -> there could irregular side switching at set changes 
    switchSFs = []
    for bf in play_crtBFs:
        Stream.from_SF(bf, cap).show()
        can_have_tiebreak = input(f'Can this segment contain a tiebreak? (equivalent to: does this segment begin at 6:5 or 5:6)? y/n: ')
        if can_have_tiebreak == 'n':
            continue
        else:
            splitSFs = plySF.conform(bf).filter(segBF.conform(bf)).split()

            for sf in splitSFs:
                is_decided = False
                while not is_decided:
                    Stream.from_SF(sf, cap).rescale(2).show()

                    top_player = sf['top_name_Flag'].unique()
                    bot_player = sf['bot_name_Flag'].unique()
                    top_hand = sf['top_hand_Flag'].unique()
                    bot_hand = sf['bot_hand_Flag'].unique()
                    assert len(top_player) == 1, f'Non-unique top_player in one segment: {top_player}'
                    top_player = top_player[0]
                    assert len(bot_player) == 1, f'Non-unique bot_player in one segment: {bot_player}'
                    bot_player = bot_player[0]
                    assert len(top_hand) == 1, f'Non-unique top_hand in one segment: {top_hand}'
                    top_hand = top_hand[0]
                    assert len(bot_hand) == 1, f'Non-unique bot_hand in one segment: {bot_hand}'
                    bot_hand = bot_hand[0]

                    do_switch = input(f'Top Player is {top_player} -- Should we switch? y/n: ')
                    
                    if do_switch == 'y':
                        switch_featDict = {'top_name_Flag': bot_player, 'top_hand_Flag': bot_hand,
                                        'bot_name_Flag': top_player, 'bot_hand_Flag': top_hand}
                        switchSF = StreamFeatures({f: switch_featDict for f in sf.get_frames()})
                        switchSFs.append(switchSF)
                        is_decided = True
                    elif do_switch == 'n':
                        is_decided = True
                    else:
                        is_decided = False
                
    switchSFs = SFList(switchSFs)

    for sf in switchSFs:
        Stream.from_SF(sf, cap).rescale(2).show(ms_between_frames = 5)

    outSF = switchSFs.concatenate()

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



