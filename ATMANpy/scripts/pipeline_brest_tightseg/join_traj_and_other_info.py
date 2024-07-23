# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:09:57 2024

@author: gianl
"""

import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.types.Abstract import PlayerMovement, Trajectory, Shot, Shots, Point, Points

from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D, Circle3D, Circles3D, Lines3D

from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous


from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

"""
### Events
# -> F: Flight
# -> STD/STA/SBD/SBA: Serve Top Deuce etc.
# -> HT/HB: Hit Top, Hit Bottom
# -> BI/BO: Bounce In, Bounce Out
# -> N: Net
"""

def load(vid, sto):
    HPCF = ConstFeatures.from_store(vid, sto, featnameList = ['H', 'P', 'M_int', 'M_ext'])
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally_warmup_removed')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    srvSF = StreamFeature.from_store(vid, sto, featname = 'srvFlag')
    scrSF = StreamFeatures.from_store(vid, sto, featnameList = ['sb_name_lower_Flag', 'sb_name_upper_Flag', 'sb_score_lower_Flag', 'sb_score_upper_Flag', 'sb_games_lower_cleaned_Flag', 'sb_games_upper_cleaned_Flag'])
    trajFrm = ff.read_feature_frame('trajTable', vid, sto, keep_only = 'last')
    return HPCF, segBF, plySF, srvSF, scrSF, trajFrm

def save(trajDFs, vid, sto):
    trajFrm = pd.concat(trajDFs)
    ff.save_feature_frame(trajFrm, 'dataTable', vid, sto)

def main(conf, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    courtLines = CourtLines.make().add_singles_netline()

    HPCF, segBF, plySF, srvSF, scrSF, trajFrm = load(vid, sto)
    # trajFrm = fix_traj_frame(trajFrm)
    trajSF = StreamFeatures.from_dataframe(trajFrm)
    trajSF = trajSF.merge(srvSF.rename('event_info'))

    # get player names
    player_names = list(trajFrm.player_name.unique())
    name_hand_map = {}
    for pl in player_names:
        hands = trajFrm.loc[trajFrm.player_name == pl, 'player_hand'].unique()
        assert len(hands) == 1
        name_hand_map[pl] = hands[0]

    ### find 3D player positions
    mySF = plySF.merge(HPCF['P'])
    ply3DSF = mySF.apply(sa.find_player_positions_3D)
    
    topSF = ply3DSF['topPlayer_3D'].apply(sa.point3D_to_coords)
    botSF = ply3DSF['botPlayer_3D'].apply(sa.point3D_to_coords)
    
    topSFs = topSF.conform(segBF).filter(segBF).split()
    botSFs = botSF.conform(segBF).filter(segBF).split()
    trajSFs = trajSF.conform(segBF).filter(segBF).split()
    scrSFs = scrSF.conform(segBF).filter(segBF).split()

    ### Prepare some utility functions for player name and handedness
    player_cols = ['player_name', 'player_hand']
    def get_fs(mySer):
        return mySer.iloc[0]
    def get_side(mySer):
        evt = get_fs(mySer)
        if evt[1] == 'T':
            return 'top'
        elif evt[1] == 'B':
            return 'bot'
    def swap_player(plStr, player_names = player_names):
        assert plStr in player_names
        return [pl for pl in player_names if pl != plStr][0]
       
    ### Postprocess Player Names from Scoreboard
    lower_names = scrSFs['sb_name_lower_Flag'].apply_whole(lambda sf: sf.first()[1])
    upper_names = scrSFs['sb_name_upper_Flag'].apply_whole(lambda sf: sf.first()[1])

    lower_name = pd.Series(lower_names).value_counts().index[0]
    upper_name = pd.Series(upper_names).value_counts().index[0]

    proper_name_1, proper_name_2 = trajSF['player_name'].unique()

    available_surnames = [full_name.split(' ')[1] for full_name in [proper_name_1, proper_name_2]]
    lower_match_ratios = [fuzz.partial_ratio(lower_name.upper(), surname.upper()) / 100 for surname in available_surnames]
    upper_match_ratios = [fuzz.partial_ratio(upper_name.upper(), surname.upper()) / 100 for surname in available_surnames]

    ratio_1 = lower_match_ratios[0] * upper_match_ratios[1]
    ratio_2 = lower_match_ratios[1] * upper_match_ratios[0]
    
    if ratio_1 > ratio_2:
        nameMap = {proper_name_1: lower_name,
                   proper_name_2: upper_name}
        print(f'Info Names to Scoreboard Names (confidence: {ratio_1}):')
        print(nameMap)
        lower_upper_names_proper = [proper_name_1, proper_name_2]
    else:
        nameMap = {proper_name_1: upper_name,
                   proper_name_2: lower_name}
        print(f'Info Names to Scoreboard Names (confidence: {ratio_2}):')
        print(nameMap)
        lower_upper_names_proper = [proper_name_2, proper_name_1]

    ### Use correspondence to make a score frame by player
    scoreFrm = scrSF[['sb_score_lower_Flag', 'sb_score_upper_Flag']].to_dataframe()
    scoreFrm.columns = lower_upper_names_proper

    gamesFrm = scrSF[['sb_games_lower_cleaned_Flag', 'sb_games_upper_cleaned_Flag']].to_dataframe()
    gamesFrm.columns = lower_upper_names_proper

    for c in scoreFrm:
        if scoreFrm[c].dtype != 'object': ### this happens if the player in column c has never reached Advantage ('A'), thus the column is read in as numeric
            scoreFrm[c] = scoreFrm[c].fillna(0)
            scoreFrm[c] = scoreFrm[c].astype('int').astype('object')
        else:
            scoreFrm[c] = scoreFrm[c].fillna('0')
        
    # -> replacing weird numbers. Very Crude, but hey.
    print('Before value replacement: ')
    print('   unique scores p1: ', list(scoreFrm[proper_name_1].unique()))
    print('   unique scores p2: ', list(scoreFrm[proper_name_2].unique()))

    if conf['video_id'] == '2021-7341-ms012-brest-maxime-janvier-vs-hugo-gaston':
        replDict = {'16': '15',
                    '18': '15',
                    'D\n7': '0',
                    '07': '1',
                    '20': '2',
                    '7': '0'} 
    elif conf['video_id'] == '2021-7341-ms014-brest-federico-gaio-vs-frederico-ferreira-silva':
        replDict = {'16': '15',
                    '18': '15',
                    'D\n7': '0',
                    '07': '0',
                    '20': '2',
                    '7': '0'} 
    else:
        replDict = {'16': '15',
                    '18': '15'}
    scoreFrm = scoreFrm.replace(replDict)

    print('After value replacement: ')
    print('   unique scores p1: ', list(scoreFrm[proper_name_1].unique()))
    print('   unique scores p2: ', list(scoreFrm[proper_name_2].unique()))
    
    # split
    scoreFrms = ut.split(scoreFrm)
    gamesFrms = ut.split(gamesFrm)

    frmList = []
    for idx in range(trajSFs.length()):
        trjFrm = trajSFs[idx].to_dataframe()
        trjFrm = trjFrm.dropna(how = 'all', axis = 0) ### narrow it down to where info about the shot actually exists (this way, we kill stuff that happens before ball leaves the racket on serve)

        plyFrm = trjFrm[player_cols]
        trjFrm = trjFrm[[c for c in trjFrm if c not in player_cols]]

        ### player movements top and bot
        topFrm = topSFs[idx].to_dataframe().reindex(trjFrm.index)
        botFrm = botSFs[idx].to_dataframe().reindex(trjFrm.index)

        ### Create some series to be used or added
        sideSer = trjFrm.groupby('shot_idx')['event'].transform(get_side)
        srvSer = trjFrm.groupby('shot_idx')['event_info'].transform(get_fs)

        trjFrm['point_idx'] = idx

        ### Extend the first_serve/second_serve flag to the whole serve traj
        trjFrm['event_info'] = srvSer

        ### Make a player frame with hitting and receiving player data: x, y, name and hand
        plyFrm.columns = plyFrm.columns.str.replace('player', 'hit')

        plyFrm['hit_side'] = sideSer

        hit_x = np.where(sideSer == 'top', topFrm['x'], botFrm['x'])
        hit_y = np.where(sideSer == 'top', topFrm['y'], botFrm['y'])
        rec_x = np.where(sideSer == 'bot', topFrm['x'], botFrm['x'])
        rec_y = np.where(sideSer == 'bot', topFrm['y'], botFrm['y'])

        plyFrm['hit_x'], plyFrm['hit_y'] = hit_x, hit_y

        plyFrm['rec_name'] = plyFrm['hit_name'].map(swap_player)
        plyFrm['rec_hand'] = plyFrm['rec_name'].map(name_hand_map)

        plyFrm['rec_side'] = sideSer.replace({'top': 'bot', 'bot': 'top'})

        plyFrm['rec_x'], plyFrm['rec_y'] = rec_x, rec_y

        ### Merge in the score information 
        scoreFrm = scoreFrms[idx]
        p1, p2 = scoreFrm.columns
        p1_vals, p2_vals = scoreFrm[p1].unique(), scoreFrm[p2].unique()
        assert len(p1_vals) == 1 and len(p2_vals) == 1
        p1_val, p2_val = p1_vals[0], p2_vals[0]
        scoreDict = {p1: p1_val, p2: p2_val}
        
        plyFrm['hit_score'] = plyFrm['hit_name'].map(scoreDict)
        plyFrm['rec_score'] = plyFrm['rec_name'].map(scoreDict)

        ### Merge in the set information
        gamesFrm = gamesFrms[idx]
        p1, p2 = gamesFrm.columns
        p1_vals, p2_vals = gamesFrm[p1].unique(), gamesFrm[p2].unique()
        assert len(p1_vals) == 1 and len(p2_vals) == 1

        p1_val, p2_val = p1_vals[0], p2_vals[0]
        p1_vals, p2_vals = [int(x) for x in p1_val.split('-')], [int(x) for x in p2_val.split('-')]
        assert len(p1_vals) == len(p2_vals)

        p1_sets, p2_sets = 0, 0
        for p1_set_score, p2_set_score in zip(p1_vals, p2_vals):
            if p1_set_score == 7 or (p1_set_score == 6 and p2_set_score <= 4):
                p1_sets += 1
            elif p2_set_score == 7 or (p2_set_score == 6 and p1_set_score <= 4):
                p2_sets += 1
            else:
                p1_games = p1_set_score
                p2_games = p2_set_score

        if len(p1_vals) == p1_sets + p2_sets: ### if the set just ended
            p1_games, p2_games = 0, 0

        gamesDict = {p1: p1_games, p2: p2_games}
        setsDict = {p1: p1_sets, p2: p2_sets}
        del p1_games, p2_games
        
        plyFrm['hit_games'] = plyFrm['hit_name'].map(gamesDict)
        plyFrm['rec_games'] = plyFrm['rec_name'].map(gamesDict)
        
        plyFrm['hit_sets'] = plyFrm['hit_name'].map(setsDict)
        plyFrm['rec_sets'] = plyFrm['rec_name'].map(setsDict)

        ### join, and write to list
        allFrm = trjFrm.merge(plyFrm, left_index = True, right_index = True)
        frmList.append(allFrm)

    frmList = [frm for frm in frmList if len(frm) > 0]

    save(frmList, vid, sto)

if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])

    name = ppl._name

    main(conf, name = name)