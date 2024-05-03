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
    trajFrm = ff.read_feature_frame('trajTable', vid, sto)
    return HPCF, segBF, plySF, srvSF, trajFrm

def save(trajDFs, vid, sto):
    trajFrm = pd.concat(trajDFs)
    ff.save_feature_frame(trajFrm, 'dataTable', vid, sto)

def main(conf, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    
    courtLines = CourtLines.make().add_singles_netline()

    HPCF, segBF, plySF, srvSF, trajFrm = load(vid, sto)
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