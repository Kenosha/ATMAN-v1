# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 03:33:20 2022

@author: gianl
"""

import os
import pandas as pd
import numpy as np

from ATMANpy.funcs.HMM.utils import read_train_test

from ATMANpy.funcs.util_funcs import get_video        
from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous
from ATMANpy.funcs.draw_funcs import draw
from ATMANpy.funcs.cv_funcs import replay

import ATMANpy.auxil.constants as const


VIDEO_ID = 'pqxAmUWIoKc'
MODEL_NAME = 'embedded_cyc3_nrel85'

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = get_video(VIDEO_ID)
################

### STORENAME ###
STORE_NAME = '10to100_ok2'
#################

train_data, test_data = read_train_test(MODEL_NAME)
 

#%% LOAD FEATURES

# test_data.IndexSF.unique() 
SHOW_SEQUENCE = 108
INTERPOLATION_TYPE = 'smart'

### GET STREAM FEATURES ###
from ATMANpy.funcs.feature_funcs import load_stored_features
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)

### GET 3D FEATURES ###

from ATMANpy.funcs.feature_funcs import read_feature_frame

try:
    ball_3D_ground = read_feature_frame(f'3D_Ball_{INTERPOLATION_TYPE}_ground', VIDEO_ID, STORE_NAME)
except FileNotFoundError:
    ball_3D_ground = pd.DataFrame()

try:
    ball_3D_infer = read_feature_frame(f'3D_Ball_{INTERPOLATION_TYPE}_inference', VIDEO_ID, STORE_NAME)
except FileNotFoundError:
    ball_3D_infer = pd.DataFrame()

try:
    players_3D = read_feature_frame('3D_players', VIDEO_ID, STORE_NAME)
except FileNotFoundError:
    players_3D = pd.DataFrame()

import cv2 as cv


### GET TEST SEQUENCE STREAM FEATURES, EVENTS
from ATMANpy.funcs.HMM.HPost import merge_words_and_event_features, narrow_down_instantaneous_events
seq_data_ground = test_data[test_data.IndexSF == SHOW_SEQUENCE] ### Event, here is what was actually marked!
seq_data_hidden = seq_data_ground.drop(columns = ['Event', 'IndexSF']) ### Drop it, and merge in inference 

test_sequence_features = {frame: featDict for frame, featDict in stream_features.items() if frame in seq_data_ground.index}
event_features = {frame: featDict['grammatic_event'] for frame, featDict in test_sequence_features.items()}
event_features = narrow_down_instantaneous_events(event_features)
# seq_data_infer = merge_words_and_event_features(seq_data_hidden, event_features)
event_features = pd.Series(event_features)
event_features = event_features[event_features != 'None']
event_frames = event_features.index

ball_3D_infer_show = ball_3D_infer[ball_3D_infer.index.isin(event_frames)]
ball_3D_ground_show = ball_3D_ground[ball_3D_ground.index.isin(event_frames)]
players_3D_show = players_3D[players_3D.index.isin(event_frames)]
    
#%% CREATE ADDITIONAL FEATURES 

import ATMANpy.funcs.HMM.HPost as HPost


# # a) for inference
# HPost.make_3D_features(seq_data_infer, ball_3D_infer_show, players_3D_show)

###### SPIN #######

### read theoretical trajectories
import ATMANpy.funcs.spin_funcs as sf
groundArr, vertArr, trajDict = sf.load_trajectories('TrajectoriesRPM', dtype = 'float32')

### initialise search spaces with proper arguments
import inspect
kwargList = inspect.getargspec(sf.initialize).args
kwargs = {k: v for k, v in trajDict.items() if k in kwargList}
v_init_ground_space, v_init_vert_space, spin_rate_space = sf.initialize(dtype = 'float32', 
                                                                        squeeze = True, 
                                                                        **kwargs)

## first: separate out trajectories
flight_events = event_features[event_features.isin(['FLIGHT_UP', 'FLIGHT_DOWN'])]
from ATMANpy.funcs.util_funcs import split_into_contiguous, widen_list

def traj_is_hit_or_serve(trajFrm):
    event_at_start = trajFrm.iloc[0]
    return event_at_start[0] in ['S', 'H'] ## serve or hit!

trajectory_frames = split_into_contiguous(flight_events.index)
trajectory_frames = [widen_list(tList, by = 1) for tList in trajectory_frames]
trajectory_frames = [ts for ts in trajectory_frames if traj_is_hit_or_serve(event_features[ts])]

print('Calculating Spins...')
data_frames = []
for tList in trajectory_frames:
    trajFrm = ball_3D_infer_show.loc[tList]
    
    ### Calculate properties of measured flight trajectory: initial velocities, position.
    pos_ground_measured, pos_vert_measured = sf.calculate_planar_ball_flight(trajFrm)
    v_ground_idx, v_vert_idx, rpm_idx = sf.get_optimal_trajectory_indices(groundArr, vertArr, 
                                                                          pos_ground_measured, pos_vert_measured)
    
    v_init_ground = v_init_ground_space[v_ground_idx]
    v_init_vert = v_init_vert_space[v_vert_idx]
    spin = spin_rate_space[rpm_idx]
    v_init = (v_init_ground ** 2 + v_init_vert ** 2) ** 0.5
    
    trajData = pd.concat([pd.Series(v_init_ground, index = tList, name = 'v_x'),
                          pd.Series(v_init_vert, index = tList, name = 'v_z'),
                          pd.Series(v_init, index = tList, name = 'v'),
                          pd.Series(spin, index = tList, name = 'spin')], axis = 1)
    data_frames.append(trajData)
print('... Done')
   
point_data = pd.concat(data_frames)
# there can be duplication in case we have a short hit event -> in this case, we take the more recent shot.
point_data = point_data[~point_data.index.duplicated(keep = 'last')] 
point_data = point_data.reindex(event_features.index).ffill()

### TODO: Check if you can speed up the routine by limiting search to initial conditions
### that are "not too far" from the initial conditions as measured 
### (see code below to find a search starting point)
# v_init_ground_measured, v_init_vert_measured = sf.calculate_initial_conditions(trajFrm, frames_to_use = 1)
# ground_initial_index = np.abs(v_init_ground_space - v_init_ground_measured).argmin()
# vert_initial_index = np.abs(v_init_vert_space - v_init_vert_measured).argmin()

# pos_ground_theoretical = groundArr[ground_initial_index, vert_initial_index, :, :len(tList)]
# pos_vert_theoretical = vertArr[ground_initial_index, vert_initial_index, :, :len(tList)]



#%% #### VISUALIZE #########

MS_BETWEEN_FRAMES = 40
SHOW_VIDEO = False
SHOW_GROUND = False
SHOW_INFERENCE = True

BACKGROUND_RES = 1
FOREGROUND_RES = 1

### 1) Real Video
if SHOW_VIDEO:
    frame_min, frame_max = min(seq_data_ground.index), max(seq_data_ground.index)
    show_video(cap, 
                start_frame = frame_min, stop_frame = frame_max,
                ms_between_frames = MS_BETWEEN_FRAMES)

# ### 2) Generated Views
from ATMANpy.types.Visual3D import View, Camera, Scene
from ATMANpy.types.ShapesTennis3D import Ball, Player, Racquet

# if SHOW_GROUND:
#     ### 2a) Ground Truth

#     myCam = Camera.calibrated_AO_behind()
#     backScn = Scene.make_court()
#     frntScn = Scene.make_empty()
#     backView = View.make(myCam, backScn, resolution = BACKGROUND_RES)
    
#     for frame in seq_data_ground.index:
        
#         try:
#             x, y, z = ball_3D_ground_show.loc[frame]
#         except KeyError:
#             tBall = None
#         else:
#             tBall = Ball.make((x, y, z))
#         frontView = View.make(myCam, frntScn.put(tBall))
        
#         showFrm = backView.combine(frontView).snap()
#         show_continuous(showFrm.array, ms_between_frames = MS_BETWEEN_FRAMES)
        
#     cv.destroyAllWindows()

if SHOW_INFERENCE:
    ### 2b) Inference
    
    ### courtside, angled
    myCam = Camera.calibrated_AO_behind().left(20).into(14).pan(-0.06).down(4)
    
    # ### side view
    # myCam = Camera.calibrated_AO_behind().right(55).pan(0.5).rotate(0.005).right(133).zoom(0.175)
    # myCam = myCam.tilt(-0.05).pan(0.008)
    
    backScn = Scene.make_court()
    backView = View.make(myCam, backScn, resolution = BACKGROUND_RES)
    
    for frame in event_frames:
        
        ### Get Ball 3D Data
        try:
            x, y, z = ball_3D_infer_show.loc[frame]
        except KeyError:
            aBall = None
        else:
            aBall = Ball.make((x, y, z))
            
        try:
            x, y, z = ball_3D_infer_show.loc[frame - 1]
        except KeyError:
            aBall_before = None
        else:
            aBall_before = Ball.make((x, y, z))
        
        try:
            x, y, z = ball_3D_infer_show.loc[frame + 1]
        except KeyError:
            aBall_after = None
        else:
            aBall_after = Ball.make((x, y, z))
        
        ### Make Player
        x_top, y_top, z_top, x_bot, y_bot, z_bot = players_3D_show.loc[frame]
        topPlayer = Player.make((x_top, y_top, z_top), name = 'Roger Federer')
        botPlayer = Player.make((x_bot, y_bot, z_bot), name = 'Tomas Berdych') 
            
        ### Make Racket
        topRacquet = Racquet.make_at_player(topPlayer, distance_from_player = 10, 
                                            dir_hori = 0.5, dir_vert = 0.5, inclination = 0)
        botRacquet = Racquet.make_at_player(botPlayer, distance_from_player = 10, 
                                            dir_hori = 0.5, dir_vert = 0.5, inclination = 0)
        if event_features[frame] in ['HF', 'SFR', 'SFL']:
            topRacquet = Racquet.make_at_contact(topPlayer, aBall, 
                                                 aBall_before = aBall_before,
                                                 aBall_after = aBall_after)
        if event_features[frame] in ['HN', 'SNR', 'SNL']:
            botRacquet = Racquet.make_at_contact(botPlayer, aBall, 
                                                 aBall_before = aBall_before,
                                                 aBall_after = aBall_after)
                                                 
        
        frntScn = Scene.make_empty().put(aBall).put(topPlayer).put(botPlayer).put(topRacquet).put(botRacquet)
        frontView = View.make(myCam, frntScn, resolution = FOREGROUND_RES)
        
        showFrm = backView.combine(frontView).snap()
        show_continuous(showFrm.array, ms_between_frames = MS_BETWEEN_FRAMES)
        
    cv.destroyAllWindows()



