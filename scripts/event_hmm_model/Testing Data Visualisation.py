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
SHOW_SEQUENCE = 63
INTERPOLATION_TYPE = 'smart'
GROUND_OR_INFERENCE = 'inference'

### GET STREAM FEATURES ###
from ATMANpy.funcs.feature_funcs import load_stored_features
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)

### GET 3D FEATURES ###

from ATMANpy.funcs.feature_funcs import read_feature_frame

try:
    ball_3D = read_feature_frame(f'3D_Ball_{INTERPOLATION_TYPE}_{GROUND_OR_INFERENCE}', VIDEO_ID, STORE_NAME)
except FileNotFoundError:
    ball_3D = pd.DataFrame()

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

if GROUND_OR_INFERENCE == 'ground':
    event_features = seq_data_ground['Event'].to_dict()
else:
    event_features = {frame: featDict['grammatic_event'] for frame, featDict in test_sequence_features.items()}

event_features = narrow_down_instantaneous_events(event_features)
# seq_data_infer = merge_words_and_event_features(seq_data_hidden, event_features)
event_features = pd.Series(event_features)
event_features = event_features[event_features != 'None']
event_frames = event_features.index

ball_3D_show = ball_3D.reindex(event_frames)
players_3D_show = players_3D.reindex(event_frames)
    
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
trajectory_frames = [[el for el in widen_list(tList, by = 1) if el in event_features.index] for tList in trajectory_frames]
trajectory_frames = [ts for ts in trajectory_frames if traj_is_hit_or_serve(event_features[ts])]

print('Calculating Spins...')
data_frames = []
for tList in trajectory_frames:
    trajFrm = ball_3D_show.loc[tList]
    
    ### Calculate properties of measured flight trajectory: initial velocities, position.
    pos_ground_measured, pos_vert_measured = sf.calculate_planar_ball_flight(trajFrm)
    v_ground_idx, v_vert_idx, rpm_idx = sf.get_optimal_trajectory_indices(groundArr, vertArr, 
                                                                          pos_ground_measured, pos_vert_measured,
                                                                          sum_of_squares = False)
    
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
   
speed_3D = pd.concat(data_frames)
# there can be duplication in case we have a short hit event -> in this case, we take the more recent shot.
speed_3D = speed_3D[~speed_3D.index.duplicated(keep = 'last')] 
speed_3D = speed_3D.reindex(event_features.index).ffill()
speed_3D[['v_x', 'v_z', 'v']] = speed_3D[['v_x', 'v_z', 'v']] * const.M_TO_KM / const.SEC_TO_MIN / const.MIN_TO_HOUR

### TODO: Check if you can speed up the routine by limiting search to initial conditions
### that are "not too far" from the initial conditions as measured 
### (see code below to find a search starting point)
# v_init_ground_measured, v_init_vert_measured = sf.calculate_initial_conditions(trajFrm, frames_to_use = 1)
# ground_initial_index = np.abs(v_init_ground_space - v_init_ground_measured).argmin()
# vert_initial_index = np.abs(v_init_vert_space - v_init_vert_measured).argmin()

# pos_ground_theoretical = groundArr[ground_initial_index, vert_initial_index, :, :len(tList)]
# pos_vert_theoretical = vertArr[ground_initial_index, vert_initial_index, :, :len(tList)]


#%% 

def make_text_features(event_features, players_3D, ball_3D, speed_3D):
    from ATMANpy.types.Points import Point2D
    
    bounce_detected = False
    lock_stroke = False
    
    features_3D = {}
    
    event_features_not_none = event_features[event_features != 'None']
    for frame in event_features_not_none.index:
        
        event = event_features_not_none.loc[frame]
        
        ### BALL ###
        x, y, z = ball_3D.loc[frame]
        ballPt = Point2D((x, y))
        
        ### PLAYER ###
        top_x_3D, top_y_3D = players_3D.loc[frame, 'top_x_3D'], players_3D.loc[frame, 'top_y_3D']
        bot_x_3D, bot_y_3D = players_3D.loc[frame, 'bot_x_3D'], players_3D.loc[frame, 'bot_y_3D']
        
        topPt = Point2D((top_x_3D, top_y_3D))
        botPt = Point2D((bot_x_3D, bot_y_3D))
            
        ### SPEED ###
        curr_speed = speed_3D.loc[frame, 'v']
        if np.isnan(curr_speed):
            curr_speed = None
        else:
            curr_speed = f'{round(curr_speed)} km/h'
    
        ### SPIN ###
        curr_spin = speed_3D.loc[frame, 'spin']
        
        if np.isnan(curr_spin):
            curr_spin = None
        else:
            abs_spin = abs(curr_spin)
            if curr_spin < -1000:
                kind = 'Slice'
            elif curr_spin < 1000:
                kind = 'Flat'
            else:
                kind = 'Topspin'
            curr_spin = f'{kind}: {round(abs_spin)} RPM (???)'
            
        ### VOLLEY VERSUS NORMAL STROKE
        if event.startswith('B'):
            bounce_detected = True
        if event.startswith('H'):
            if not lock_stroke:
                if bounce_detected:
                    stroke_type = 'Groundstroke'
                else:
                    stroke_type = 'Volley'
                lock_stroke = True
                bounce_detected = False ### reset the value as soon as ball is hit
        if event.startswith('FLIGHT'):
            lock_stroke = False 
        
        ### STROKE TYPE ###
        if event.startswith('S'):
            stroke_side = 'Serve'
            stroke_type = ''
        elif event == 'HF':
            ### figure out if forehand or backhand. Assume two right handed players for the moment
            OVERHEAD_THRESHOLD = 2.25 * 3.281 * 8 ### 2.25 meter in feet times scale factor
            if z > OVERHEAD_THRESHOLD:
                stroke_side = 'Overhead'
            else:
                if ballPt.is_left(topPt):
                    stroke_side = 'Forehand'
                else:
                    stroke_side = 'Backhand'
        elif event == 'HN':
            ### figure out if forehand or backhand. Assume two right handed players for the moment
            OVERHEAD_THRESHOLD = 2.25 * 3.281 * 8 ### 2.25 meter in feet times scale factor
            if z > OVERHEAD_THRESHOLD:
                stroke_side = 'Overhead'
            else:
                if ballPt.is_right(botPt):
                    stroke_side = 'Forehand'
                else:
                    stroke_side = 'Backhand'
    
        stroke = stroke_side + ' ' + stroke_type
    
        ### MAKE FEATDICT ###
        featDict = {'stroke': stroke, 'ballSpeed': curr_speed, 
                    'ballSpin': curr_spin}
        
        features_3D[frame] = featDict
    
    return features_3D

txtDict = make_text_features(event_features, players_3D_show, ball_3D_show, speed_3D)
    


#%% #### VISUALIZE #########

from ATMANpy.funcs.util_funcs import get_folder_struct
from ATMANpy.types.Frames import BGRFrame

from ATMANpy.types.Visual3D import View, Camera, Scene
from ATMANpy.types.ShapesTennis3D import Ball, Player, Racquet, Mark

from ATMANpy.funcs.draw_funcs import draw, split_and_check

MS_BETWEEN_FRAMES = 30
SHOW_VIDEO = False
SAVE_VIDEO = False
SHOW_ANIMATION = True
SAVE_ANIMATION = True

BACKGROUND_RES = 1
FOREGROUND_RES = 2

PERSPECTIVE = 'tv' ### 'tv', 'behind', 'lateral'

### VIDEO SAVING ###
SIZE = (int(cap.get(3)), int(cap.get(4)))
SAVEPATH = get_folder_struct('video_files')['main']
CODEC = 'H265'
SUFFDICT = {'H265': 'mpeg',
            'MJPG': 'avi'}
suffix = SUFFDICT[CODEC]
###


### 1) Real Video
if SHOW_VIDEO:
    frame_min, frame_max = min(seq_data_ground.index), max(seq_data_ground.index)
    show_video(cap, 
               start_frame = frame_min, stop_frame = frame_max,
               ms_between_frames = MS_BETWEEN_FRAMES)
          
    if SAVE_VIDEO:
        ### Save Video
        videoWriter = cv.VideoWriter(os.path.join(SAVEPATH, f'real_{SHOW_SEQUENCE}.{suffix}'), 
                                     cv.VideoWriter_fourcc(*CODEC),
                                     25, SIZE)
            
        for frame in range(frame_min, frame_max):
            videoWriter.write(BGRFrame.from_video(cap, frame).array) 

    
# ### 2) Generated Views
ballMarks = []

if SAVE_ANIMATION:
    ### Save Video
    videoWriter = cv.VideoWriter(os.path.join(SAVEPATH, f'{GROUND_OR_INFERENCE}_{PERSPECTIVE}_{SHOW_SEQUENCE}.{suffix}'), 
                                 cv.VideoWriter_fourcc(*CODEC),
                                 25, SIZE)
        
if SHOW_ANIMATION:
    
    myCam = Camera.calibrated_AO_behind()
    
    if PERSPECTIVE == 'behind':
        ### low, angled
        myCam = myCam.left(20).into(14).pan(-0.06).down(4)
    
    if PERSPECTIVE == 'lateral':
        # # ### side view
        myCam = myCam.right(55).pan(0.5).rotate(0.005).right(133).zoom(0.175)
        myCam = myCam.tilt(-0.05).pan(0.008)
    
    backScn = Scene.make_court()
    backView = View.make(myCam, backScn, resolution = BACKGROUND_RES)

    ### get the text region    
    backFrm = backView.snap()
    boolArr = (backFrm.array == 0).all(axis = 2)
    textRegion = split_and_check(boolArr, left_prop = 2, overall_prop = 3)

    for frame in event_frames:
        
        ### Get prev event
        try:
            prev_event = event_features[frame - 1]
        except KeyError:
            prev_event = None
        
        ### Get next event
        try:
            next_event = event_features[frame + 1]
        except KeyError:
            next_event = None
        
        ### Get Ball 3D Data
        try:
            x, y, z = ball_3D_show.loc[frame]
        except KeyError:
            aBall = None
        else:
            aBall = Ball.make((x, y, z))
            
        try:
            x, y, z = ball_3D_show.loc[frame - 1]
        except KeyError:
            aBall_before = None
        else:
            aBall_before = Ball.make((x, y, z))
        
        try:
            x, y, z = ball_3D_show.loc[frame + 1]
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
                                            dir_hori = -0.5, dir_vert = 0.5, inclination = 0)
        if event_features[frame] in ['HF', 'SFR', 'SFL']:
            if not next_event in ['HF', 'SFR', 'SFL']:
                topRacquet = Racquet.make_at_contact(topPlayer, aBall, 
                                                     aBall_before = aBall_before,
                                                     aBall_after = aBall_after)
        if event_features[frame] in ['HN', 'SNR', 'SNL']:
            if not next_event in ['HN', 'SNR', 'SNL']:
                botRacquet = Racquet.make_at_contact(botPlayer, aBall, 
                                                     aBall_before = aBall_before,
                                                     aBall_after = aBall_after)
        ### Make Ball Mark
        if event_features[frame].startswith('B'):
            if prev_event is not None:
                if not prev_event.startswith('B'):
                    
                    if next_event is None:
                        ballPt = aBall.get_point()
                    elif next_event.startswith('B'):
                        ballPt = (aBall.get_point() + aBall_after.get_point()) / 2
                    else:
                        ballPt = aBall.get_point()
                    
                    if event_features[frame][1] == 'I': 
                        ballMark = Mark.make(ballPt, color = (0, 252, 124))
                    else:
                        ballMark = Mark.make(ballPt, color = (124, 0, 252))    
                    
                    ballMarks.append(ballMark)
                    
        frntScn = Scene.make_empty().put(aBall).put(topPlayer).put(botPlayer)
        frntScn = frntScn.put(topRacquet).put(botRacquet)
        frntScn = frntScn.put(ballMarks)
        frontView = View.make(myCam, frntScn, resolution = FOREGROUND_RES)
        
        showFrm = backView.combine(frontView).snap()
        
        if GROUND_OR_INFERENCE == 'ground':
            add_text_string = 'Event Detection: Ground Truth'
        else:
            add_text_string = 'Event Detection: Inference'
            
        drawFunc = draw(['stroke', 'ballSpeed', 'ballSpin'], 
                        textRegion = None, add_text = add_text_string)
        showFrm = drawFunc({'cFrm': showFrm}, txtDict[frame])
        
        show_continuous(showFrm.array, ms_between_frames = MS_BETWEEN_FRAMES)
        
        if SAVE_ANIMATION:
            videoWriter.write(showFrm.array) 
  

    if SAVE_ANIMATION:
        ### make last picture more 'permanent'
        frntScn = Scene.make_empty().put(topPlayer).put(botPlayer)
        frntScn = frntScn.put(topRacquet).put(botRacquet)
        frntScn = frntScn.put(ballMarks)
        frontView = View.make(myCam, frntScn, resolution = FOREGROUND_RES)
        
        showFrm = backView.combine(frontView).snap()
        
        if GROUND_OR_INFERENCE == 'ground':
            add_text_string = 'Event Detection: Ground Truth'
        else:
            add_text_string = 'Event Detection: Inference'
            
        drawFunc = draw(['stroke', 'ballSpeed', 'ballSpin'], 
                        textRegion = None, add_text = add_text_string)
        showFrm = drawFunc({'cFrm': showFrm}, txtDict[frame])
    
        videoWriter.write(showFrm.array) 
        videoWriter.write(showFrm.array) 
        videoWriter.write(showFrm.array) 
        videoWriter.write(showFrm.array) 
        videoWriter.write(showFrm.array) 
        videoWriter.write(showFrm.array) 
        videoWriter.write(showFrm.array) 
  
    cv.destroyAllWindows()

