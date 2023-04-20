# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:27:28 2022

@author: gianl
"""


import os
import pandas as pd
import numpy as np

from ATMANpy.funcs.HMM.utils import read_hmm_params, read_train_test

from ATMANpy.funcs.util_funcs import get_video        
from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous

from ATMANpy.funcs.draw_funcs import draw

from ATMANpy.funcs.cv_funcs import replay


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
hmmDict = read_hmm_params(MODEL_NAME)

from ATMANpy.funcs.HMM.TokenPass import read_grammar, infer, ModelShell
grammarStr = 'full_grammar_test'
grammar = read_grammar(grammarStr)

from ATMANpy.funcs.HMM.HVite import token_pass_sequence

#%% VISUAL INSPECTION ###


### GET STREAM FEATURES ###
from ATMANpy.funcs.feature_funcs import load_stored_features
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)

### ENRICH STREAM FEATURES WITH INFERENCE ###

def expand_inferred_sequence(inferred_sequence, word_beginning_frame = 0):
    
    ls_word = inferred_sequence[-1]
    last_frame = ls_word[0]
    
    curr_word_tuple, other_word_tuples = inferred_sequence[0], inferred_sequence[1:]
    stop_frame, word = curr_word_tuple

    outDict = {}
    for frame in range(word_beginning_frame, last_frame + 1):
        if frame <= stop_frame:
            outDict[frame] = word
        else:
            if not other_word_tuples == []:
                curr_word_tuple, other_word_tuples = other_word_tuples[0], other_word_tuples[1:]
                stop_frame, word = curr_word_tuple
                outDict[frame] = word
    
    return outDict

infer_arbitrary_dict = {}
infer_grammatic_dict = {}
for test_sequence in [test_data.IndexSF.unique()[-1]]:
    print(f'Doing Test Sequence {test_sequence}...')
    
    randSeq = test_data[test_data.IndexSF == test_sequence]
    words = randSeq.drop(columns = ['Event', 'IndexSF'])

    print('    Doing arbitrary inference...')
    arbitrary_sequence = token_pass_sequence(hmmDict, words)
    print('    Doing grammatical inference...')
    grammatic_sequence = infer(words, hmmDict, grammar)    

    infer_arbitrary_dict[test_sequence] = arbitrary_sequence
    infer_grammatic_dict[test_sequence] = grammatic_sequence
    
    frame_min, frame_max = words.index.min(), words.index.max()
    
    arbitrary_expanded = expand_inferred_sequence(arbitrary_sequence, 
                                                  word_beginning_frame = frame_min)
    grammatic_expanded = expand_inferred_sequence(grammatic_sequence, 
                                                  word_beginning_frame = frame_min)

    for frame, inferred_event in arbitrary_expanded.items():
        stream_features[frame]['arbitrary_event'] = inferred_event
        
    for frame, inferred_event in grammatic_expanded.items():
        stream_features[frame]['grammatic_event'] = inferred_event

    # replay(cap, stream_features, 
    #         start_frame = frame_min, stop_frame = frame_max,
    #         ms_between_frames = 100,
    #         drawFunc = draw(['playerTop', 'playerBot', 'ball', 'courtLines', 
    #                         'arbitrary_event', 'grammatic_event']))
    

#%% Testing 


# ### Issues:
#     # terminal events are duplicated. Why? Usually it's HF, BIN, FLIGHT_DOWN
#     # how to evaluate performance?
#     # do we truncate short words (bounce, hit, cord) to only be max. three frames long?
#     # how do we visualize?
#         # coolest: reconstruct a three dimensional model of the point (with players and ball)
#         # cool: reconstruct two dimensional model of point (with players and ball)
#     # how do we train and test on more?
#     # how do we categorize the problems in recognition (especially seeing a point end)
#         # use sound (for let/cord)
#         # avoid interpolation between two balls that are not the same ball 
#             # (when scene does not end between two serves, for example)
#             # (alternatively, we could check the arbitrary sequence for a second serve...)
#             # in the best case, a second ball would have been detected prior, and removed.
#         # i bet most miscategorizations are only due to the sequence being too long or too short.
#         # the others are likely bounce in or out
#             # -> can we improve bounce in or out with knowing the homography, and where the lines are?
        
# for test_sequence in test_data.IndexSF.unique():
    
#     randSeq = test_data[test_data.IndexSF == test_sequence]
#     words = randSeq.drop(columns = ['Event', 'IndexSF'])
#     frame_min, frame_max = words.index.min(), words.index.max()
    
#     replay(cap, stream_features, 
#             start_frame = frame_min, stop_frame = frame_max,
#             ms_between_frames = 100,
#             drawFunc = draw(['playerTop', 'playerBot', 'ball', 'courtLines', 
#                             'arbitrary_event', 'grammatic_event']))

#%% Visualization 

import cv2 as cv

######## GET SCHEMA ########
from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

### CHOOSE A SEQUENCE ###
test_sequence = 108
testSeq = test_data[test_data.IndexSF == test_sequence]
words = testSeq.drop(columns = ['Event', 'IndexSF'])
frames = list(words.index)

inferred_events = infer_grammatic_dict[test_sequence]

from ATMANpy.funcs.court_funcs import calibrate_camera, make_P_cond_inv
from ATMANpy.types.Frames import GreyFrame

### CALIBRATE THE CAMERA ON EACH FRAME ###
projDict = {}

for frame in frames:
    H = stream_features[frame]['H']
    gFrm = GreyFrame.from_video(cap, frame)
    
    P = calibrate_camera(gFrm, schemaLines, H)
    projDict[frame] = P

### FIND PLAYER POSITIONS ###
players_3D = pd.DataFrame(index = frames)

for frame in frames:
    P = projDict[frame]
    
    topPlayer = stream_features[frame]['playerTop']
    botPlayer = stream_features[frame]['playerBot']
    
    top_x, top_y = topPlayer.enclosing_region()['Bottom'].center().coords
    bot_x, bot_y = botPlayer.enclosing_region()['Bottom'].center().coords
    
    cond_val = 0 ### feet are on the ground!
    cond_dim = 2 ### axis is z dimension -> height!
    P_cond_inv = make_P_cond_inv(P, cond_val, cond_dim)
    
    top_3D_homogeneous = P_cond_inv @ np.array([top_x, top_y, 1])
    top_3D = top_3D_homogeneous[:2] / top_3D_homogeneous[2]
    
    bot_3D_homogeneous = P_cond_inv @ np.array([bot_x, bot_y, 1])
    bot_3D = bot_3D_homogeneous[:2] / bot_3D_homogeneous[2]
    
    top_x_3D, top_y_3D, top_z_3D = np.insert(top_3D, cond_dim, cond_val)
    bot_x_3D, bot_y_3D, bot_z_3D = np.insert(bot_3D, cond_dim, cond_val)
    
    players_3D.at[frame, 'top_x_3D'] = top_x_3D
    players_3D.at[frame, 'top_y_3D'] = top_y_3D
    players_3D.at[frame, 'top_z_3D'] = top_z_3D
    
    players_3D.at[frame, 'bot_x_3D'] = bot_x_3D
    players_3D.at[frame, 'bot_y_3D'] = bot_y_3D
    players_3D.at[frame, 'bot_z_3D'] = bot_z_3D
    

### INITIALIZE BALL POSITIONS ### 

def count_consecutive(evt, evtList):
    i = 0
    for other in evtList:
        if evt == other:
            i += 1
        else:
            return i, other
    raise Exception
    

def narrow_down_instantaneous_events(expanded_events):
    
    min_frame = min(expanded_events)
    max_frame = max(expanded_events)
    MAX_REACH = 15

    narrow_event_function = lambda x: x.startswith('B') or x.startswith('H')
    
    narrow_expanded_events = {}
    for frame in expanded_events:
        curr_event = expanded_events[frame]
        
        ### figure out left and right look
        comparison_reach = min(frame - min_frame, min(max_frame - frame, MAX_REACH))
        
        isNarrow = narrow_event_function(curr_event)
        
        if not isNarrow:
            narrow_expanded_events[frame] = curr_event
        else:
            events_before = [expanded_events[frame - comp] for comp in range(1, comparison_reach + 1)]
            events_after  = [expanded_events[frame + comp] for comp in range(1, comparison_reach + 1)]
            
            count_bef, prev_event = count_consecutive(curr_event, events_before)
            count_aft, succ_event = count_consecutive(curr_event, events_after)
            
            if count_bef + 1 < count_aft: ### there are substantially more of our narrow event after than before
                narrow_expanded_events[frame] = prev_event ### -> insert the previous event!
            elif count_bef - 1 > count_aft:
                narrow_expanded_events[frame] = succ_event 
            else:
                narrow_expanded_events[frame] = curr_event 
                
    return narrow_expanded_events
        

def merge_words_and_inference(words, inference):
    fs_frm = min(words.index)
    
    expanded_events = expand_inferred_sequence(inferred_events, word_beginning_frame = fs_frm)
    narrow_events = narrow_down_instantaneous_events(expanded_events)
    event_series = pd.Series(narrow_events, name = 'Events')
    
    inf_words = pd.concat([words, event_series], axis = 1)
    
    return inf_words


def init_spatial_dimensions(inf_words):            

    x_true = pd.Series(np.nan, index = inf_words.index, name = 'x_3D')
    y_true = pd.Series(np.nan, index = inf_words.index, name = 'y_3D')
    z_true = pd.Series(np.nan, index = inf_words.index, name = 'z_3D')
    
    inf_words = pd.concat([inf_words, x_true, y_true, z_true], axis = 1)
    
    return inf_words
    
def impose_serve(inf_words, schemaLines):
           
    ### Use the known latitude for serve -> always at y coord of one of the baselines!
    serve_near_cond = inf_words.Events.str.startswith('SN')
    serve_far_cond = inf_words.Events.str.startswith('SF')
    serve_near_latitude = schemaLines['Bottom Baseline'].center().y
    serve_far_latitude = schemaLines['Top Baseline'].center().y
    
    preset_values = inf_words['y_3D']
    imposed_values = np.where(serve_near_cond, serve_near_latitude, 
                              np.where(serve_far_cond, serve_far_latitude, preset_values))          
    inf_words['y_3D'] = imposed_values
    
    return inf_words
       
def impose_net(inf_words, schemaLines):

    ### Use the known latitude for NET -> always at y coord of net!
    net_cond = inf_words.Events.str.startswith('NET')
    net_latitude = schemaLines['Net Line'].center().y
    
    preset_values = inf_words['y_3D']
    imposed_values = np.where(net_cond, net_latitude, preset_values)               
    inf_words['y_3D'] = imposed_values          
    
    return inf_words

def impose_bounce(inf_words, bounce_height = 0.2):
    
    ### Use the height for bounce -> ball is low!
    bounce_cond = inf_words.Events.str.startswith('B')
    
    preset_values = inf_words['z_3D']
    imposed_values = np.where(bounce_cond, bounce_height, preset_values)         
    inf_words['z_3D'] = imposed_values        
    
    return inf_words

def impose_hit(inf_words, player_positions):
    
    ### Use the fact that the ball is hit slightly in front of player!
    how_far_in_front = 8 ### this corresponds to one foot
    
    hit_far_cond = inf_words.Events == 'HF'
    hit_near_cond = inf_words.Events == 'HN'
    
    hypothetical_ball_top = player_positions.top_y_3D + how_far_in_front
    hypothetical_ball_bot = player_positions.bot_y_3D - how_far_in_front

    preset_values = inf_words['y_3D']
    imposed_values = np.where(hit_far_cond, hypothetical_ball_top, 
                              np.where(hit_near_cond, hypothetical_ball_bot, preset_values))
    inf_words['y_3D'] = imposed_values          
    
    return inf_words


inf_words = merge_words_and_inference(words, inferred_events)
inf_words = init_spatial_dimensions(inf_words)
inf_words = impose_serve(inf_words, schemaLines)
inf_words = impose_net(inf_words, schemaLines)
inf_words = impose_bounce(inf_words, bounce_height = 0.2)
inf_words = impose_hit(inf_words, players_3D)

### FIND BALL POSITIONS ###
# -> where we have imposed one dimension, we can uniquely identify the other two.

for frame in frames:
    row = inf_words.loc[frame]
    
    x, y, z = row.x_3D, row.y_3D, row.z_3D
    if not np.isnan(x):
        cond_val = x
        cond_dim = 0
    elif not np.isnan(y):
        cond_val = y
        cond_dim = 1
    elif not np.isnan(z):
        cond_val = z
        cond_dim = 2
    else:
        continue
        
    P = projDict[frame]
    P_cond_inv = make_P_cond_inv(P, cond_val, cond_dim)

    pt_3D_homogeneous = P_cond_inv @ np.array([row.x, row.y, 1])
    pt_3D = pt_3D_homogeneous[:2] / pt_3D_homogeneous[2]

    x_3D, y_3D, z_3D = np.insert(pt_3D, cond_dim, cond_val)

    inf_words.at[frame, 'x_3D'] = x_3D
    inf_words.at[frame, 'y_3D'] = y_3D
    inf_words.at[frame, 'z_3D'] = z_3D
    
### INTERPOLATE BALL POSITION ###
# where we have not imposed one dimension, we interpolate between the events where we did find one.

ball_3D = inf_words[['x_3D', 'y_3D', 'z_3D']].interpolate(axis = 0, limit_area = 'inside').dropna()
ball_3D = ball_3D.round().astype(int)

dist_euclid = ((ball_3D - ball_3D.shift(-1)) ** 2).sum(axis = 1) ** 0.5 


def calc_speed(dist_euclid, frames_per_second = 25, pixel_per_foot = 8):
    ### dist_euclid: distance in pixel across one frame
    foot_per_second = dist_euclid * frames_per_second / pixel_per_foot
    meter_per_second = foot_per_second * 0.3048
    kilometer_per_hour = meter_per_second * 3600 / 1000
    return kilometer_per_hour

ball_speed = calc_speed(dist_euclid)

### MAKE 3D STREAM FEATURES

height_to_size = lambda height: max(2, min(round(height / 25), 4))

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Region

curr_speed = '0 km/h'
bounce_detected = False
lock_stroke = False

features_3D = {}
for frame in ball_3D.index:
    
    event = inf_words.loc[frame, 'Events']
    try:
        next_event = inf_words.loc[frame + 1, 'Events'] ### for speeds
    except:
        next_event = None
    
    ### BALL ###
    x, y, z = ball_3D.loc[frame]
    ballPt = Point2D((x, y))
    ballSize = height_to_size(z)
    
    ### PLAYER ###
    top_x_3D, top_y_3D = players_3D.loc[frame, 'top_x_3D'], players_3D.loc[frame, 'top_y_3D']
    bot_x_3D, bot_y_3D = players_3D.loc[frame, 'bot_x_3D'], players_3D.loc[frame, 'bot_y_3D']
    
    topPt = Point2D((top_x_3D, top_y_3D))
    botPt = Point2D((bot_x_3D, bot_y_3D))
    
    topReg = Region.from_center(topPt, radius = 5)
    botReg = Region.from_center(botPt, radius = 5)
    
    ### BOUNCE ###
    if event.startswith('BI'):
        bounce_loc = ballPt
        bounce_col = (0, 255, 0)
    elif event.startswith('BO') or event == 'NET':
        bounce_loc = ballPt
        bounce_col = (0, 0, 255)
    else:
        bounce_col = (0, 255, 255)
    
    ### SPEEDS ###
    if event.startswith('S') or event.startswith('H'):
        if not (next_event.startswith('S') or next_event.startswith('H')):
            curr_speed = ball_speed.loc[frame]
            curr_speed = f'{round(curr_speed)} km/h'
    
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
    featDict = {'ball': ballPt, 'ballSize': ballSize, 'ballColor': bounce_col,
                'stroke': stroke, 'ballSpeed': curr_speed, 
                'playerTop': topReg, 'playerBot': botReg}
    
    features_3D[frame] = featDict
    
strokes = [featDict['stroke'] for featDict in features_3D.values()]
    
    
### show!

show_video(cap, 
        start_frame = frame_min, stop_frame = frame_max,
        ms_between_frames = 50)

bounce_registered = False ### this is needed to transform two-bounce frames into a single bounce location
good_bounces = []
bad_bounces = []
for frame in ball_3D.index:
    
    backgroundFrm = schemaFrm.to_BGR()
    frmDict = {'cFrm': backgroundFrm}
    featDict = features_3D[frame]
    
    drawFunc = draw(['ball', 'playerTop', 'playerBot', 
                     'ballSpeed', 'stroke']) ### takes frmDict and featDict, returns show frame
    
    showFrm = drawFunc(frmDict, featDict)
    
    ### add permanent ball marks at bounce points
    if featDict['ballColor']  == (0, 255, 0): ### green (in)
        if bounce_registered:    
            good_bounces[-1] = (featDict['ball'] + good_bounces[-1]) // 2
        else:    
            good_bounces.append(featDict['ball'])
        bounce_registered = True
    elif featDict['ballColor'] == (0, 0, 255):
        if bounce_registered:    
            bad_bounces[-1] = (featDict['ball'] + bad_bounces[-1]) // 2
        else:    
            bad_bounces.append(featDict['ball'])
        bounce_registered = True
    else:
        bounce_registered = False

    showFrm = showFrm.put(good_bounces, thickness = 1, color = (0, 255, 0), radius = 3)
    showFrm = showFrm.put(bad_bounces, thickness = 1, color = (0, 0, 255), radius = 3)
    
    show_continuous(showFrm.array, ms_between_frames = 50)
    
    
cv.destroyAllWindows()
    











