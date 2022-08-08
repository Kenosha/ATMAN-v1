# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 03:35:09 2022

@author: gianl
"""

import numpy as np
import pandas as pd

from ATMANpy.funcs.court_funcs import make_P_cond_inv

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

def make_projection_matrices(cap, stream_features, schemaLines):
    from ATMANpy.funcs.court_funcs import calibrate_camera
    from ATMANpy.types.Frames import GreyFrame
    
    frames = list(stream_features.keys())
    ### CALIBRATE THE CAMERA ON EACH FRAME ###
    projDict = {}
    
    for frame in frames:
        H = stream_features[frame]['H']
        gFrm = GreyFrame.from_video(cap, frame)
        
        P = calibrate_camera(gFrm, schemaLines, H)
        projDict[frame] = P
    
    return projDict
        
def find_player_positions_3D(projDict, stream_features):
    frames = list(projDict.keys())
    
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
    
    return players_3D

def find_ball_positions_3D(word_inference_frame, projDict):
    
    ball_3D = word_inference_frame[['x_3D', 'y_3D', 'z_3D']]
    frames = word_inference_frame.index
    
    for frame in frames:
        row = word_inference_frame.loc[frame]
        
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
    
        ball_3D.at[frame, 'x_3D'] = x_3D
        ball_3D.at[frame, 'y_3D'] = y_3D
        ball_3D.at[frame, 'z_3D'] = z_3D
        
    ### INTERPOLATE BALL POSITION ###
    # where we have not imposed one dimension, we interpolate between the events where we did find one.
    
    ball_3D = ball_3D.interpolate(axis = 0, limit_area = 'inside').dropna()
    ball_3D = ball_3D.round().astype(int)
    
    return ball_3D

def calc_ball_dist(ball_3D):
    dist_euclid = ((ball_3D - ball_3D.shift(-1)) ** 2).sum(axis = 1) ** 0.5 
    return dist_euclid

def calc_ball_speed(dist_euclid, frames_per_second = 25, pixel_per_foot = 8):
    ### dist_euclid: distance in pixel across one frame
    foot_per_second = dist_euclid * frames_per_second / pixel_per_foot
    meter_per_second = foot_per_second * 0.3048
    kilometer_per_hour = meter_per_second * 3600 / 1000
    return kilometer_per_hour


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
        

def merge_words_and_inference(words, inferred_events):
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

def make_3D_features(word_inference_frame, ball_3D, players_3D):
    from ATMANpy.types.Points import Point
    from ATMANpy.types.Lines import Region
    
    height_to_size = lambda height: max(2, min(round(height / 25), 4))
    ball_speed = calc_ball_speed(calc_ball_dist(ball_3D))

    curr_speed = '0 km/h'
    bounce_detected = False
    bounce_detected_in_previous_frame = False
    lock_stroke = False
    
    good_bounces = []
    bad_bounces = []
    
    features_3D = {}
    for frame in ball_3D.index:
        
        event = word_inference_frame.loc[frame, 'Events']
        try:
            next_event = word_inference_frame.loc[frame + 1, 'Events'] ### for speeds
        except:
            next_event = None
        
        ### BALL ###
        x, y, z = ball_3D.loc[frame]
        ballPt = Point((x, y))
        ballSize = height_to_size(z)
        
        ### PLAYER ###
        top_x_3D, top_y_3D = players_3D.loc[frame, 'top_x_3D'], players_3D.loc[frame, 'top_y_3D']
        bot_x_3D, bot_y_3D = players_3D.loc[frame, 'bot_x_3D'], players_3D.loc[frame, 'bot_y_3D']
        
        topPt = Point((top_x_3D, top_y_3D))
        botPt = Point((bot_x_3D, bot_y_3D))
        
        topReg = Region.from_center(topPt, radius = 5)
        botReg = Region.from_center(botPt, radius = 5)
        
        ### BOUNCE ###
        if event.startswith('BI'):
            bounce_loc = ballPt
            bounce_col = (0, 255, 0)
            
            ### Permanent Ball Mark
            if bounce_detected_in_previous_frame:    
                good_bounces[-1] = (bounce_loc + good_bounces[-1]) // 2
            else:    
                good_bounces.append(bounce_loc)
            bounce_detected_in_previous_frame = True
            
        elif event.startswith('BO') or event == 'NET':
            bounce_loc = ballPt
            bounce_col = (0, 0, 255)
            
            if bounce_detected_in_previous_frame:    
                bad_bounces[-1] = (bounce_loc + bad_bounces[-1]) // 2
            else:    
                bad_bounces.append(bounce_loc)
            bounce_detected_in_previous_frame = True
            
        else:
            bounce_col = (0, 255, 255)
            bounce_detected_in_previous_frame = False

        
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
                    'goodBounces': good_bounces.copy(), 'badBounces': bad_bounces.copy(),
                    'playerTop': topReg, 'playerBot': botReg}
        
        features_3D[frame] = featDict
    
    return features_3D
