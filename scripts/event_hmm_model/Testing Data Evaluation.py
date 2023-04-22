# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 03:33:20 2022

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

#%% MAKE INFERENCE ON TEST CASES

from ATMANpy.funcs.feature_funcs import save_events

SAVE_TRUE_EVENTS = True ### save the labelled events as features
if SAVE_TRUE_EVENTS:
    true_features = {frame: {'true_event': event} for frame, event in test_data.Event.to_dict().items()}
    save_events(true_features, 'true_event', VIDEO_ID, STORE_NAME)

### GET STREAM FEATURES ###
from ATMANpy.funcs.feature_funcs import load_stored_features
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)

### FIGURE OUT WHAT WAS INFERRED ALREADY, AND REMOVE FROM TO_DO
inferred_features = load_stored_features(VIDEO_ID, STORE_NAME, 
                                         featnameList = ['arbitrary_event', 'grammatic_event'])

test_data_to_do = test_data[~test_data.index.isin(inferred_features)]
sequences_to_do = test_data_to_do.IndexSF.unique()

### ENRICH STREAM FEATURES WITH INFERENCE ###
from ATMANpy.funcs.HMM.HPost import expand_inferred_sequence

infer_arbitrary_dict = {}
infer_grammatic_dict = {}
for test_sequence in sequences_to_do:
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
        # if frame in arbitrary_sequence:
        #     stream_features[frame]['arbitrary_event_unexpanded'] = arbitrary_sequence[frame]
        
    for frame, inferred_event in grammatic_expanded.items():
        stream_features[frame]['grammatic_event'] = inferred_event
        # if frame in arbitrary_sequence:
        #     stream_features[frame]['grammatic_event_unexpanded'] = grammatic_sequence[frame]
    

if len(sequences_to_do) > 0:
    ### WRITE STREAM FEATURES    
    inferred_features = {frame: featDict for frame, featDict in stream_features.items() if 'grammatic_event' in featDict}
    save_events(inferred_features, 'grammatic_event', VIDEO_ID, STORE_NAME)
    
    inferred_features = {frame: featDict for frame, featDict in stream_features.items() if 'arbitrary_event' in featDict}
    save_events(inferred_features, 'arbitrary_event', VIDEO_ID, STORE_NAME)
    
#%% READ EXISTENT 3D FEATURES

INTERPOLATION_TYPE = 'smart' # linear or smart
EVENT_SOURCE = 'ground' # ground or inference

from ATMANpy.funcs.feature_funcs import read_feature_frame

try:
    ball_3D = read_feature_frame(f'3D_Ball_{INTERPOLATION_TYPE}_{EVENT_SOURCE}', VIDEO_ID, STORE_NAME)
except FileNotFoundError:
    ball_3D = pd.DataFrame()

try:
    top_player_3D = read_feature_frame('3D_topPlayer', VIDEO_ID, STORE_NAME)
    bot_player_3D = read_feature_frame('3D_botPlayer', VIDEO_ID, STORE_NAME)
except FileNotFoundError:
    top_player_3D = pd.DataFrame()
    bot_player_3D = pd.DataFrame()

#%% CREATE 3D FEATURES

from ATMANpy.funcs.feature_funcs import save_feature_frame
import ATMANpy.funcs.HMM.HPost as HPost

######## GET SCHEMA ########
from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)
    
### CHOOSE A SEQUENCE ###
for test_sequence in test_data.IndexSF.unique():
#########################
    
    ## PREPARE 3D FEATURES ##
    seq_data = test_data[test_data.IndexSF == test_sequence] ### Event, here is what was actually marked!
    test_sequence_features = {frame: featDict for frame, featDict in stream_features.items() if frame in seq_data.index}
    
    if EVENT_SOURCE == 'inference':
        seq_data_hidden = seq_data.drop(columns = ['Event', 'IndexSF']) ### Drop it, and merge in inference 
        event_features = {frame: featDict['grammatic_event'] for frame, featDict in test_sequence_features.items()}
        seq_data = HPost.merge_words_and_event_features(seq_data_hidden, event_features)

    ## UNDERSTAND IF THE FEATURES HAVE ALREADY BEEN CREATED
    players_done = seq_data.index.isin(top_player_3D.index).all()
    ball_done = seq_data.index.isin(ball_3D.index).any() ### used to be .all(), but that is slightly too strict, as there is no interpolation if flight at the very end.
    
    if players_done and ball_done:
        print(f'Found in stored 3D features: {test_sequence}...')
        continue
    else:
        print(f'Post Processing Sequence Number: {test_sequence}...')
    
    ### POST PROCESS FEATURES
    projDict = HPost.make_projection_matrices(cap, test_sequence_features, schemaLines)
    
    if not players_done:
        ## 1) Players -> independent of whether we use ground truth events or inferred events!
        top_player_3D, bot_player_3D = HPost.find_player_positions_3D(projDict, test_sequence_features)
        save_feature_frame(top_player_3D, '3D_topPlayer', VIDEO_ID, STORE_NAME)
        save_feature_frame(bot_player_3D, '3D_botPlayer', VIDEO_ID, STORE_NAME)
    
    if not ball_done:
        ## 2) Ball
        process_data = HPost.init_spatial_dimensions(seq_data)
        process_data = HPost.impose_serve(process_data, schemaLines)
        process_data = HPost.impose_net(process_data, schemaLines)
        process_data = HPost.impose_bounce(process_data, bounce_height = 0.2)
        process_data = HPost.impose_hit(process_data, top_player_3D, bot_player_3D)
        ball_2D = process_data[['x', 'y']]
        
        if INTERPOLATION_TYPE == 'linear':
            ball_3D = HPost.find_known_ball_positions_3D(process_data, projDict)
            ball_3D = HPost.interpolate_ball_positions_3D(ball_3D)
            
        elif INTERPOLATION_TYPE == 'smart':
            ### Smarter Interpolation of Ball in Flight
            # Idea: Ball is on a ray for each frame between points where the ball location is known
            # Additionally, we know that the ball, projected onto the ground, has to be on the line between
            # the two known ball locations --> no sidespin!
            # With this, we should be able to identify the ball height.
            ball_3D = HPost.find_known_ball_positions_3D(process_data, projDict)
            
            ### Get Camera Internals and Externals
            internalDict, externalDict = HPost.make_internal_external_matrices(projDict)
            internalDict = HPost.interpolate_matrices(internalDict)
            externalDict = HPost.interpolate_matrices(externalDict)
            
            ### Do interpolation
            ball_3D = HPost.interpolate_ball_positions_3D_smart(ball_3D, ball_2D, internalDict, externalDict)
            
        save_feature_frame(ball_3D, f'3D_Ball_{INTERPOLATION_TYPE}_{EVENT_SOURCE}', VIDEO_ID, STORE_NAME)
    

#%% #### VISUALIZE GROUND v. INFERENCE #########

### RE-READ EVERYTHING WITH GOOD NAMES, BOTH GROUND AND INFERENCE

INTERPOLATION_TYPE = 'linear'

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

# test_data.IndexSF.unique() 
SHOW_SEQUENCE = 63

### GET TEST SEQUENCE STREAM FEATURES, EVENTS

seq_data_ground = test_data[test_data.IndexSF == SHOW_SEQUENCE] ### Event, here is what was actually marked!
seq_data_hidden = seq_data_ground.drop(columns = ['Event', 'IndexSF']) ### Drop it, and merge in inference 

test_sequence_features = {frame: featDict for frame, featDict in stream_features.items() if frame in seq_data_ground.index}
event_features = {frame: featDict['grammatic_event'] for frame, featDict in test_sequence_features.items()}
seq_data_infer = HPost.merge_words_and_event_features(seq_data_hidden, event_features)

ball_3D_infer_show = ball_3D_infer[ball_3D_infer.index.isin(seq_data_infer.index)]
ball_3D_ground_show = ball_3D_ground[ball_3D_ground.index.isin(seq_data_ground.index)]
players_3D_show = players_3D[players_3D.index.isin(seq_data_infer.index)]

features_3D_ground = HPost.make_3D_features(seq_data_ground.reindex(ball_3D_ground_show.index), ball_3D_ground_show, players_3D_show)
features_3D_infer = HPost.make_3D_features(seq_data_infer.reindex(ball_3D_infer_show.index), ball_3D_infer_show, players_3D_show)

### Show Video
frame_min, frame_max = min(seq_data_ground.index), max(seq_data_ground.index)
show_video(cap, 
           start_frame = frame_min, stop_frame = frame_max,
           ms_between_frames = 50)

### Show Ground Truth Events in Testing Data
for frame in features_3D_ground.keys():
    
    backgroundFrm = schemaFrm.to_BGR()
    frmDict = {'cFrm': backgroundFrm}
    featDict = features_3D_ground[frame]
    
    drawFunc = draw(['ball', 'playerTop', 'playerBot', 
                     'goodBounces', 'badBounces',
                     'stroke', 'ballSpeed']) ### takes frmDict and featDict, returns show frame
    
    showFrm = drawFunc(frmDict, featDict)
    
    show_continuous(showFrm.array, ms_between_frames = 40)
    
cv.destroyAllWindows()

### Show Inferred Events from HMM
for frame in features_3D_infer.keys():
    
    backgroundFrm = schemaFrm.to_BGR()
    frmDict = {'cFrm': backgroundFrm}
    featDict = features_3D_infer[frame]
    
    drawFunc = draw(['ball', 'playerTop', 'playerBot', 
                     'goodBounces', 'badBounces',
                     'stroke', 'ballSpeed']) ### takes frmDict and featDict, returns show frame
    
    showFrm = drawFunc(frmDict, featDict)
    
    show_continuous(showFrm.array, ms_between_frames = 75)
    
cv.destroyAllWindows()













