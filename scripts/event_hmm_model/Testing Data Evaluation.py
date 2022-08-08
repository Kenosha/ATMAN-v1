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

#%% <AKE INFERENCE ON TEST CASES

### GET STREAM FEATURES ###
from ATMANpy.funcs.feature_funcs import load_stored_features
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)

### ENRICH STREAM FEATURES WITH INFERENCE ###
from ATMANpy.funcs.HMM.HPost import expand_inferred_sequence

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

    
#%% VISUAL INSPECTION
import cv2 as cv

######## GET SCHEMA ########
from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)
    
### CHOOSE A SEQUENCE ###
TEST_SEQUENCE = 108
#########################

## PREPARE 3D FEATURES ##
testSeq = test_data[test_data.IndexSF == TEST_SEQUENCE]
words = testSeq.drop(columns = ['Event', 'IndexSF'])

inferred_events = infer_grammatic_dict[test_sequence]
    
test_sequence_features = {frame: featDict for frame, featDict in stream_features.items() if frame in words.index}

import ATMANpy.funcs.HMM.HPost as HPost

projDict = HPost.make_projection_matrices(cap, test_sequence_features, schemaLines)

players_3D = HPost.find_player_positions_3D(projDict, test_sequence_features)

inf_words = HPost.merge_words_and_inference(words, inferred_events)
inf_words = HPost.init_spatial_dimensions(inf_words)
inf_words = HPost.impose_serve(inf_words, schemaLines)
inf_words = HPost.impose_net(inf_words, schemaLines)
inf_words = HPost.impose_bounce(inf_words, bounce_height = 0.2)
inf_words = HPost.impose_hit(inf_words, players_3D)

ball_3D = HPost.find_ball_positions_3D(inf_words, projDict)

features_3D = HPost.make_3D_features(inf_words, ball_3D, players_3D)
########################

#### VISUALIZE #########
show_video(cap, 
        start_frame = frame_min, stop_frame = frame_max,
        ms_between_frames = 50)

for frame in features_3D.keys():
    
    backgroundFrm = schemaFrm.to_BGR()
    frmDict = {'cFrm': backgroundFrm}
    featDict = features_3D[frame]
    
    drawFunc = draw(['ball', 'playerTop', 'playerBot', 
                     'goodBounces', 'badBounces',
                     'ballSpeed', 'stroke']) ### takes frmDict and featDict, returns show frame
    
    showFrm = drawFunc(frmDict, featDict)
    
    show_continuous(showFrm.array, ms_between_frames = 50)
    
cv.destroyAllWindows()
    