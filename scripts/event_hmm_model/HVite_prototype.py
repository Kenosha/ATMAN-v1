# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:27:28 2022

@author: gianl
"""


import os
import pandas as pd
import numpy as np

from ATMANpy.funcs.HMM.utils import read_hmm_params, read_train_test

VIDEO_ID = 'pqxAmUWIoKc'
MODEL_NAME = 'embedded_cyc3_nrel85'


train_data, test_data = read_train_test(MODEL_NAME, VIDEO_ID)
hmmDict = read_hmm_params(MODEL_NAME, VIDEO_ID)

#%% MAKE INFERENCE

### Token passing algorithm - Full Sentence, no grammar

randseqIdx = np.random.choice(test_data.IndexSF.unique())
randSeq = test_data[test_data.IndexSF == randseqIdx]
words = randSeq.drop(columns = ['Event', 'IndexSF'])

from ATMANpy.funcs.HMM.HVite import token_pass_sequence
inferred_sequence = token_pass_sequence(hmmDict, words)

print(inferred_sequence)

#%% VISUAL INSPECTION ###

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.funcs.draw_funcs import draw

from ATMANpy.funcs.cv_funcs import replay

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

### STORENAME ###
STORE_NAME = '10to100_ok2'
#################

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


for test_sequence in test_data.IndexSF.unique():
    randSeq = test_data[test_data.IndexSF == test_sequence]
    words = randSeq.drop(columns = ['Event', 'IndexSF'])

    inferred_sequence = token_pass_sequence(hmmDict, words)
        
    frame_min, frame_max = words.index.min(), words.index.max()
    expanded_sequence = expand_inferred_sequence(inferred_sequence, word_beginning_frame = frame_min)

    for frame, inferred_event in expanded_sequence.items():
        stream_features[frame]['event'] = inferred_event

    replay(cap, stream_features, 
           start_frame = frame_min, stop_frame = frame_max,
           ms_between_frames = 100,
           drawFunc = draw(['playerTop', 'playerBot', 'ball', 'courtLines', 'event']))
    
    
#%%

from ATMANpy.funcs.HMM.TokenPass import read_grammar, infer

grammarStr = 'serve_fault'
grammar = read_grammar(grammarStr)
infer(words, hmmDict, grammar)


start = min(words.index)
stop = max(words.index)

from ATMANpy.funcs.util_funcs import get_video        
from ATMANpy.funcs.cv_funcs import show_video, get_frame, show

cap = get_video(VIDEO_ID)
show_video(cap, start_frame = start, stop_frame = stop)
    
show(get_frame(cap, 53687))  




