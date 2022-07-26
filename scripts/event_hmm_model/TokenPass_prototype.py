# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 00:40:44 2022

@author: gianl
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:30:11 2022

@author: gianl
"""


import os
import pandas as pd
import numpy as np

from ATMANpy.funcs.util_funcs import get_folder_struct

np.seterr('ignore')

VIDEO_ID = 'pqxAmUWIoKc'
DATA_SET = 'events_final.csv'

hmmPath = get_folder_struct('ball_event_hmm')[VIDEO_ID]
dataPath = os.path.join(hmmPath, 'hmm_data_final')
dataFile = os.path.join(dataPath, DATA_SET)

dataFrm = pd.read_csv(dataFile, index_col = 0).drop(columns = 'SeqStatus')
dataFrm.index.name = 'Frame'
dataFrm = dataFrm[~dataFrm.Event.isna()]
all_words = dataFrm.Event.unique()

from ATMANpy.funcs.HMM.utils import split_into_training_and_testing
train_data, test_data = split_into_training_and_testing(dataFrm, relative_split = 0.8)

####################################################
###### Initialize HMMs using HInit and HRest #######
####################################################

### 1) Choose Words
from ATMANpy.funcs.HMM.utils import choose_random_sample_words
word_training_sample = choose_random_sample_words(train_data)

### 2) Assign a number of states to each word ####
# -> for the moment, assume they're all three state hmms, except None, which is one state.
event_to_num_states = lambda wordStr: 1 if wordStr == 'None' else 3

from ATMANpy.funcs.HMM.high_level import init_all_single_mixture

initDict = init_all_single_mixture(word_training_sample, 
                                   event_to_num_statesFunc = event_to_num_states)

print('_____________________')
print('')
print('INITIALIZATION DONE!')        
print('_____________________')


####################################################
###### Use HERest to improve HMM Parameters ########
####################################################

NUM_CYCLES = 3

from ATMANpy.funcs.HMM.HERest import HERest
hmmDict = HERest(initDict, train_data, num_cycles = NUM_CYCLES)


print('____________________________')
print('')
print('EMBEDDED RE-ESTIMATION DONE!')        
print('____________________________')


#### Token Passing Algorithm - One Word
from ATMANpy.funcs.HMM.utils import get_word_data

randwordStr = np.random.choice(all_words)
word_dataFrm = get_word_data(test_data, randwordStr, extend_word = 1)

randIdx = np.random.choice(word_dataFrm.index.get_level_values('IndexWord').unique())
word = word_dataFrm.loc[randIdx]

from ATMANpy.funcs.HMM.HVite import token_pass_single_word

inferFrm = token_pass_single_word(hmmDict, word)
inferFrm.head(3)

print(f'The random word is: {randwordStr}')
print('Best inference:')
print(inferFrm.head(3))

### Token passing algorithm - Full Sentence, no grammar

randseqIdx = np.random.choice(test_data.IndexSF.unique())
randSeq = dataFrm[dataFrm.IndexSF == randseqIdx]
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
from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
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
    randSeq = dataFrm[dataFrm.IndexSF == test_sequence]
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
    
    
    


