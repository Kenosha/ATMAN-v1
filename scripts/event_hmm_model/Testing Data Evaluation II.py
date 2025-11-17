# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:13:36 2023

@author: gianl
"""


import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from ATMANpy.types.Abstract import Points, Point, Shots, Shot, Trajectory
from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous

from ATMANpy.types.Visual3D import View, Camera, Scene
from ATMANpy.types.ShapesTennis3D import Ball, Player, Racquet, Mark

from ATMANpy.types.Features import StreamFeature, StreamFeatures
from ATMANpy.funcs.feature_funcs import save_events

import ATMANpy.auxil.constants as const

from ATMANpy.funcs.util_funcs import get_folder_struct, make_timestamp
graphDir = get_folder_struct('graphs')['hmm_validation']

VIDEO_ID = 'pqxAmUWIoKc'
STORE_NAME = '10to100_inout_logit_tight'
MODEL_NAME = 'embedded_cyc3_nrel85_inout_logit'
GRAMMAR_NAME = 'full_grammar_test'

from ATMANpy.funcs.util_funcs import get_video  
cap = get_video(VIDEO_ID)

#%% MAKE INFERENCE ON TEST CASES

from ATMANpy.funcs.HMM.utils import read_hmm_params, read_train_test
train_data, test_data = read_train_test(MODEL_NAME)
hmmDict = read_hmm_params(MODEL_NAME)

from ATMANpy.funcs.HMM.TokenPass import read_grammar, infer
grammar = read_grammar(GRAMMAR_NAME)

from ATMANpy.funcs.HMM.HVite import token_pass_sequence

### SAVE TRUE FEATURES
trueSF = StreamFeature.from_series(test_data.Event).rename('true_event')
trueSF.save(VIDEO_ID, STORE_NAME)

### FIGURE OUT WHAT WAS INFERRED ALREADY, AND REMOVE FROM TO_DO
inferSF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME, 
                                    featnameList = ['arbitrary_event', 'grammatic_event'])

test_data_to_do = test_data[~test_data.index.isin(inferSF.get_frames())]
sequences_to_do = test_data_to_do.IndexSF.unique()


### ENRICH STREAM FEATURES WITH INFERENCE ###
from ATMANpy.funcs.HMM.HPost import expand_inferred_sequence

arbitrarySFs = []
grammaticSFs = []
for test_sequence in sequences_to_do:
    print(f'Doing Test Sequence {test_sequence}...')
    
    randSeq = test_data[test_data.IndexSF == test_sequence]
    words = randSeq.drop(columns = ['Event', 'IndexSF'])
    frame_min, frame_max = words.index.min(), words.index.max()
    
    print('    Doing arbitrary inference...')
    arbitrary_sequence = token_pass_sequence(hmmDict, words)
    print('    Doing grammatical inference...')
    grammatic_sequence = infer(words, hmmDict, grammar)    
    
    arbitrary_expanded = expand_inferred_sequence(arbitrary_sequence, 
                                                  word_beginning_frame = frame_min)
    grammatic_expanded = expand_inferred_sequence(grammatic_sequence, 
                                                  word_beginning_frame = frame_min)

    arbitrarySFs.append(StreamFeature(arbitrary_expanded, name = 'arbitrary_event'))
    grammaticSFs.append(StreamFeature(grammatic_expanded, name = 'grammatic_event'))

### SAVE
from ATMANpy.funcs.feature_funcs import join_split_features
arbitrarySF = StreamFeature(join_split_features(arbitrarySFs), name = 'arbitrary_event')
grammaticSF = StreamFeature(join_split_features(grammaticSFs), name = 'grammatic_event')

if not arbitrarySF.is_empty():
    arbitrarySF.save(VIDEO_ID, STORE_NAME)
if not grammaticSF.is_empty():
    grammaticSF.save(VIDEO_ID, STORE_NAME)

#%% READ EXISTENT FEATURES

SF = StreamFeatures.from_store(VIDEO_ID, STORE_NAME)

INTERPOLATION_TYPE = 'smart' # linear or smart
EVENT_SOURCE = 'ground' # ground or inference

from ATMANpy.funcs.feature_funcs import read_feature_frame

try:
    top_player_3D = read_feature_frame('3D_topPlayer', VIDEO_ID, STORE_NAME)
except:
    top_player_3D = pd.DataFrame()

try:
    bot_player_3D = read_feature_frame('3D_botPlayer', VIDEO_ID, STORE_NAME)
except:
    bot_player_3D = pd.DataFrame()

try:
    ball_3D = read_feature_frame(f'3D_Ball_{INTERPOLATION_TYPE}_{EVENT_SOURCE}', VIDEO_ID, STORE_NAME)
except:
    ball_3D = pd.DataFrame()



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
    seqSF = SF[list(seq_data.index)]
    
    if EVENT_SOURCE == 'inference':
        seq_data_hidden = seq_data.drop(columns = ['Event', 'IndexSF']) ### Drop it, and merge in inference 
        seq_data = HPost.merge_words_and_event_features(seq_data_hidden, seqSF['grammatic_event'])

    ## UNDERSTAND IF THE FEATURES HAVE ALREADY BEEN CREATED
    players_done = seq_data.index.isin(top_player_3D.index).all()
    ball_done = seq_data.index.isin(ball_3D.index).any() ### used to be .all(), but that is slightly too strict, as there is no interpolation if flight at the very end.
    
    if players_done and ball_done:
        print(f'Found in stored 3D features: {test_sequence}...')
        continue
    else:
        print(f'Post Processing Sequence Number: {test_sequence}...')
    
    ### POST PROCESS FEATURES
    projDict = HPost.make_projection_matrices(cap, seqSF, schemaLines)
    
    if not players_done:
        ## 1) Players -> independent of whether we use ground truth events or inferred events!
        top_player_3D, bot_player_3D = HPost.find_player_positions_3D(projDict, SF)
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
    

#%%###### FEATURE READ IN #######

event_name = 'grammatic_event'
ball_name = '3D_Ball_smart_inference'
top_player_name = '3D_topPlayer'
bot_player_name = '3D_botPlayer'

infPts = Points.from_stored_features(VIDEO_ID, STORE_NAME, name = 'Inference',
                                     event_name = event_name, ball_name = ball_name,
                                     top_player_name = top_player_name, bot_player_name = bot_player_name,
                                     narrow_instantaneous_events = False) ### get raw, unnarrowed events

event_name = 'true_event'
ball_name = '3D_Ball_smart_ground'
truPts = Points.from_stored_features(VIDEO_ID, STORE_NAME, name = 'Ground Truth',
                                     event_name = event_name, ball_name = ball_name,
                                     top_player_name = top_player_name, bot_player_name = bot_player_name,
                                     narrow_instantaneous_events = False)  ### get raw, unnarrowed events

#%% EVALUATION FUNCTIONS ####

def make_folder(graphDir = graphDir, model_name = MODEL_NAME):
    model_folder = os.path.join(graphDir, model_name)
        
    try:
        os.mkdir(model_folder)
    except FileExistsError:
        pass
            
    save_folder = os.path.join(model_folder, make_timestamp())
    
    try:
        os.mkdir(save_folder)
    except FileExistsError:
        pass
    
    return save_folder
            
def make_text(textDict, save_where):
    with open(os.path.join(save_where, 'general_info.txt'), mode = 'w') as f:
        for k in textDict:
            if k[0] == ':': ### these are headings!
                indent = ''
            else:
                indent = '  '
            f.writelines(f'{indent}{k}: {textDict[k]}\n')
    
def make_boxplot(jnt_dev, jnt_dur, name, save_where):
    
    STYLE = 'dark_background'
    plt.style.use(STYLE)
    
    medColor = 'gold'
    propDict = {'linewidth': 2}
    
    lw = 2
    widths = 0.3
    
    fig, axes = plt.subplots(1, 2, figsize = (14, 7))
    fig.suptitle(name, fontsize = 22)
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['bottom'].set_linewidth(lw)
        ax.spines['left'].set_linewidth(lw)
        # ax.spines['bottom'].set_visible(False)
    
    axes[0].set_title('Frame Duration', fontsize = 18)
    axes[1].set_title('Frame Difference (Inference rel. to Ground Truth)', fontsize = 18)
    
    axes[0].boxplot(jnt_dur, labels = jnt_dur.columns, widths = widths,
                    # color = colorDict, 
                    capprops = propDict, boxprops = propDict, whiskerprops = propDict, 
                    flierprops = propDict, medianprops = {**propDict, 'color': medColor})
    axes[1].boxplot(jnt_dev, labels = jnt_dev.columns, widths = widths,
                    # color = colorDict, 
                    capprops = propDict, boxprops = propDict, whiskerprops = propDict, 
                    flierprops = propDict, medianprops = {**propDict, 'color': medColor})
    
    axes[0].tick_params(axis='x', labelsize = 18)
    axes[1].tick_params(axis='x', labelsize = 18)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_where, f'{name}.png'))
    
    plt.show()
    

#%% PREPARING FOLDER AND FUNCS

save_folder = make_folder()

textDict = {}
textDict['::: 1 ::'] = 'Overview :::'
paramDict = dict(video_id = VIDEO_ID, store_name = STORE_NAME, model_name = MODEL_NAME)

textDict = {**textDict, **paramDict}

###### criteria to evaluate the inference on:
    ## General criteria:
        # determine winner of point correctly?
        # general sequence of events
        
    ## Tightness of events (temporally):
        # - how close are serves, hits
        # - how close are inferred bounce frames from real bounces?
        
    ## Correctness of events:
        # - in/out
        
### General Criteria 
def winner_comp(infPt, truPt):
    return truPt.get_winner() == infPt.get_winner()
        
def event_comp(infPt, truPt):
    return truPt.get_events() == infPt.get_events()

def first_divergence(infPt, truPt):
    inf_events, tru_events = infPt.get_events(), truPt.get_events()
    for idx in range(len(inf_events)):
        if inf_events[idx] != tru_events[idx]:
            return f'{tru_events[idx]} -> {inf_events[idx]}'
    return None

### Difference in event duration
def frame_duration(myFrm, dropna = True):
    if dropna:
        myFrm = myFrm.copy()
        myFrm = myFrm.dropna()    
    return myFrm['end'] - myFrm['start'] + 1

### Difference in starting, ending frames
def frame_difference(myFrm, dropna = True):
    if dropna:
        myFrm = myFrm.copy()
        myFrm = myFrm.dropna()    
    
    strFrm = myFrm['start']
    str_dev = strFrm['Inference'] - strFrm['Ground Truth'] # we want deviation of inference w/r/t ground
    str_dev.name = 'Start Frame'
    
    endFrm = myFrm['end']
    end_dev = endFrm['Inference'] - endFrm['Ground Truth'] # we want deviation of inference w/r/t ground
    end_dev.name = 'End Frame'
    
    return pd.concat([str_dev, end_dev], axis = 1)

def to_pct(inFloat):
    return f'{round(inFloat * 100, 2)} %' 

#%% EVALUATION

######## GENERAL OVERVIEW ########
winnerList = []
eventList = []
divList = []
for infPt, truPt in zip(infPts, truPts):
    winnerList.append(winner_comp(infPt, truPt))
    eventList.append(event_comp(infPt, truPt))
    divList.append(first_divergence(infPt, truPt))

evalFrm = pd.concat([pd.Series(winnerList, name = 'winner_correct'), 
                     pd.Series(eventList, name = 'event_correct')], 
                    axis = 1)
winner, event = evalFrm.mean()

textDict['::: 1A ::'] = 'General Metrics :::'
textDict['Point Winner Correct'] = to_pct(winner)
textDict['Event Sequence Correct'] = to_pct(event)

### first divergence -> where in the event chain does inference diverge from ground first?
textDict['::: 1B ::'] = 'First Divergence :::'

divList = sorted([el for el in divList if el is not None])
divDict = pd.Series(divList).value_counts().to_dict()

textDict = {**textDict, **divDict}

##################################

###### DEEPER ANALYSIS ###########
compFrm = truPts.compare(infPts)

tru_event = compFrm[('event', 'Ground Truth')].fillna('-')
inf_event = compFrm[('event', 'Inference')].fillna('-')

##### A) SERVE #######
textDict['::: 2 ::'] = 'Serve Metrics :::'
myFrm = compFrm[tru_event.str.startswith('S') | inf_event.str.startswith('S')]
evtFrm = myFrm['event']

### False Positive, Negative Rates ###
false_negative = evtFrm['Inference'].isna().mean()
false_positive = evtFrm['Ground Truth'].isna().mean()

textDict['Serve: False Negative'] = to_pct(false_negative)
textDict['Serve: False Positive'] = to_pct(false_positive)

### Correctness
bothFrm = evtFrm.dropna()
false_identification = (bothFrm['Ground Truth'] != bothFrm['Inference']).mean()

textDict['Serve: False Identification'] = to_pct(false_identification)

### Timings ###
jnt_dev = frame_difference(myFrm, dropna = True)
jnt_dur = frame_duration(myFrm, dropna = True)

make_boxplot(jnt_dev, jnt_dur, 'Serve', save_folder)

##### B) HITS ####
textDict['::: 3 ::'] = 'Hit Metrics :::'
myFrm = compFrm[tru_event.str.startswith('H') | inf_event.str.startswith('H')]
evtFrm = myFrm['event']

### False Positive, Negative Rates ###
false_negative = evtFrm['Inference'].isna().mean()
false_positive = evtFrm['Ground Truth'].isna().mean()

textDict['Hit: False Negative'] = to_pct(false_negative)
textDict['Hit: False Positive'] = to_pct(false_positive)

### Correctness
bothFrm = evtFrm.dropna()
false_identification = (bothFrm['Ground Truth'] != bothFrm['Inference']).mean()

textDict['Hit: False Identification'] = to_pct(false_identification)

### Timings ###
jnt_dev = frame_difference(myFrm, dropna = True)
jnt_dur = frame_duration(myFrm, dropna = True)
make_boxplot(jnt_dev, jnt_dur, 'Hits', save_folder)

##### C) BOUNCES #####
textDict['::: 4 ::'] = 'Bounce Metrics :::'
myFrm = compFrm[tru_event.str.startswith('B') | inf_event.str.startswith('B')]
evtFrm = myFrm['event']

### False Positive, Negative Rates ###
false_negative = evtFrm['Inference'].isna().mean()
false_positive = evtFrm['Ground Truth'].isna().mean()

textDict['Bounce: False Negative'] = to_pct(false_negative)
textDict['Bounce: False Positive'] = to_pct(false_positive)

### Correctness
bothFrm = evtFrm.dropna()
false_identification = (bothFrm['Ground Truth'] != bothFrm['Inference']).mean()

textDict['Bounce: False Identification'] = to_pct(false_identification)

### Timings ###
jnt_dev = frame_difference(myFrm, dropna = True)
jnt_dur = frame_duration(myFrm, dropna = True)
make_boxplot(jnt_dev, jnt_dur, 'Bounces', save_folder)

### In versus Out ###
inFrm = evtFrm[evtFrm['Ground Truth'].str[1] == 'I'].dropna() # only among inferred bounces!
outFrm = evtFrm[evtFrm['Ground Truth'].str[1] == 'O'].dropna() # only among inferred bounces!

false_identification_out = (inFrm['Inference'].str[1] == 'O').mean() ### if the shot was in, what percentage was judged out?
false_identification_in = (outFrm['Inference'].str[1] == 'I').mean() ### if the shot was in what percentage was judged out?
 
textDict['Bounce: Called Out where In'] = to_pct(false_identification_out)
textDict['Bounce: Called In where Out'] = to_pct(false_identification_in)


###### WRITE #######

make_text(textDict, save_folder)

#%% Testing




