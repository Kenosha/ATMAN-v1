# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:13:36 2023

@author: gianl
"""



####### READ IN #######

import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from ATMANpy.types.Abstract import Points, Point, Shots, Shot, Trajectory
from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous

from ATMANpy.types.Visual3D import View, Camera, Scene
from ATMANpy.types.ShapesTennis3D import Ball, Player, Racquet, Mark

import ATMANpy.auxil.constants as const

from ATMANpy.funcs.util_funcs import get_folder_struct, make_timestamp
graphDir = get_folder_struct('graphs')['hmm_validation']

video_id = 'pqxAmUWIoKc'
store_name = '10to100_ok2'
model_name = 'embedded_cyc3_nrel85'

event_name = 'grammatic_event'
ball_name = '3D_Ball_smart_inference'
top_player_name = '3D_topPlayer'
bot_player_name = '3D_botPlayer'

infPts = Points.from_stored_features(video_id, store_name, name = 'Inference',
                                     event_name = event_name, ball_name = ball_name,
                                     top_player_name = top_player_name, bot_player_name = bot_player_name,
                                     narrow_instantaneous_events = False) ### get raw, unnarrowed events

event_name = 'true_event'
ball_name = '3D_Ball_smart_ground'
truPts = Points.from_stored_features(video_id, store_name, name = 'Ground Truth',
                                     event_name = event_name, ball_name = ball_name,
                                     top_player_name = top_player_name, bot_player_name = bot_player_name,
                                     narrow_instantaneous_events = False)  ### get raw, unnarrowed events

#%%

def make_folder(graphDir = graphDir, model_name = model_name):
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
paramDict = dict(video_id = 'pqxAmUWIoKc', store_name = '10to100_ok2', model_name = 'embedded_cyc3_nrel85')

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




