# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:57:32 2022

@author: gianl
"""



import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

from ATMANpy.funcs.draw_funcs import draw

# image operation
import cv2 as cv
import numpy as np

import pandas as pd

import os
from random import randrange

from copy import deepcopy

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################


### STORENAME ###
STORE_NAME = '10to100_ok2'

###### GET SEQUENCES #####
from ATMANpy.funcs.feature_funcs import read_play_sequences
seqFrm = read_play_sequences(VIDEO_ID, STORE_NAME, keep_which = 'last')


### GET LABELS ###

vidPath = atu.get_folder_struct('ball_event_hmm')[VIDEO_ID]
labPath = os.path.join(vidPath, 'labelled_sequences_segment_correction')

frmList = []
for file in os.listdir(labPath):
    frm = pd.read_csv(os.path.join(labPath, file), index_col = 0)
    frmList.append(frm)
    
    
labFrm = pd.concat(frmList)

# ### remove first versions of relabelled sequences ###
# keep_frames = []
# for index in labFrm.Index.unique():
#     keepFrm = labFrm[labFrm.Index == index]
    
#     starts, stops = keepFrm.StartFrame.unique(), keepFrm.StopFrame.unique()
#     if not len(starts) == len(stops) == 1:
#         keepFrm = keepFrm[keepFrm.StartFrame == starts[-1]] ### keep the last instance -> the relabelled data.
    
#     keep_frames.append(keepFrm)

# labFrm = pd.concat(keep_frames)
    

labFrm = labFrm.groupby('Index', group_keys = False).apply(lambda df: df.sort_values('FrameNum'))
labFrm = labFrm[labFrm.Event != 'NOHIT']


# labFrm = labFrm[~labFrm['FrameNum'].duplicated(keep = 'last')]


def show_index(idx):
    labels = labFrm[labFrm.Index == idx]
    start, stop = labels.StartFrame.unique()[0], labels.StopFrame.unique()[0]
    
    from ATMANpy.funcs.cv_funcs import show_video
    show_video(cap, start_frame = start, stop_frame = stop)


def show_index_with_labels(idx, ms_between_frames = 80):
    labels = labFrm[labFrm.Index == idx]
    start, stop = labels.StartFrame.unique()[0], labels.StopFrame.unique()[0]
    
    label_features = {}
    for frame in range(start, stop + 1):
        frameRow = labels[labels.FrameNum == frame]
        
        if len(frameRow) == 0:
            label_features[frame] = {'Event': ''}
        else:
            label_features[frame] = {'Event': frameRow.Event.values[0]}
    
    from ATMANpy.funcs.cv_funcs import replay
    replay(cap, label_features, start_frame = start, stop_frame = stop, 
           drawFunc = draw(['Event', 'frameNum']), ms_between_frames = ms_between_frames)




#### First, fill in the gaps of the labelled frames depending on the last event(s) recorded.

possible_events = sorted(labFrm.Event.unique())
print(possible_events)

serve_events = [event for event in possible_events if event.startswith('S')]
bounce_events = [event for event in possible_events if event.startswith('B')]
serve_bounce_events = [event for event in bounce_events if 'S' in event]
hit_bounce_events = [event for event in bounce_events if not 'S' in event]
hit_events = [event for event in possible_events if event.startswith('H')]


### 1) On Serve:

# Serve
    # {BONS, BOFS} -> Point over
    # CORD -> {BINS, BONS, BIFS, BOFS} -> Over
    # {BINS, BONS} -> Play Begins!
# between these: Insert flight event (direction -> if serve is near, UP, else DOWN)    
# before and after: insert NOISE event

def flight_direction(current_direction, current_event):
    if current_direction is None:
        if current_event in serve_events:
            if 'N' in current_event: ### near player serves -> ball flies up
                return 'FLIGHT_UP'
            else:
                return 'FLIGHT_DOWN'
        else:
            return None ### only a serve can INITIALIZE flight. Hits reverse it, bounces preserve it.
    
    else:
        if current_event in hit_events:
            if current_direction == 'FLIGHT_UP':
                assert current_event == 'HF', 'flight UP, but then hit by NEAR player...'
                return 'FLIGHT_DOWN'
            else:
                assert current_event == 'HN', 'flight DOWN, but then hit by FAR player...'
                return 'FLIGHT_UP'
    
        if current_event in bounce_events:
            if current_direction == 'FLIGHT_UP':
                assert 'F' in current_event, 'flight UP, but then bounce in NEAR field...'
                return 'FLIGHT_UP'
            else:
                assert 'N' in current_event, 'flight DOWN, but then bounce in FAR field...'
                return 'FLIGHT_DOWN'
            
        if current_event in serve_events:
            if current_direction == 'FLIGHT_UP':
                assert 'N' in current_event, 'flight UP, but FAR player served...'
                return 'FLIGHT_UP'
            else:
                assert 'F' in current_event, 'flight DOWN, but NEAR player served...'
                return 'FLIGHT_DOWN'
        
        if current_event == 'CORD':
            return current_direction
        
        if current_event in serve_events:
            return current_direction
      
    return None
    # print(current_event)
    # print(current_direction)
    # raise Exception('Should not get to this point')

def decide_point_status(current_event, previous_event):
    point_immediately_over = ['NET', 'BOF', 'BON', 'BOFS', 'BONS']
    
    if current_event is None:
        return 'BEFORE' ### point not yet started
        
    if previous_event is None:
        return 'DURING' ### there can not be a point end without at least TWO events!
    
    if current_event in point_immediately_over: ### fault
        return 'AFTER'  ### point done
    
    if current_event in bounce_events and previous_event in bounce_events: ### double bounce 
        return 'AFTER'
    
    if current_event in serve_bounce_events and previous_event == 'CORD': ### let (fault is dealt with above)
        return 'AFTER'
    
    return 'DURING' ### else, we are during rally!
    


def fill_in_directions(sampleFrm):
    
    ### IDEA -> here, we add in the directions of the ball during play
    ### There is a direction only for as long as the ball is in play, 
    ### otherwise, there is None
    
    start, stop = sampleFrm.StartFrame.unique()[0], sampleFrm.StopFrame.unique()[0]
    index = sampleFrm.Index.unique()[0]
    
    def make_row(frameNum, event):
        return pd.Series({'FrameNum': frameNum, 
                          'Event': str(event),
                          'Index': index,
                          'StartFrame': start,
                          'StopFrame': stop})
        
    
    outFrm_rows = []
    
    current_direction = None
    
    current_event = None
    previous_event = None

    current_status = None
    
    point_definitely_over = False

    for frame in range(start, stop + 1):
        
        if current_status != 'AFTER':
            
            ### what if we HAVE an event recorded at this frame?
            if frame in sampleFrm.FrameNum.unique(): 
                frameRow = sampleFrm[sampleFrm.FrameNum == frame]
                
                ### avoid recording as an event a continuation of an event in progress
                if not current_event == frameRow.Event.values[0]: 
                    previous_event = current_event
                    current_event = frameRow.Event.values[0]
            
                    current_status = decide_point_status(current_event, previous_event)
                    current_direction = flight_direction(current_direction, current_event)
                    
                outFrm_rows.append(make_row(frame, current_event)) ### basically, reappend the row...
            
            
            ### what if we DONT HAVE an event recorded at this frame.
            else:
                outFrm_rows.append(make_row(frame, current_direction))
                
            # print(f'{frame}: {current_direction}')
                
        else:
            if frame in sampleFrm.FrameNum.unique() and not point_definitely_over:
                frameRow = sampleFrm[sampleFrm.FrameNum == frame]
                if current_event == frameRow.Event.values[0]: ### this is for the case in which
                                                                # a point-ending event takes more than
                                                                # one frame.
                    outFrm_rows.append(make_row(frame, current_event))
                else:
                    outFrm_rows.append(make_row(frame, None))
            
            else:
                outFrm_rows.append(make_row(frame, None))
                
                point_definitely_over = True ### if status is 'AFTER', and we detect only one non-event 
                                                ### -> all is over!
                
    return pd.concat(outFrm_rows, axis = 1).T.sort_values('FrameNum')


# fillFrm = labFrm.groupby('Index', group_keys = False).apply(fill_in_directions)

### needs corrections before applying

# sampleIdx = 130
# sampleFrm = labFrm[labFrm.Index == sampleIdx]

# chk = fill_in_directions(sampleFrm)

# print(sampleFrm)

# show_index(sampleIdx)
# show_index_with_labels(sampleIdx, ms_between_frames = 160) ### mislabelled!

### fixing this issue...
problematic_frames = [95417, 95422, 95423]
problem_filter = labFrm.FrameNum.isin(problematic_frames), 'Event'
labFrm.loc[problem_filter] = labFrm.loc[problem_filter].str.replace('N', 'F')

### ... and another issue.
problematic_frames = [95552]
problem_filter = labFrm.FrameNum.isin(problematic_frames), 'Event'
labFrm.loc[problem_filter] = labFrm.loc[problem_filter].str.replace('N', 'F')


fillFrm = labFrm.groupby('Index', group_keys = False).apply(fill_in_directions)

###### Now, add in 'observation' data -> the location, velocity and acceleration of the ball.

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)
splitList = split_joint_features(stream_features)

sf = splitList[0]
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours
sf_itpl = smart_interpolate_ball_contours(sf, verbose = False)


def get_observation_frame(sf):
    
    VELOCITY_SHIFT = 3 ### window size of six in almajai et al (2010)
    ACCELERATION_SHIFT = 1 ### window size of 2 ibid.
    
    def get_ball(fd):
        if not fd['ball'] is None:
            return fd['ball'].center().coords
        else:
            return (np.nan, np.nan)
    
    ballDict = {frmNum: get_ball(featDict) for frmNum, featDict in sf.items()}
    ballFrm = pd.DataFrame.from_dict(ballDict, orient = 'index', columns = ['x', 'y'])
    
    ###### SPEED #######
    speedFrm = ballFrm.shift(-VELOCITY_SHIFT) - ballFrm.shift(VELOCITY_SHIFT) 
    
    ### help out with nans at beginning and end by using an approximate speed and accel formula
    # a) only using information from earlier (for end of sequences.)
    secondary_speedFrm = (ballFrm - ballFrm.shift(VELOCITY_SHIFT)) * 2
    speedFrm = speedFrm.fillna(secondary_speedFrm)
    
    # b) only using information from later (for beginning of sequences.)
    secondary_speedFrm = (ballFrm.shift(-VELOCITY_SHIFT) - ballFrm) * 2
    speedFrm = speedFrm.fillna(secondary_speedFrm)
    
    #### ACCELERATION ####
    accelFrm = speedFrm.shift(-ACCELERATION_SHIFT) - speedFrm.shift(ACCELERATION_SHIFT)
    
    ### help out with nans at beginning and end by using an approximate speed and accel formula
    # a) only using information from earlier (for end of sequences.)
    secondary_accelFrm = (speedFrm - speedFrm.shift(ACCELERATION_SHIFT)) * 2
    accelFrm = accelFrm.fillna(secondary_accelFrm)
    
    # b) only using information from later (for beginning of sequences.)
    secondary_accelFrm = (speedFrm.shift(-ACCELERATION_SHIFT) - speedFrm) * 2
    accelFrm = accelFrm.fillna(secondary_accelFrm)
    
    #### LABEL AND CONCAT
    speedFrm.columns = ['v_x', 'v_y']
    accelFrm.columns = ['a_x', 'a_y']
    
    obsFrm = pd.concat([ballFrm, speedFrm, accelFrm], axis = 1)

    return obsFrm
    

### JOIN OBSERVATIONS AND LABELS ###

fillSer = fillFrm[['FrameNum', 'Event', 'Index']].set_index('FrameNum')

jointList = []

for idx, sf in enumerate(splitList):
    sf_itpl = smart_interpolate_ball_contours(sf)
    
    obsFrm = get_observation_frame(sf_itpl)
    
    ### join this specific set of observations with all labels...
    jointFrm = pd.concat([obsFrm, fillSer], axis = 1)
    ### ... and keep only the ones we have data for!
    jointFrm = jointFrm[~jointFrm[obsFrm.columns].isna().any(axis = 1)] 
    ### (adding labels is easy and ad hoc by nature, data is created by process pipeline.)
    
    ### initialize an index and a sequence status that is OK as long as we don't find any problems
    jointFrm['IndexSF'] = idx
    jointFrm['SeqStatus'] = 'OK'
    
    ### Check if a point needs relabeling (or labels in the first place).
    eventList = []
    for event in jointFrm.Event:
        if eventList == []:
            eventList.append(event)
        else:
            if event is np.nan:
                if not eventList[-1] is np.nan:
                    eventList.append(event)
            else:
                if event != eventList[-1]:
                    eventList.append(event)
    
    try:
        fst, sec = eventList[:2]
    except: ### happens only if there is just one value 
        jointFrm['SeqStatus'] = 'RELABEL'
    else:
        if fst is np.nan and sec in serve_events: ### in this case, the serve migtht be a bit longer
            jointFrm['SeqStatus'] = 'RELABEL' 
        else:
            if fst is np.nan and sec == 'None': ### in this case, the np.nan is redundant!
                fst = sec ### so second becomes first...
                try:
                    sec = eventList[2] ### ... and third becomes second element!
                except:
                    jointFrm['SeqStatus'] = 'RELABEL'
        
            if not (fst in serve_events or sec in serve_events and fst == 'None'): ### a 'legal' sequence starts with serve or none and serve
                jointFrm['SeqStatus'] = 'PROBLEM'
    
    jointList.append(jointFrm)

jointFrm = pd.concat(jointList)


###### ANALYSE PROBLEMS AND RELABELS #######
probFrm = jointFrm[jointFrm.SeqStatus == 'PROBLEM']
probs = probFrm.IndexSF.unique()

redoFrm = jointFrm[jointFrm.SeqStatus == 'RELABEL']
redos = redoFrm.IndexSF.unique()


def show_idx(idx):
    from ATMANpy.funcs.cv_funcs import replay

    sf = splitList[idx]
    sf_itpl = smart_interpolate_ball_contours(sf)

    print(f'Sequence from: {min(sf)} to {max(sf)}')

    replay(cap, sf_itpl, drawFunc = draw(['playerTop', 'playerBot', 'ball', 'courtLines']),
           ms_between_frames = 30)


### CHECK PROBLEMS ###
# -> not a single one!
print(len(probs))
# idx = probs[0]
# chk = jointFrm[jointFrm.IndexSF == idx]
# sf = splitList[idx]

### CHECK RELABELS ###
idx = redos[0]
chk = jointFrm[jointFrm.IndexSF == idx]
#### -> long sequence we did not catch!


idx = redos[1]
chk = jointFrm[jointFrm.IndexSF == idx]
### -> this is a play sequence was split in two by ballboy segmentation.
### thus, we labelled it as if it was one sequence 
### (since at the time of labelling we did not have ballboy segmentation).
#### -> sequence needs to be split ahead of fill_in_directions!!!
## DONE!

show_idx(idx)

show_idx(idx - 1)


idx = redos[2]
chk = jointFrm[jointFrm.IndexSF == idx]
#### -> long sequence we did not catch!

idx = redos[3]
chk = jointFrm[jointFrm.IndexSF == idx]
#### -> nan into serve


idx = redos[4]
chk = jointFrm[jointFrm.IndexSF == idx]
#### -> nan into serve

#### CREATE THE REDO PLAY SEQUENCES ###

redo_sequences = pd.DataFrame(columns = ['StartFrame', 'StopFrame'])
for redoIdx in redos:
    myRedo = redoFrm[redoFrm.IndexSF == redoIdx]
    start, stop = myRedo.index.min(), myRedo.index.max()

    appSer = pd.Series({'StartFrame': start, 'StopFrame': stop}, name = redoIdx)
    redo_sequences = redo_sequences.append(appSer)
    
    
savePath = atu.get_folder_struct('ball_event_hmm')[VIDEO_ID]
seqPath = os.path.join(savePath, 'relabel_sequences')

import datetime as dt
timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
savename = f'relabel_sequences_{timestamp}.csv'
redo_sequences.to_csv(os.path.join(seqPath, savename))


import os
evtDict = atu.get_folder_struct('ball_event_hmm')
savePath = os.path.join(evtDict['main'], VIDEO_ID, 'labelled_sequences_corrections')

to_drop = list(redoFrm.Index.dropna().unique()) + list(probFrm.Index.dropna().unique())

redos_removed = labFrm[~labFrm.Index.isin(to_drop)]

redos_removed.to_csv(os.path.join(savePath, 'events_without_redos.csv'))


