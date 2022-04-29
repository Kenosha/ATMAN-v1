#!/usr/bin/env python

'''
BASED ON: open cv video_threded sample script

Multithreaded video processing sample.
Usage:
   video_threaded.py {<video device number>|<video file name>}
   Shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.
Keyboard shortcuts:
   ESC - exit
   space - switch between multi and single threaded processing
'''

import numpy as np
import cv2 as cv

from multiprocessing.pool import ThreadPool
from collections import deque

import ATMANpy.funcs.util_funcs as atu

#%% Functions ###

import time
from copy import deepcopy
from ATMANpy.types.Frames import BGRFrame, GreyFrame

from ATMANpy.funcs.high_level_funcs import track_courtlines
from functools import partial


VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
start_frame = 10280
stop_frame = 100000

def get_features_parallel(cap, start_frame, stop_frame, verbose = True):

    ### do the fast function -> 225 only
    my_courtline_func = partial(track_courtlines, get_net = False, only_225 = True)    

    time_start = time.time()
    time_now = time_start
    
    stream_features = {}
    H, config = None, None
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    while True:
        
        if len(pending) < threadn:
            frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            _ret, frame = cap.read()
            
            cFrm = BGRFrame(frame)
            gFrm = cFrm.to_grey()
            
            task = pool.apply_async(my_courtline_func, (gFrm, H, config))

            pending.append((frmNum, gFrm, task))
            
        while len(pending) > 0 and pending[0][2].ready():
            num, frm, task = pending.popleft()
            courtLines, H, config, trackFlag = task.get()
            line_feature_dict = {'courtLines': courtLines, 
                                 'H': H, 
                                 'trackFlag': trackFlag}
            stream_features[num] = deepcopy(line_feature_dict)
            
            if num % 1000 == 0:
                time_lst = time_now
                time_now = time.time()
                time_elapsed = round(time_now - time_lst, 2)
                print(f'  Done with {num}, took {time_elapsed} seconds...')
            
            if verbose:
                showArr = frm.to_BGR().put(courtLines).array
                cv.imshow('threaded video', showArr)
            
        ch = cv.waitKey(1)
        if ch == 27:
            break
        
        if frmNum > stop_frame:
            break

    totFrames = stop_frame - start_frame
    time_elapsed = time.time() - time_start
    minutes_elapsed = round(time_elapsed / 60, 1)
    print(f'Done with {totFrames} frames. Took {minutes_elapsed} minutes')

    cv.destroyAllWindows()
    return stream_features

my_features = get_features_parallel(cap, start_frame, stop_frame, verbose = False)

###################
### Second Pass ###
###################

# here, we:
# -> re-try "stutter frames" (where tracking was not found for a few frames)

def get_redo_frames(my_features, 
                    redo_threshold = 24): 
                    ### below this many consequent frames that were not found,
                    ### we will retry all of them.
    
    def list_to_ranges(inList):
        outList = []
        left_elem = None
        last_elem = None
        for i in inList:
            if last_elem is None:
                last_elem = i
                left_elem = i
                continue
                
            if left_elem is None:
                left_elem = i
                continue
            
            if i == last_elem + 1:
                last_elem = i
                continue
            elif i > last_elem + 1:
                outList.append((left_elem, last_elem))
                left_elem = i
                last_elem = i
                continue
            
        return outList
            
    def flatten(list_of_lists):
        if list_of_lists == []:
            return []
        else:
            return [*list_of_lists[0], *flatten(list_of_lists[1:])]
    
    not_found = [frm for frm, frmDict in my_features.items() if frmDict['trackFlag'] == 'Not Found']
    contiguous = list_to_ranges(not_found)      
    to_retry = [list(range(start, end + 1)) for start, end in contiguous if end - start < redo_threshold]
    to_retry_flat = flatten(to_retry)
    
    return to_retry_flat

def redo_features_parallel(cap, frameList, verbose = True):
    general_homography_flagStr = 'Found (w. gen. H.)'
    
    totFrames = len(frameList)
    
    ### do the thorough function -> general homography finding.
    my_courtline_func = partial(track_courtlines, get_net = False, only_225 = False)    

    time_start = time.time()
    
    stream_features = {}
    H, config = None, None
    
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    while frameList != []:
        
        if len(pending) < threadn:
            frmNum, frameList = frameList[0], frameList[1:]
            cap.set(cv.CAP_PROP_POS_FRAMES, frmNum)
            _ret, frame = cap.read()
            
            cFrm = BGRFrame(frame)
            gFrm = cFrm.to_grey()
            
            task = pool.apply_async(my_courtline_func, (gFrm, None, None))

            pending.append((frmNum, gFrm, task))
            
        while len(pending) > 0 and pending[0][2].ready():
            num, frm, task = pending.popleft()
            courtLines, H, config, trackFlag = task.get()
            line_feature_dict = {'courtLines': courtLines, 
                                 'H': H, 
                                 'trackFlag': general_homography_flagStr}
            stream_features[num] = deepcopy(line_feature_dict)
            
            if verbose:
                showArr = frm.to_BGR().put(courtLines).array
                cv.imshow('threaded video', showArr)
            
        ch = cv.waitKey(1)
        if ch == 27:
            break
        
        if frmNum > stop_frame:
            break

    time_elapsed = time.time() - time_start
    minutes_elapsed = round(time_elapsed / 60, 1)
    print(f'Done with {totFrames} frames. Took {minutes_elapsed} minutes')

    cv.destroyAllWindows()
    return stream_features


redos = get_redo_frames(my_features)
redone_features = redo_features_parallel(cap, redos, verbose = False)

#### combine the newly found courtlines

combined_features = deepcopy(my_features)

for frameNum, featDict in redone_features.items():
    combined_features[frameNum] = deepcopy(featDict)
    
#### find contiguous passages of 'similar-enough' courtlines

def find_contiguous_play_sequences(stream_features, length_threshold = 25):
    outList = []
    current_first_frame = None
    for frameNum, featDict in stream_features.items():
        tF = featDict['trackFlag']
        if tF == 'Tracked':
            current_last_frame = frameNum
            continue
        if tF in ['Found', 'Found (w. gen. H.)']:
            if current_first_frame is None:
                current_first_frame = frameNum
                current_last_frame = frameNum
            else:
                ### here, we should maybe also check deviation from neighbours?
                current_last_frame = frameNum
        if tF == 'Not Found':
            if current_first_frame is not None:
                length_of_continuous_stretch = current_last_frame - current_first_frame
                if length_of_continuous_stretch > length_threshold:
                    outList.append((current_first_frame, current_last_frame))
                current_first_frame = None
                current_last_frame = None
    return outList

outList = find_contiguous_play_sequences(combined_features)


#### Write List to text ####
import pandas as pd
outFrm = pd.DataFrame(outList, columns = ['StartFrame', 'StopFrame'])

import os
ballDict = atu.get_folder_struct('ball_event_hmm')
savePath = os.path.join(ballDict['main'], VIDEO_ID, 'play_sequences')

try:
    os.mkdir(savePath)
except FileExistsError:
    print('   Adding to existing play_sequence data')

import datetime as dt
timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

filePath = os.path.join(savePath, f'play_start_stop_{timestamp}.csv')

outFrm.to_csv(filePath)   


    
#### check if the re-done feature lines deviate strongly from neighbouring lines.

do_replay = False

def draw_lines(frmDict, featDict):
    cFrm = frmDict['cFrm']
    
    tF = featDict['trackFlag']
    courtLines = featDict['courtLines']
    
    showFrm = cFrm.put(tF, color = (0, 0, 255), size = 2, thickness = 2)
    if courtLines is not None:
        showFrm = showFrm.put(courtLines, color = (0, 255, 0), thickness = 2)

    return showFrm


if do_replay:
    #### Replay the stretches of play
    from ATMANpy.funcs.cv_funcs import replay
    
    for idx, playTup in enumerate(outList):
        start, stop = playTup
        print(f'Sequence of Play: #{idx}. Length: {stop - start}')
        replay(cap, combined_features, start_frame = start, stop_frame = stop,
                drawFunc = draw_lines)


    
    
    
    
    
    

