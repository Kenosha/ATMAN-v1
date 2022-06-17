# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:27:14 2022

Idea is to detect ballboys well enough to be able to interrupt the play sequence if they move.

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



######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)


#%% Reading these features back in

ACCEPTABLE_LENGTH_OF_SEGMENT = 25

from ATMANpy.funcs.feature_funcs import load_stored_features, split_joint_features
from ATMANpy.funcs.cv_funcs import replay
from ATMANpy.funcs.high_level_funcs import smart_interpolate_ball_contours

# featnameList = ['ball', 'playerTop', 'playerBot', 'H']
stream_features_original = load_stored_features(VIDEO_ID, featnameList = None, keep_which = 'last')

### add a detected flag
for k, featDict in stream_features_original.items():
    if featDict['ball'] is not None:
        featDict['ballDetected'] = True
    else:
        featDict['ballDetected'] = False
    
origList = split_joint_features(stream_features_original, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

print(f'Before splitting ballboy clutter: {len(origList)} segments')

#%% Split features according to play sequences created earlier

########## READ-IN PASSAGES OF PLAY ##########
evtDict = atu.get_folder_struct('ball_event_hmm')

try:
    readPath = evtDict[VIDEO_ID]
except KeyError:
    print('Data not found.')
    
playseqPath = os.path.join(readPath, 'play_sequences')    
start_stop_timestamp = '2022-06-11-02-42-30'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)
##############

from ATMANpy.funcs.feature_funcs import filter_features_by_play_sequence_frame
stream_features_decluttered = filter_features_by_play_sequence_frame(stream_features_original, seqFrm)        
declList = split_joint_features(stream_features_decluttered, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)

print(f'After splitting ballboy clutter: {len(declList)} segments')


#%% Testing Decluttering versus Original Frames

    
def stream_features_add_splitbool(orig_sf, split_sfs):
    
    """
    takes an original stream feature that has been split, 
    and compares it with a list of split features. 
    It then marks each frame with a boolean denoting whether it found a correspondence 
    in the split frames.
    This boolean can then be used to brighten or darken frames in the replay, 
    indicating that a frame was kept or left our respectively.
    """

    out_sf = deepcopy(orig_sf)
    
    all_keys = [frmNum for sf in split_sfs for frmNum in sf]
    
    for k, featDict in out_sf.items():
        if k in all_keys:
            featDict['isPlay'] = True
        else:
            featDict['isPlay'] = False
            
    return out_sf
    
def sequence_bools(boolList):
    b, bs = boolList[0], boolList[1:]
    bool_seq = [b]
    for myBool in boolList:
        if myBool != bool_seq[-1]:
            bool_seq.append(myBool)
    return bool_seq


compList = [stream_features_add_splitbool(sf, declList) for sf in origList]

def make_addstr(sf, idx = None):

    ### number of del frames ###    
    boolList = [fD['isPlay'] for fD in sf.values()]
    number_of_frames_deleted = len(boolList) - sum(boolList)
    delStr = f'Del. Frms.: {number_of_frames_deleted}'

    ### number of segments
    distinct_sequences = sequence_bools(boolList)
    seqStr = f'Seqs.: {sum(distinct_sequences)}'

    if not idx is None:
        addStr = str(idx) + ': ' + delStr + '  ' + seqStr
    else:
        addStr = delStr + '  ' + seqStr
    
    return addStr

# for idx, sf in enumerate(compList):
#     addStr = make_addstr(sf)
#     drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'isPlay']
#     drawFunc = draw(drawList, add_text = addStr)
#     replay(cap, sf, drawFunc = drawFunc)
        


#%% Problematic Cases: Analysis.

def show_sf(sf, splits, ms_between_frames = 25):
    print(f'Number of segments: {len(splits)}')
    
    addStr = make_addstr(orig_sf)
    drawList = ['frmNum', 'courtLines', 'ball', 'playerTop', 'playerBot', 'isPlay']
    drawFunc = draw(drawList, add_text = addStr)
    replay(cap, orig_sf, drawFunc = drawFunc, ms_between_frames = ms_between_frames)
    
    
def get_matching_split_sfs(orig_sf, split_sfs):
    outList = []
    
    ks = list(orig_sf.keys())
    start_frame, end_frame = min(ks), max(ks)
    
    for idx, sf in enumerate(split_sfs):
        ks_split = list(sf.keys())
        split_start, split_end = min(ks_split), max(ks_split)
        
        if split_end < start_frame: ### all sfs before our sf of interest
            continue
        elif split_start > end_frame:
            continue
        else:
            outList.append(sf)
    
    return outList
    

drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'isPlay']
drawFunc = draw(drawList)

### 1) 
sfNum = 8
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits)

### -> ISSUE: segment with ballkids not entirely deleted.
problSeq = splits[1]
print(len(problSeq))


### 2) 
sfNum = 16
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits)


### -> ISSUE: second segment is a bit short. 
### but that might be unavoidable, since it IS a play segment. 

end_prev = max(splits[1].keys())
beg_next = min(compList[sfNum + 1].keys())

chkFrame = (end_prev + beg_next) // 2

BGRFrame.from_video(cap, chkFrame).show() ### indeed, there is a cut!

len(splits[1])

### 3) 
sfNum = 19
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits)

replay(cap, splits[1], drawFunc = drawFunc)

### ISSUE: scoreboard is taken for player...

# let's see if the clutter that causes dropping of frames is from player

### b) SPLIT THE FEATURE ACCORDING TO CLUTTER MOVEMENT
from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
move_sf = detect_movement_contours(orig_sf) 

drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours']
drawFunc = draw(drawList)
replay(cap, move_sf, drawFunc = drawFunc, ms_between_frames = 40)

from ATMANpy.funcs.sf_postprocess_funcs import calculate_contiguous_movement
cont_sf = calculate_contiguous_movement(move_sf)
drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
drawFunc = draw(drawList)
replay(cap, cont_sf, drawFunc = drawFunc, ms_between_frames = 40)

### -> the clutter indeed comes from the player shape!!!


### 3) 
sfNum = 20
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)

### -> player is taken for patch on the ground...

### is the clutter caused by player shape?

### b) SPLIT THE FEATURE ACCORDING TO CLUTTER MOVEMENT
from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
move_sf = detect_movement_contours(orig_sf) 

drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours']
drawFunc = draw(drawList)
replay(cap, move_sf, drawFunc = drawFunc, ms_between_frames = 40)

from ATMANpy.funcs.sf_postprocess_funcs import calculate_contiguous_movement
cont_sf = calculate_contiguous_movement(move_sf)
drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
drawFunc = draw(drawList)
replay(cap, cont_sf, drawFunc = drawFunc, ms_between_frames = 40)

### -> the clutter indeed comes from the player shape!!!

### 4) 
sfNum = 26
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)

### what the fuck is up with player detection?

drawList = ['ball', 'playerTop', 'playerBot', 'isPlay']
drawFunc = draw(drawList)

import ATMANpy.funcs.high_level_funcs as hlf
from ATMANpy.funcs.court_funcs import find_courtlines

playerTop, playerBot = None, None

ks = list(orig_sf.keys())
for k in ks[:10]:
    cFrm = BGRFrame.from_video(cap, k)
    hFrm = HSVFrame.from_video(cap, k)
    
    fD = orig_sf[k]
    frmDict = {'cFrm': cFrm}
    
    # sendFrm = drawFunc(frmDict, fD)
    # sendFrm.show()
    
    H = fD['H']
    courtLines = find_courtlines(H, schemaLines)
    
    new_fD = deepcopy(fD)
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    new_fD['roiMask_bot'], new_fD['roiMask_top'] = roiMask_bot, roiMask_top
    
    ### Players with adaptive ROIs.
    playerTop, playerBot = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True,
                                                       playerTop_prev = playerTop, playerBot_prev = playerBot,
                                                       roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    
    new_fD['playerTop'], new_fD['playerBot'] = playerTop, playerBot
    
    cFrm.mask(roiMask_bot).show()
    
    sendFrm = drawFunc(frmDict, new_fD)
    sendFrm.show()
    
    
orig_sf[list(orig_sf.keys())[0]]

### possible solutions: 
    # -> max area for player shape

### 4) 
sfNum = 31
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)


playerTop, playerBot = None, None

ks = list(orig_sf.keys())
for k in ks[:1]:
    cFrm = BGRFrame.from_video(cap, k)
    hFrm = HSVFrame.from_video(cap, k)
    
    fD = orig_sf[k]
    frmDict = {'cFrm': cFrm}
    
    # sendFrm = drawFunc(frmDict, fD)
    # sendFrm.show()
    
    H = fD['H']
    courtLines = find_courtlines(H, schemaLines)
    
    new_fD = deepcopy(fD)
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    new_fD['roiMask_bot'], new_fD['roiMask_top'] = roiMask_bot, roiMask_top
    
    ### Players with adaptive ROIs.
    playerTop, playerBot = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True, 
                                                       verbose = True,
                                                       playerTop_prev = playerTop, playerBot_prev = playerBot,
                                                       roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    
    new_fD['playerTop'], new_fD['playerBot'] = playerTop, playerBot
    
    cFrm.mask(roiMask_top).show()
    
    sendFrm = drawFunc(frmDict, new_fD)
    sendFrm.show()


### possible solutions: 
    # -> don't take contour with largest size, but contour that best conforms to
    # a rectangle that is taller than wide -> this is typically what the player looks like!
    ### (alternatively, we could train an svm...)
    ### svm features:
        # height, width
        # shape?
        # colour?
    # player colour jumps out more! look at the verbose find_both_players_smart
    

### 4) 
sfNum = 57
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)


playerTop, playerBot = None, None

ks = list(orig_sf.keys())
for k in ks[:17]:
    cFrm = BGRFrame.from_video(cap, k)
    hFrm = HSVFrame.from_video(cap, k)
    
    fD = orig_sf[k]
    frmDict = {'cFrm': cFrm}
    
    # sendFrm = drawFunc(frmDict, fD)
    # sendFrm.show()
    
    H = fD['H']
    courtLines = find_courtlines(H, schemaLines)
    
    new_fD = deepcopy(fD)
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    new_fD['roiMask_bot'], new_fD['roiMask_top'] = roiMask_bot, roiMask_top
    
    ### Players with adaptive ROIs.
    playerTop, playerBot = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True, 
                                                       verbose = False,
                                                       playerTop_prev = playerTop, playerBot_prev = playerBot,
                                                       roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    
    new_fD['playerTop'], new_fD['playerBot'] = playerTop, playerBot
    
    cFrm.mask(roiMask_top).show()
    
    sendFrm = drawFunc(frmDict, new_fD)
    sendFrm.show()
    
##### HAHAHA... there is a huge foot in the transition... roger's foot.
## since the foot is so large, it extends the ROI all over the screen.
## what if we limit the player ROI to only the half court? is this enough?

### -> yes it works! but we need to have a lower limit of player size!

### 4) 
sfNum = 63
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)

### remove the scoreboard, and a little bit around it... 
# the problem is a little pop-up saying that rog has two set points.

### -> could we maybe have an automatic score board detector?
# using edges, color?

### 4) 
sfNum = 67
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)

drawList = ['ball', 'playerTop', 'playerBot', 'isPlay']
drawFunc = draw(drawList)

import ATMANpy.funcs.high_level_funcs as hlf
from ATMANpy.funcs.court_funcs import find_courtlines


playerTop, playerBot = None, None

ks = list(orig_sf.keys())
for k in ks[40:100]:
    cFrm = BGRFrame.from_video(cap, k)
    hFrm = HSVFrame.from_video(cap, k)
    
    fD = orig_sf[k]
    frmDict = {'cFrm': cFrm}
    
    # sendFrm = drawFunc(frmDict, fD)
    # sendFrm.show()
    
    H = fD['H']
    courtLines = find_courtlines(H, schemaLines)
    
    new_fD = deepcopy(fD)
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    new_fD['roiMask_bot'], new_fD['roiMask_top'] = roiMask_bot, roiMask_top
    
    ### Players with adaptive ROIs.
    playerTop, playerBot = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True, 
                                                       verbose = False,
                                                       playerTop_prev = playerTop, playerBot_prev = playerBot,
                                                       roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    
    new_fD['playerTop'], new_fD['playerBot'] = playerTop, playerBot
    
    cFrm.mask(roiMask_top).show()
    
    sendFrm = drawFunc(frmDict, new_fD)
    sendFrm.show()

    
### zooming in on play sequences really messes with the burn in... 
# highly specific to a single homography.
# need to work on this long term.
### the good news is that the halfcourt roi implemented earlier contains the problem until
### the camera has zoomed in back to the known homography. additionally, we could impose
# a maximum player size, as thought about before.
# issue with the maximum player size: 
    ## if we impose it, and the player has already left the service_ROI, then defaulting to 
    ## the service ROI is bad!
    
    ### again, conforming to a perfect, optimal-ratio player rectangle could be the solution!
    


### 4) 
sfNum = 86
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)

## issue: scoreboard



### 4) 
sfNum = 127
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)

### issue -> initial scoreboard graphic.

playerTop, playerBot = None, None

ks = list(orig_sf.keys())
for k in ks[0:20]:
    cFrm = BGRFrame.from_video(cap, k)
    hFrm = HSVFrame.from_video(cap, k)
    
    fD = orig_sf[k]
    frmDict = {'cFrm': cFrm}
    
    # sendFrm = drawFunc(frmDict, fD)
    # sendFrm.show()
    
    H = fD['H']
    courtLines = find_courtlines(H, schemaLines)
    
    new_fD = deepcopy(fD)
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    new_fD['roiMask_bot'], new_fD['roiMask_top'] = roiMask_bot, roiMask_top
    
    ### Players with adaptive ROIs.
    playerTop, playerBot = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True, 
                                                       verbose = True,
                                                       playerTop_prev = playerTop, playerBot_prev = playerBot,
                                                       roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    
    new_fD['playerTop'], new_fD['playerBot'] = playerTop, playerBot
    
    cFrm.mask(roiMask_bot).show()
    
    sendFrm = drawFunc(frmDict, new_fD)
    sendFrm.show()



########## BOT PLAYER REGION IS MESSED UP
## because region in schema is too large, and overflows...
# therefore, make a simple line (netline / netline+overlap) and cut off everything above/below.
# to this tomorrow, first thing.



### 4) 
sfNum = 128
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
show_sf(orig_sf, splits, ms_between_frames = 40)


drawList = ['ball', 'playerTop', 'playerBot', 'isPlay']
drawFunc = draw(drawList)

import ATMANpy.funcs.high_level_funcs as hlf
from ATMANpy.funcs.court_funcs import find_courtlines

playerTop, playerBot = None, None

ks = list(orig_sf.keys())
for k in ks[0:20]:
    cFrm = BGRFrame.from_video(cap, k)
    hFrm = HSVFrame.from_video(cap, k)
    
    fD = orig_sf[k]
    frmDict = {'cFrm': cFrm}
    
    # sendFrm = drawFunc(frmDict, fD)
    # sendFrm.show()
    
    H = fD['H']
    courtLines = find_courtlines(H, schemaLines)
    
    new_fD = deepcopy(fD)
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    new_fD['roiMask_bot'], new_fD['roiMask_top'] = roiMask_bot, roiMask_top
    
    ### Players with adaptive ROIs.
    playerTop, playerBot = hlf.find_both_players_smart(hFrm, courtLines, H, remove_shadow = True, 
                                                       verbose = False,
                                                       iterative_burn_in = True,
                                                       playerTop_prev = playerTop, playerBot_prev = playerBot,
                                                       roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    
    new_fD['playerTop'], new_fD['playerBot'] = playerTop, playerBot
    
    cFrm.mask(roiMask_bot).show()
    
    sendFrm = drawFunc(frmDict, new_fD)
    sendFrm.show()




### 4) 
sfNum = 131
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
# show_sf(orig_sf, splits, ms_between_frames = 40)

orig_sf.keys()


### 4) 
sfNum = 132
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
# show_sf(orig_sf, splits, ms_between_frames = 40)

orig_sf.keys()

### 4) 
sfNum = 133
orig_sf = compList[sfNum]
splits = get_matching_split_sfs(orig_sf, declList)
# show_sf(orig_sf, splits, ms_between_frames = 40)

orig_sf.keys()


#### -> these three are simply missing two/three frames between each other!


cFrm = BGRFrame.from_video(cap, 95413)
gFrm = GreyFrame.from_video(cap, 95143)


import ATMANpy.funcs.high_level_funcs as hlf
courtLines, netLines, H, config, outFlag = hlf.track_courtlines(gFrm, None, None)

cFrm.put(courtLines).show()

"""


# for sf in corrList:

sf = corrList[7]

sfList = get_matching_split_sfs(sf, split_sfs)
    
drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
drawFunc = draw(drawList, add_text = 'Original Sequence')
replay(cap, sf, drawFunc = drawFunc)

for idx, sf in enumerate(sfList):
    drawFunc = draw(drawList, add_text = f'Split Sequence: {idx}')
    replay(cap, sf, drawFunc = drawFunc)

    
len(split_sfs)

idx


#%% Split the loaded stream features if there is movement in frame.

########################
##### FOR TESTING ######
########################

### a) CHOOSE A STREAM FEATURE
sf_num = 12
corr_sf = corrList[sf_num]

### b) SPLIT THE FEATURE ACCORDING TO CLUTTER MOVEMENT
from ATMANpy.funcs.sf_postprocess_funcs import detect_movement_contours
move_sf = detect_movement_contours(corr_sf) 

drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours']
drawFunc = draw(drawList)
# replay(cap, move_sf, drawFunc = drawFunc)

from ATMANpy.funcs.sf_postprocess_funcs import calculate_contiguous_movement
cont_sf = calculate_contiguous_movement(move_sf)
drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'movInt']
drawFunc = draw(drawList)
# replay(cap, cont_sf, drawFunc = drawFunc)

from ATMANpy.funcs.sf_postprocess_funcs import drop_movement_frames
drop_sf = drop_movement_frames(cont_sf)

from ATMANpy.funcs.feature_funcs import split_joint_features
ACCEPTABLE_LENGTH_OF_SEGMENT = 25
sfList = split_joint_features(drop_sf, min_feature_length = ACCEPTABLE_LENGTH_OF_SEGMENT)
        
### c) TEST THE RESULTS
frames_before = len(corr_sf)
frames_after = sum([len(sf) for sf in sfList])
frames_killed = frames_before - frames_after 

print(f'Feature Number: {sf_num}')
print(f'  Number of segments created: {len(sfList)}')
print(f'  Number of frames killed: {frames_killed}')

for sf in sfList:
    replay(cap, sf, drawFunc = drawFunc, ms_between_frames = 40)
    
    
#### Results 

sf_num = 8
corr_sf = corrList[sf_num]
# -> we get three segments, with one being a clutter segment of length 42 ### len(sfList[1])
    

sf_num = 9
corr_sf = corrList[sf_num]
# -> weird 'breathing' clutter detected at picture's edge.
### line detection for this one is systematically bad! can not be normalized away!


drawList = ['frmNum', 'ball', 'playerTop', 'playerBot', 'movContours', 'courtLines']
drawFunc = draw(drawList)
replay(cap, move_sf, drawFunc = drawFunc, ms_between_frames = 40)
        
"""
        
        