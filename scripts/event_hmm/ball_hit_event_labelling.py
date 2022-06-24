# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:45:04 2022

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

# image operation
import cv2 as cv
import numpy as np

import pandas as pd

import os
from random import randrange

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################


########## READ-IN PASSAGES OF PLAY ##########

evtDict = atu.get_folder_struct('ball_event_hmm')

try:
    readPath = evtDict[VIDEO_ID]
except KeyError:
    print('Data not found.')
    
playseqPath = os.path.join(readPath, 'relabel_sequences')    

frmList = []
for file in os.listdir(playseqPath):
    frm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)
    frmList.append(frm)
    
seqFrm = pd.concat(frmList, ignore_index = True)

###############################################

######## UNDERSTAND WHAT HAS ALREADY BEEN DONE ######

import os
evtDict = atu.get_folder_struct('ball_event_hmm')
savePath = os.path.join(evtDict['main'], VIDEO_ID, 'labelled_sequences_corrections')

try:
    os.mkdir(savePath)
except FileExistsError:
    print('   Adding to existing labelled sequence data')

### Serve Events
sList = []
for file in os.listdir(savePath):
    if file.startswith('serve_events'):
        sFrm = pd.read_csv(os.path.join(savePath, file), index_col = 0)
        sList.append(sFrm)
    
### Hit Events
hList = []
for file in os.listdir(savePath):
    if file.startswith('hit_events'):
        hFrm = pd.read_csv(os.path.join(savePath, file), index_col = 0)
        hList.append(hFrm)
    
if not sList == []:
    sFrm = pd.concat(sList, ignore_index = True)
    serves_done = sFrm[['StartFrame', 'StopFrame']].drop_duplicates()
    serves_done = list(serves_done.itertuples(index=False, name=None))
else:
    serves_done = []

if not hList == []:
    hFrm = pd.concat(hList, ignore_index = True)
    hits_done = hFrm[['StartFrame', 'StopFrame']].drop_duplicates()
    hits_done = list(hits_done.itertuples(index=False, name=None))
else:
    hits_done = []

print(f'Serves: Done {len(serves_done)} already!')
print(f'Hits: Done {len(hits_done)} already!')

# ###### PRE-SELECT PASSAGES OF PLAY ############

# #### Replay the stretches of play
# from ATMANpy.funcs.cv_funcs import show_video

# for idx, playTup in seqFrm.iterrows():
#     start, stop = playTup
#     print(f'Sequence of Play: #{idx}. Length: {stop - start}')
#     show_video(cap, start_frame = start, stop_frame = stop)

######## LABEL A PASSAGE OF PLAY #######

#%% FUNCTIONS ####


##### TODO:
    
### separate record_serve and record_hit function.
# with record serve, 
    #you record first serve and subsequent bounce, and second serve and bounce
# witrh record hit,
        #you record all hits and bounces after

# the reason is that event recording is much simpler then
# -> serve bounces are just in or out or net; the serve area can be inferred.

# with hits, there is no need to record sides at all.
# only near/far and in/out/net

##### AUXILIARIES ######

eventDict = {'SFR': 'Serve by Far player, Right side',
             'SFL': 'Serve by Far player, Left side',
             'SNR': 'Serve by Near player, Right side',
             'SNL': 'Serve by Near player, Left side',
             'BIF': 'Bounce Inside Far player half court',
             'BOF': 'Bounce Outside Far player half court',
             'BIN': 'Bounce Inside Near player half court',
             'BON': 'Bounce Outside Near player half court',
             'HF':  'Hit by Far player',
             'HN':  'Hit by Near player',
             'BIFSR': 'Bounce Inside Far player Serve area on the Right',
             'BIFSL': 'Bounce Inside Far player Serve area on the Left',
             'BOFS': 'Bounce Outside of Far player Serve area',
             'BINSR': 'Bounce Inside Near player Serve area on the Right',
             'BINSL': 'Bounce Inside Near player Serve area on the Left',
             'BONS': 'Bounce Outside of Near player Serve area',
             'NET': 'Bounce on NET',
             'CORD': 'Bounce on NET CORD'} ### new!

# assert 1 == 2, """
#                 Man! I added the Net Cord as an event. 
#                 This way, NET always ends the point. 
#                 Work this into the encodings below, if needed!
#               """

def invert_binary_symbol(symbol):
    if symbol == 'N': ### near -> far
        return 'F'          
    if symbol == 'F': ### far -> near
        return 'N'
    if symbol == 'L': ### left -> right
        return 'R'
    if symbol == 'R':
        return 'L'    

########################

### SERVE FUNCTIONS ####

def serve_toggle(keypress,
                 typeStr = None, goodStr = None, 
                 playStr = None, sideStr = None):

    ### TOGGLE EVENT TYPE
    if keypress == ord('1'):
        typeStr = 'S' ### toggle serve
    if keypress == ord('2'):
        typeStr = 'B' ### toggle bounce
    
    if keypress == ord('w'):
        playStr = 'F'
    if keypress == ord('s'):
        playStr = 'N'
    
    if keypress == ord('a'):
        sideStr = 'L'
    if keypress == ord('d'):
        sideStr = 'R'
        
    if keypress == ord('z'):
        goodStr = 'I'
    if keypress == ord('x'):
        goodStr = 'O'
    if keypress == ord('c'):
        goodStr = 'N'
    if keypress == ord('v'):
        goodStr = 'C'
    
    return typeStr, goodStr, playStr, sideStr


def serve_encode(typeStr, goodStr, playStr, sideStr, 
                 state_variable = None): ### this allows to feed in a value 
                                           ### that makes labelling easier. 
                                           ### here, we save the first serve,
                                           ### every other subsequent event
                                           ### can be deduced from it.
                                           ### (i.e: bounce is opposite,
                                           ### second serve is the same.)
    if typeStr == 'S':
        if state_variable is None:
            try:
                event_code = typeStr + playStr + sideStr
            except:
                print('Please assign all necessary values for Serve!')
                return None
            else:
                return event_code
        else:
            return state_variable
    
    if typeStr == 'B':
        if goodStr == 'I':
            if state_variable is None:
                try:
                    event_code = typeStr + goodStr + playStr + 'S' + sideStr
                except:
                     print('Please assign all necessary values for Bounce!')
                     return None
                else:
                     return event_code
            else:
                _, serveplayStr, servesideStr = state_variable # SFR, for instance.
                playStr = invert_binary_symbol(serveplayStr)
                sideStr = invert_binary_symbol(servesideStr)
                event_code = typeStr + goodStr + playStr + 'S' + sideStr
                return event_code
        
        if goodStr == 'O':
            if state_variable is None:
                try:
                    event_code = typeStr + goodStr + playStr + 'S'
                except:
                     print('Please assign all necessary values for Bounce!')
                     return None
                else:
                     return event_code
            else:
                _, serveplayStr, servesideStr = state_variable # SFR, for instance.
                playStr = invert_binary_symbol(serveplayStr)
                event_code = typeStr + goodStr + playStr + 'S'
                return event_code
            
        if goodStr == 'N':
            return 'NET'
        
        if goodStr == 'C':
            return 'CORD'
     
    
def serve_record_state(event_code, current_state):
    if event_code is not None:
        if event_code[0] == 'S':
            return event_code
        else:
            return current_state
    else:
        return current_state

#################################


####### HIT FUNCTIONS ###########

def hit_toggle(keypress,
               typeStr = None, goodStr = None, 
               playStr = None, sideStr = None):

    ### TOGGLE EVENT TYPE
    if keypress == ord('1'):
        typeStr = 'H' ### toggle serve
    if keypress == ord('2'):
        typeStr = 'B' ### toggle bounce
    
    if keypress == ord('w'):
        playStr = 'F'
    if keypress == ord('s'):
        playStr = 'N'
    
    if keypress == ord('z'):
        goodStr = 'I'
    if keypress == ord('x'):
        goodStr = 'O'
    if keypress == ord('c'):
        goodStr = 'N'
    if keypress == ord('v'):
        goodStr = 'C'
    
    return typeStr, goodStr, playStr, sideStr


def hit_encode(typeStr, goodStr, playStr, sideStr,
               state_variable = None):
    
    if typeStr == 'H':
        try:
            event_code = typeStr + playStr
        except:
            print('Please assign all necessary values for hit!')
            return None
        else:
            return event_code

    if typeStr == 'B':
        if goodStr in ['I', 'O']:
            try:
                event_code = typeStr + goodStr + playStr
            except:
                 print('Please assign all necessary values for bounce!')
                 return None
            else:
                 return event_code
        if goodStr == 'N':
            return 'NET'
        if goodStr == 'C':
            return 'CORD'
        
def put_event_string(frame, event_code, rec_text = None):
    if event_code is not None:
        eventName = eventDict[event_code]
    else:
        eventName = 'No Event Toggled'
        
    if 'Outside' in eventName or 'NET' in eventName:
        textColor = (34, 34, 255) ### Red!
    elif 'Inside' in eventName: 
        textColor = (0, 255, 124) ### Green!
    else:
        textColor = (0,153,153) ### Yellow!
    
    if 'Left' in eventName:
        text_x = 30
    elif 'Right' in eventName:
        text_x = 700
    else:
        text_x = 500
    
    if 'Near' in eventName:
        text_y = 1000
    elif 'Far' in eventName:
        text_y = 50
    else:
        text_y = 500
    
    textLoc = (text_x, text_y)
    
    text = f'Recording: {eventName}'
    
    cv.putText(frame, text, org = textLoc, 
               fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1.2,
               color = textColor, thickness = 3)
    
    if rec_text is not None: 
        textLoc_below = (text_x, text_y + 40)
        cv.putText(frame, rec_text, org = textLoc_below, 
               fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1.2,
               color = textColor, thickness = 3)
      

def record_events(cap, start_frame = None, stop_frame = None,
                  rec_stateFunc = None, toggleFunc = None, encodeFunc = None,
                  no_event_sentinel = None):
    ### Idea: record HF and HN simultaneously, with different keys.
    ### Record only apex of hit, but can be two frames if ambiguous
    typeStr = None ### s,h,b for serve, hit and bounce resp.
    sideStr = None ### l,r for left and right resp.
    playStr = None ### n,f for near and far resp.
    goodStr = None ### i,o,n for in, out and net resp.
    
    event_code = None
    state_variable = None

    breakBool = False ### sentinel value sent to the outside -> break out of program

    recorded_event_dict = {}
    
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frameNum = start_frame
    
    while cap.isOpened() and (start_frame <= frameNum <= stop_frame):
        frameNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        
        if ret:
            try:
                curr_text = recorded_event_dict[frameNum]
            except KeyError:
                curr_text = None
            
            put_event_string(frame, event_code, rec_text = curr_text)
            cv.imshow('frame', frame)
            
            keypress = cv.waitKey(0) 
        
            #### EXIT ALL
            if keypress == ord('0'):
                breakBool = True ### break the outer loop of all sequences
                break
        
            #### EXIT    
            if keypress == 27: ### ESC key!
                break
            
            ### GO AHEAD
            if keypress == ord(' '):
                continue
            
            ### REWIND
            if keypress == ord('b'): 
                back_to_frame = max(start_frame, frameNum - 1) ### can't go farther
                                                                ## then start_frame
                cap.set(cv.CAP_PROP_POS_FRAMES, back_to_frame) 
                continue
            
            ### RECORD
            if keypress == 13: ### return key
                if not event_code is None:
                    recorded_event_dict[frameNum] = event_code
                else:
                    print('None event can not be recorded.')
                
                ### deal with state variable 
                ### (a variable we want to remember internally, 
                ### like which side the serve was on)
                if not rec_stateFunc is None:
                    state_variable = rec_stateFunc(event_code, state_variable)
                    
            ### DELETE
            if keypress == 8: ### backspace key
                recorded_event_dict.pop(frameNum, None)
             
            ### DEAL WITH TOGGLES
            typeStr, goodStr, playStr, sideStr = toggleFunc(keypress, 
                                                            typeStr = typeStr, 
                                                            goodStr = goodStr, 
                                                            playStr = playStr, 
                                                            sideStr = sideStr)
            event_code = encodeFunc(typeStr, goodStr, playStr, sideStr,
                                    state_variable = state_variable)
            cap.set(cv.CAP_PROP_POS_FRAMES, frameNum) 

            
            
        else:
            break
        
    else: ### if the while loop ends, we record a possible no hit
            # this is so we can get out of going through all these things twice.
            # (bob dylan)
            # (this is because the read-in of prior label sessions does not include
            # empty event sets for a play sequence, and therefore plays it again.)
        if not no_event_sentinel is None:
            if recorded_event_dict == {}:
                recorded_event_dict[frameNum] = no_event_sentinel
        
    # cap.release()
    cv.destroyAllWindows()
                        
    
                         

    return breakBool, recorded_event_dict 

#%% MAIN EVENT LOOP #####

label_serves = True
label_hits = True

##### SERVE EVENT LOOP #####

if label_serves:
    
    print('Doing SERVES now....')
    
    serveList = []
    for idx, frame_range in seqFrm.iterrows():
        
        frame_tuple = tuple(frame_range)
        if frame_tuple in serves_done:
            continue
     
        start_frame, stop_frame = frame_tuple
        
        breakBool, outDict = record_events(cap, 
                                           start_frame = start_frame, 
                                           stop_frame = stop_frame,
                                           rec_stateFunc = serve_record_state,
                                           toggleFunc = serve_toggle,
                                           encodeFunc = serve_encode)
        
        outFrm = pd.Series(outDict).reset_index()
        outFrm.columns = ['FrameNum', 'Event']
        outFrm['Index'] = idx
        outFrm['StartFrame'] = start_frame
        outFrm['StopFrame'] = stop_frame 
        
        serveList.append(outFrm)
        
        if breakBool:
            break
    
    if len(serveList) > 0:
        serveFrm = pd.concat(serveList, ignore_index = True)    
    else:
        serveFrm = None
else:
    serveFrm = None  

##### HIT EVENT LOOP #####
  
if label_hits:
    
    print('Doing HITS now....')
     
    hitList = []
    for idx, frame_range in seqFrm.iterrows():
            
        frame_tuple = tuple(frame_range)
        if frame_tuple in hits_done:
            continue
     
        start_frame, stop_frame = frame_tuple
        
        breakBool, outDict = record_events(cap, 
                                           start_frame = start_frame, 
                                           stop_frame = stop_frame,
                                           rec_stateFunc = None,
                                           toggleFunc = hit_toggle,
                                           encodeFunc = hit_encode,
                                           no_event_sentinel = 'NOHIT')
        
        outFrm = pd.Series(outDict).reset_index()
        outFrm.columns = ['FrameNum', 'Event']
        outFrm['Index'] = idx
        outFrm['StartFrame'] = start_frame
        outFrm['StopFrame'] = stop_frame 
        
        hitList.append(outFrm)
        
        if breakBool:
            break

    if len(hitList) > 0:
        hitFrm = pd.concat(hitList, ignore_index = True)    
    else:
        hitFrm = None
else:
    hitFrm = []
                      
             


##### CONCATENATE AND SAVE ######0

# # since we're relabelling things, and our infrastructure is not optimal...
# ad_hoc_index_add = 136
# serveFrm['Index'] += ad_hoc_index_add
# hitFrm['Index'] += ad_hoc_index_add

import os
evtDict = atu.get_folder_struct('ball_event_hmm')
savePath = os.path.join(evtDict['main'], VIDEO_ID, 'labelled_sequences_corrections')

import datetime as dt
timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
servePath = os.path.join(savePath, f'serve_events_{timestamp}.csv')
hitPath = os.path.join(savePath, f'hit_events_{timestamp}.csv')

if not serveFrm is None:
    serveFrm.to_csv(servePath)    
                   
if not hitFrm is None:
    hitFrm.to_csv(hitPath)
        

######## TEST KEYS ########

# ret, img = cap.read()
# while(1):
#     cv.ims                      how('img',img)
#     k = cv.waitKey(33)
#     if k==27:    # Esc key to stop
#         break
#     elif k==-1:  # normally -1 returned,so don't print it
#         continue
#     else:
#         print(k) # else print its value






 
 














