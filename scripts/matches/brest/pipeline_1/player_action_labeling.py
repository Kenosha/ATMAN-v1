# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:26:01 2023

@author: gianl
"""


import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import join_video_captures

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)


if __name__ == '__main__':

    conf = ut.read_config('brest')
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']

    frameList = list(range(conf['start_frame'], conf['stop_frame'] + 1))
    
    ### Load Courtlines ###
    H = ConstFeatures.from_store(vid, sto, featnameList = ['H'], frameList = frameList).squeeze()
    courtLines = sa.find_courtlines(H.feature)
    base_length = courtLines['Bottom Baseline'].length() ### take the larger of the two (bot side) so that our maximum ratio is larger!
    side_length = courtLines['Bottom Singles Ad Line'].length() + courtLines['Top Singles Deuce Line'].length()
    
    ### Load Play Segmenter
    playBF = BoolFeatures.from_store(vid, sto, featnameList = ['isPlay']).squeeze()
    playBFs = playBF.filter(playBF).split()    
    
    ### Make Streams    
    myStms = StreamList.from_SFList(playBFs, cap)
    
    #%% ### Check Sequences and Add/Remove Candidate Contours as Needed
    
    serveDict = {1: 'begin bouncing',
                 2: 'stop bouncing', ### i.e. ball is now in throat of racket, and is then thrown.
                 3: 'ball leaves hand', ### i.e, complete throw, and enter trophy position
                 4: 'begin swinging', ## i.e. accelerating towards ball
                 5: 'impact',
                 6: 'end swing' ### both legs land back on ground (?)
                 }
    
    hitDict = {1: 'begin loading', ### i.e, charging legs, waiting for ball in fixed position
               2: 'begin swinging', ## i.e. accelerating towards ball
               3: 'impact',
               4: 'end swing' ### both legs land back on ground
               }
    
    otherDict = {1: 'Ready',
                  2: 'Running'}
    
    jointDict = {'serve': serveDict,
                 'hit': hitDict,
                 'other': otherDict}
    
    def record_events(myStm, which_player = 'top', which_event = 'hit',
                      zoom_factor = 3):
        ### Depending on whether which_event is Hit, Serve or Other...
            ### This function is ONLY for hit sequencing.
            # a) When the player begins loading the shot, charging legs, waiting for ball in fixed position
            # b) When the player begins swinging at the ball, up until contact
            # c) When the swing is wound down, until the player is in ready position or starts repositioning
            ### Also, forehand/backhand/volley/slice
            ### This function is ONLY for serve sequencing, i.e.
            # a) When ball is started being bounced 
            # b) When player enters trophy pose (begins at ball leaving hand)
            # c) When player enters swing (begins at upwards acceleration towards ball, until moment of hit)
            # d) When serve motion is wound down (from hit, until player is in ready position again)
            ### Also, the side from which the serve occurs (left or right) is recorded.
            ### This is for other events, i.e.
            # a) Ready Position
            # b) Walking 
            # c) Shuffling sideways
            # d) Running
            # e) Sliding
            # ...
            ####
        cap = myStm.cap
        start_frame, stop_frame = myStm.start_frame, myStm.stop_frame
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        
        eventUpper = which_event.upper()
        eventLower = which_event.lower()
        
        playerUpper = which_player.upper()
        playerLower = which_player.lower()
        
        break_outer = False
    
        recorded_event_dict = {}
        
        while cap.isOpened():
            frameNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            
            ### Create Base Frame
            showFrm = BGRFrame(frame).zoom(zoom_factor)
            
            ### Add Event Associated with Frame
            try:
                recorded_event = recorded_event_dict[frameNum]
            except KeyError:
                recorded_event = 'None'
            
            showFrm = showFrm.put(f'Recording {eventUpper} for {playerUpper}')
            showFrm = showFrm.put(f'Recorded: {recorded_event}', where = (50, 100))
            
            cv.imshow('frame', showFrm.array)
        
            ### Collect key
            keypress = cv.waitKey(0) 
        
            ### Sub Events
            subeventDict = jointDict[eventLower]
            try:
                num = int(chr(keypress))
            except:
                pass
            else:
                if num in subeventDict:
                    sub_event = subeventDict[num]
                    recorded_event_dict[frameNum] = eventLower + ' ' + sub_event
                    
            ### DELETE
            if keypress == 8: ### backspace key
                recorded_event_dict.pop(frameNum, None)
            
            ### GO AHEAD
            if keypress == ord(' '):
                go_to_frame = min(frameNum + 1, stop_frame) ### can't go farther
                                                                ## than start_frame
                cap.set(cv.CAP_PROP_POS_FRAMES, go_to_frame) 
                continue
            
            ### REWIND
            if keypress == ord('b'): 
                back_to_frame = max(start_frame, frameNum - 1) ### can't go farther
                                                                ## than start_frame
                cap.set(cv.CAP_PROP_POS_FRAMES, back_to_frame) 
                continue
            
            #### EXIT    
            if keypress == ord('q'): 
                break
            
            #### EXIT ALL
            if keypress == 27: ### ESC key!
                break_outer = True ### break the outer loop of all sequences
                break
        
            cap.set(cv.CAP_PROP_POS_FRAMES, frameNum) 
        
        # cap.release()
        cv.destroyAllWindows()
        
        evtSF = StreamFeature(recorded_event_dict, name = f'labelled_{playerLower}_{eventLower}_Flag')
        
        return evtSF, break_outer
        
    #######################
    #######################
    #######################
    
    WHICH_PLAYER = 'top'
    WHICH_EVENT = 'serve'
    ZOOM_FACTOR = 2 ## zoom before displaying
    
    readSF = StreamFeatures.from_store(vid, sto, featnameList = [f'labelled_{WHICH_PLAYER}_{WHICH_EVENT}_Flag']).squeeze()
    print(readSF.value_counts())
    
    ### check how many have been done...
    if readSF.is_empty():
        first_idx_not_done = 0
    else:
        last_frame, _ = readSF.last()
        last_indices_done = [idx for idx, myStm in enumerate(myStms) if myStm.start_frame < last_frame < myStm.stop_frame]
        try:
            first_idx_not_done = last_indices_done[0] + 1
        except:
            first_idx_not_done = 0
        
        
    ### Record Loop
    for myStm in myStms[first_idx_not_done:]:
        evtSF, breakBool = record_events(myStm, 
                                         which_player = WHICH_PLAYER, 
                                         which_event = WHICH_EVENT,
                                         zoom_factor = ZOOM_FACTOR)
        
        evtSF.save(vid, sto)
        
        if breakBool:
            break
        
    
    
    
    