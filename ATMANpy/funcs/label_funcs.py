# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 17:20:47 2023

@author: gianl
"""

import cv2 as cv
from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.Features import StreamFeature


def get_event_dict(event):
    ### For the point-wise, single moment events.
    
    ### ACTIONS TO RECORD
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
    
    otherDict = {1: 'begin standing to serve',
                 2: 'begin standing idle', 
                 3: 'begin standing ready', ### waiting in position for a shot
                 4: 'begin split stepping',
                 5: 'begin repositioning', ### shifting of feet in position, not for movement, but for unit turn, small adjustments, etc. 
                 6: 'begin shuffling',
                 7: 'begin walking',
                 8: 'begin running',
                 9: 'begin sliding',
                 0: 'end' ### usually, not necessary -> a move is ended by the beginning of a next move. However, sometimes, we can not categorize some frames between, so we just say this.
                 }
    
    jointDict = {'serve': serveDict,
                 'hit': hitDict,
                 'other': otherDict}
    ###############
    return jointDict[event]

def get_fill_dict(event):
    ### For the space between events    
    
    ### ACTIONS TO RECORD
    serveDict = {'begin bouncing': 'bouncing',
                 'stop bouncing': 'throwing', ### i.e. ball is now in throat of racket, and is then thrown.
                 'ball leaves hand': 'trophy position', ### i.e, complete throw, and enter trophy position
                 'begin swinging': 'swinging', ## i.e. accelerating towards ball
                 'impact': 'winding down'}
    
    hitDict = {'begin loading': 'loading', ### i.e, charging legs, waiting for ball in fixed position
               'begin swinging': 'swinging', ## i.e. accelerating towards ball
               'impact': 'winding down'}
    
    jointDict = {'serve': serveDict,
                 'hit': hitDict}
    
    return jointDict[event]

def get_dispensables(event):
    if event == 'serve':
        return []
    if event == 'hit':
        return ['begin loading']

def get_category_dict(event, category):
    if event == 'serve':
        if category == 'side':
            return {1: 'left', ### top -> deuce; bot -> ad
                    2: 'right' ### top -> ad; bot -> deuce
                    }
    if event == 'hit':
        if category == 'side':
            return {1: 'forehand',
                    2: 'backhand'}
        if category == 'spin':
            return {1: 'topspin',
                    2: 'slice',
                    3: 'flat'}
        if category == 'type':
            return {1: 'groundstroke',
                    2: 'volley',
                    3: 'overhead'}
    

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
    
    feature_name = f'labelled_{playerLower}_{eventLower}_Flag'
    
    break_outer = False

    try:
        previously_recorded_event_dict = myStm.features[feature_name].streamDict ###  (mostly, there will be none, so this will be empty)
    except:
        previously_recorded_event_dict = {}
    recorded_event_dict = dict(previously_recorded_event_dict) ### initialize with a COPY of pre-recorded features
    
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
        showFrm = showFrm.put(str(frameNum), where = (50, 150))
        
        cv.imshow('frame', showFrm.array)
    
        ### Collect key
        keypress = cv.waitKey(0) 
    
        ### Sub Events
        subeventDict = get_event_dict(eventLower)
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
    
    ### avoid duplicates -> write only what is necessary! -> changes, additions, deletions
    write_event_dict = {}
    frames_with_some_label = sorted(set(list(recorded_event_dict.keys()) + list(previously_recorded_event_dict.keys())))
    for frame in frames_with_some_label:
        try:
            rec_evt = recorded_event_dict[frame]
        except KeyError: ### deletion!
            write_event_dict[frame] = previously_recorded_event_dict[frame]
        
        try:
            prev_evt = previously_recorded_event_dict[frame]
        except KeyError: ### addition!
            write_event_dict[frame] = recorded_event_dict[frame]
    
        try:
            was_changed = rec_evt != prev_evt
        except NameError: ### handled above.
            pass
        else: ### event exists previously and curently
            if was_changed: ### changed!
                write_event_dict[frame] = rec_evt
            
    evtSF = StreamFeature(write_event_dict, name = feature_name)
    
    return evtSF, break_outer

def group_events(singleSF, event):
    
    eventDict = get_event_dict(event)
    rev_eventDict = {event + ' ' + v: k for k, v in eventDict.items()}
    streamDict_with_keys = {frame: rev_eventDict[eventStr] for frame, eventStr in singleSF.streamDict.items() if not eventStr == 'deletion'}
    
    groupList = []
    current_group = {}
    prev_event_key = 0
    prev_frame = 0
    for frame, event_key in streamDict_with_keys.items():
        is_adjacent = frame == prev_frame + 1 ### adjacent! -> can have same event again!
        is_same_event = event_key == prev_event_key
        is_next_event = event_key == prev_event_key + 1
        is_subsequent_event = event_key > prev_event_key
        
        if is_same_event and is_adjacent:
            current_group[frame] = event_key
        elif is_next_event:
            current_group[frame] = event_key
        elif is_subsequent_event:
            current_group[frame] = event_key
        else:
            groupList.append(current_group)
            current_group = {frame: event_key}
        
        prev_event_key = event_key
        prev_frame = frame
        
    return groupList
        
def check_group(groupDict, eventDict, dispensable_events = None):
    if dispensable_events is not None:
        needed_events = [evt for evt in eventDict.values() if evt not in dispensable_events]
    else:
        needed_events = list(eventDict.values())
    
    groupDict_as_event = {frame: eventDict[key] for frame, key in groupDict.items()}
    missing_events = [evt for evt in needed_events if not evt in groupDict_as_event.values()]
    return missing_events == []

def unambiguify_group(groupDict):
    ### takes in a group of events in which events may be ambiguous 
    ### (the impact can occur between frames, so is marked at two moments, for example)
    ### and turns them into a list of unambiguous, single event groups with weights attached to them.
    ### (the weights are simply one over the length of the list)
    from itertools import product
    
    ### first, construct a dictionary with events in the key and the list of frames they appear in
    def get_frames_of_event(groupDict, event):
        return [frame for frame in groupDict if groupDict[frame] == event]
    
    frameDict = {event: get_frames_of_event(groupDict, event) for event in groupDict.values()}
    
    list_of_groups = []
    current_group = {}
    for frame_combos in product(*frameDict.values()):
        for frame in frame_combos:
            current_group[frame] = groupDict[frame]
        list_of_groups.append(current_group)
        current_group = {}

    return list_of_groups


def fill_in_group(groupDict, fillDict):
    frames = sorted(groupDict.keys())
    fs, ls = frames[0], frames[-1]
    
    filled_groupDict = {}
    for frame in range(fs, ls): ### note how the last frame (which coincides with end swing) will not be iterated over!
        try:
            current_event = groupDict[frame] ### this will succeed in first iteration!
        except:
            pass ### in this case, previous current_event remains in this iteration
        
        current_filler = fillDict[current_event]
        
        filled_groupDict[frame] = current_filler
        
    filled_groupDict[frame] = current_filler ### add in last frame (not necessary, matter of convention. But in our case, the end swing event is last frame of "winding down")
        
    return filled_groupDict
    
def classify_event(myStm, player, event, category, zoom_factor = 2, fps = 15):
    
    cap = myStm.cap
    start, stop = myStm.start_frame, myStm.stop_frame
    catDict = get_category_dict(event, category)
    
    break_outer = False
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start)
    while cap.isOpened():
        
        frameNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        if frameNum > stop:
            cap.set(cv.CAP_PROP_POS_FRAMES, start)
        ret, frame = cap.read()
        
        ### Create Base Frame
        showFrm = BGRFrame(frame).zoom(zoom_factor)
        showFrm = showFrm.put(f'Classifying {player} {event} {category}')
        showFrm = showFrm.put(f'Options: {catDict}', where = (50, 100))

        cv.imshow('frame', showFrm.array)
    
        ### Collect key
        keypress = cv.waitKey(1000 // fps) 

        #### EXIT    
        if keypress == ord('q'): 
            break
        
        #### EXIT ALL
        if keypress == 27: ### ESC key!
            break_outer = True ### break the outer loop of all sequences
            break
        
        ### Check keypress against category dictionary
        try:
            num = int(chr(keypress))
        except:
            pass
        else:
            if num in catDict:
                cat_label = catDict[num]
                break
    
    try:
        outDict = {frame: cat_label for frame in range(start, stop + 1)}
        catSF = StreamFeature(outDict, name = f'labelled_{player}_{event}_{category}_Flag')
    except NameError:
        catSF = StreamFeature({})

    # cap.release()
    cv.destroyAllWindows()

    return catSF, break_outer


