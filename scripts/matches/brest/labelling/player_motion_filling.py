# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:39:11 2023

@author: gianl
"""


#### READ IN ######
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Features import BoolFeature, BoolFeatures, StreamFeature, StreamFeatures
from ATMANpy.types.Streams import Stream, StreamList

from ATMANpy.funcs.label_funcs import get_event_dict, get_fill_dict, get_dispensables
from ATMANpy.funcs.label_funcs import record_events, group_events
from ATMANpy.funcs.label_funcs import check_group, unambiguify_group, fill_in_group

vidList = ['2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa',
           '2021-7341-ms002-brest-brandon-nakashima-vs-alexandre-muller',
           '2021-7341-ms003-brest-joao-sousa-vs-maxime-janvier',
           '2021-7341-ms004-brest-alexandre-muller-vs-luca-van-assche',
           '2021-7341-ms005-brest-brandon-nakashima-vs-henri-laaksonen',
           # '2021-7341-ms006-brest-maxime-janvier-vs-richard-gasquet',
           '2021-7341-ms007-brest-joao-sousa-vs-federico-gaio',
           '2021-7341-ms008-brest-luca-van-assche-vs-manuel-guinard',
           '2021-7341-ms009-brest-alexandre-muller-vs-stefano-travaglia',
           '2021-7341-ms010-brest-brandon-nakashima-vs-bernabe-zapata-miralles',
           '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano']
sto = 'segmentation_and_labelling'

zoom_factor = 2

event_label_names = ['labelled_top_hit_Flag', 'labelled_bot_hit_Flag', 
                     'labelled_top_serve_Flag', 'labelled_bot_serve_Flag']
other_label_names = ['labelled_top_other_Flag', 'labelled_bot_other_Flag']

sfDict = {}
othDict = {}
countDict = {'top_other': [],
             'bot_other': []}
for vid in vidList:
    
    ### Get the labelled events
    labSF = StreamFeatures.from_store(vid, sto, featnameList = event_label_names)
    sfDict[vid] = labSF
    
    ### Check which player and which event has been worked on the least!
    othSF = StreamFeatures.from_store(vid, sto, featnameList = other_label_names)
    othDict[vid] = labSF
    
    for feat in other_label_names:
        _, player, event, _ = feat.split('_')
        if feat in othSF.get_features():
            
            vcSer = othSF[feat].value_counts()
            count = vcSer[event + ' ' + 'begin split stepping']
            
            countDict[player + '_' + event].append(count)
        else:
            countDict[player + '_' + event].append(0)
 
sumDict = {k: sum(v) for k, v in countDict.items()}
todo_player_event = [k for k, v in sumDict.items() if v == min(sumDict.values())][0] ### take first, at random
player_event_arg = np.argmin(countDict[todo_player_event])
todo_video = vidList[player_event_arg]
todo_player, todo_event = todo_player_event.split('_')

#%% FILL IN LABELLED SEGMENTS

labSF = sfDict[todo_video]

def group_dict_with_events(groupDict, eventDict):
    return {frame: eventDict[key] for frame, key in groupDict.items()}

def make_fill_from_eventSF(singleSF, event):
    eventDict = get_event_dict(event)
    fillDict = get_fill_dict(event)
    
    groupList = group_events(singleSF, event)
    
    ok_groupList = [groupDict for groupDict in groupList if check_group(groupDict, event)]
    event_groupList = [group_dict_with_events(groupDict, eventDict) for groupDict in ok_groupList]
    unambiguous_groupList = [unambiguous_groupDict  for groupDict in event_groupList for unambiguous_groupDict in unambiguify_group(groupDict)]
    
    filled_groupList = [fill_in_group(groupDict, fillDict) for groupDict in unambiguous_groupList] 

    return filled_groupList
    
try:
    serveSF = labSF[f'labelled_{todo_player}_serve_Flag'].dropna()
except KeyError:
    serveDict = {}
else:
    serveLists = make_fill_from_eventSF(serveSF, 'serve')
    serveDict = {}
    for serve in serveLists:
        for frame, event in serve.items():
            serveDict[frame] = event ### duplicates are not important here! all that matters is segmentation, so we can classify walking, running, etc
     
try:
    hitSF = labSF[f'labelled_{todo_player}_hit_Flag'].dropna()
except KeyError:
    hitDict = {}
else:
    hitLists = make_fill_from_eventSF(hitSF, 'hit')
    hitDict = {}
    for hit in hitLists:
        for frame, event in hit.items():
            hitDict[frame] = event ### duplicates are not important here! all that matters is segmentation, so we can classify walking, running, etc
      
jntDict = {**serveDict, **hitDict}
sortDict = dict(sorted(jntDict.items()))
fillSF = StreamFeature(sortDict, name = f'filled_{todo_player}')

### Construct a new segmenter --> where do we have no hit or serve, but still a court?
segBF = BoolFeature.from_store(todo_video, sto, featname = 'isCourt')
fs, ls = fillSF.first()[0], fillSF.last()[0]
frameList = range(fs, ls + 1)
emptyBF = fillSF.conform(frameList).isna()
non_event_BF = emptyBF.filter(emptyBF).filter(segBF)
non_event_BFs = non_event_BF.split()

cap = ut.get_video(todo_video)
myStms = StreamList.from_SFList(non_event_BFs, cap)

### check which have been done, and which havent
readSF = StreamFeatures.from_store(todo_video, sto, featnameList = [f'labelled_{todo_player}_other_Flag']).squeeze()

is_done = []
if not readSF.is_empty():
    for idx, myStm in enumerate(myStms):
        frameList = list(range(myStm.start_frame, myStm.stop_frame + 1))
        partSF = readSF.conform(frameList).dropna()
        if not partSF.is_empty():
            is_done.append(idx)        
   

for idx, myStm in enumerate(myStms):
    
    print(f'Stream {idx}...')
    
    if idx in is_done:
        print('Previously Done! Skipping.')
        continue
    
    evtSF, breakBool = record_events(myStm, 
                                     which_player = player, which_event = 'other', 
                                     zoom_factor = zoom_factor, auto_ending_event = 'other end')

    evtSF.save(todo_video, sto)
    
    if breakBool:
        break











