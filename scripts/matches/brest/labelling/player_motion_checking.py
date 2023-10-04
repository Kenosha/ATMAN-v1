# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:34:57 2023

@author: gianl
"""


#### READ IN ######
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Features import BoolFeature, BoolFeatures, StreamFeatures, SFList
from ATMANpy.types.Streams import Stream, StreamList
from ATMANpy.funcs.label_funcs import record_events, group_events, check_group, check_other, fill_in_group, group_map, classify_event

vidDict = {
            # '2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa': (0, 2),
            # '2021-7341-ms002-brest-brandon-nakashima-vs-alexandre-muller': (0, 1),
            # '2021-7341-ms003-brest-joao-sousa-vs-maxime-janvier': (0, 2),
            # '2021-7341-ms004-brest-alexandre-muller-vs-luca-van-assche': (0, 2),
            # '2021-7341-ms005-brest-brandon-nakashima-vs-henri-laaksonen': (0, 2),
            # '2021-7341-ms007-brest-joao-sousa-vs-federico-gaio': (0, 2),
            # '2021-7341-ms008-brest-luca-van-assche-vs-manuel-guinard': (0, 1),
            # '2021-7341-ms009-brest-alexandre-muller-vs-stefano-travaglia': (1, 1), ### has warm up as first isCourt segment!
            # '2021-7341-ms010-brest-brandon-nakashima-vs-bernabe-zapata-miralles': (1, 2), ### has warm up as first isCourt segment!
            '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano': (0, 3)
           }
sto = 'segmentation_and_labelling'

zoom_factor = 2

label_names = ['labelled_top_hit_Flag', 'labelled_bot_hit_Flag', 
               'labelled_top_serve_Flag', 'labelled_bot_serve_Flag']
other_names = ['labelled_top_other_Flag', 'labelled_bot_other_Flag']

labDict = {}
othDict = {}
segDict = {}
countDict = {'top_hit': [],
             'bot_hit': [],
             'top_serve': [],
             'bot_serve': []}
for vid in vidDict:
    
    ### Check which player and which event has been worked on the least!
    labSF = StreamFeatures.from_store(vid, sto, featnameList = label_names)
    othSF = StreamFeatures.from_store(vid, sto, featnameList = other_names)
    segBF = BoolFeatures.from_store(vid, sto, featnameList = ['isCourt']).squeeze()
    labDict[vid] = labSF
    othDict[vid] = othSF
    segDict[vid] = segBF
    
    for feat in label_names:
        _, player, event, _ = feat.split('_')
        if feat in labSF.get_features():
            
            vcSer = labSF[feat].value_counts()
            count = vcSer[event + ' ' + 'begin swinging']
            
            countDict[player + '_' + event].append(count)
        else:
            countDict[player + '_' + event].append(0)
 
    

#%% CHECK FOR INCOMPLETE OR BROKEN EVENTS 

### TODO: Mark out incomplete or broken events
# Broken: Has a hole in duplicate event (impact - none - impact)
# Broken: Wrong order
# Incomplete: Not finished or not started (i.e. good order but missing beginning or end)

########### MAIN CHECK LOOP ###########

redo_margin_in_frames = 40

for event in ['hit', 'serve']:
    
    for player in ['bot', 'top']:
        for vid in vidDict:
            ### Get the labelled StreamFeatures for video, event, player
            labSF = labDict[vid]
            
            try:
                singleSF = labSF[f'labelled_{player}_{event}_Flag'].dropna()
            except KeyError:
                continue
            
            groupList = group_events(singleSF, event)

            incomplete_groups = [group for group in groupList if not check_group(group, event)]

            if incomplete_groups != []:
                
                print('_________')
                print(f'{len(incomplete_groups)} Incomplete groups in: {vid}')
                print(f'  ({event}, {player})')
                
                cap = ut.get_video(vid)
                
                for group in incomplete_groups:
                    frames = sorted(group.keys())
                    fs, ls = frames[0], frames[-1]
                    
                    # print({frame: eventDict[key] for frame, key in group.items()})
                    
                    ### Re-record!
                    myStm = Stream(cap, fs - redo_margin_in_frames, ls + redo_margin_in_frames).attach(singleSF)
                    evtSF, breakBool = record_events(myStm, 
                                                      which_player = player, 
                                                      which_event = event,
                                                      zoom_factor = zoom_factor)
                    evtSF.save(vid, sto)
                    
                    if breakBool:
                        break


redo_margin_in_frames = 10
event = 'other'

for player in ['bot', 'top']:
    othSF = othDict[vid]
    
    try:
        singleSF = othSF[f'labelled_{player}_{event}_Flag'].dropna()
    except KeyError:
        continue

    redo_frames = check_other(singleSF)

    for fs, ls in redo_frames:
        myStm = Stream(cap, fs - redo_margin_in_frames, ls + redo_margin_in_frames).attach(singleSF)
        evtSF, breakBool = record_events(myStm, 
                                          which_player = player, 
                                          which_event = event,
                                          zoom_factor = zoom_factor)
        evtSF.save(vid, sto)
        
        if breakBool:
            break

#%% CHECK FOR UNCAUGHT EVENTS

VERSION = 'v3'

VIDNUM = 9
vid = list(vidDict.keys())[VIDNUM]

cap = ut.get_video(vid)
fs_seg, ls_seg = vidDict[vid]

### Figure out which frames to cut 
### -> we want to look at segments which have not at all been labelled!
segBF = segDict[vid]
labSF = labDict[vid]

segBFs = segBF.filter(segBF).split()
for sub_courtBF in segBFs[fs_seg: ls_seg + 1]:
    
    all_labels = []
    sub_labelSF = labSF.conform(sub_courtBF)
    
    for event in ['hit', 'serve']:
        for player in ['bot', 'top']:
            singleSF = labSF[f'labelled_{player}_{event}_Flag'].dropna()
            for group in group_events(singleSF, event):
                frames = sorted(group.keys())
                fs, ls = frames[0], frames[-1]
                current_labels = list(range(fs, ls + 1))
                all_labels.extend(current_labels)
                
    sub_labelBF = BoolFeature({frame: False for frame in all_labels}, name = 'notLabelled').conform(sub_courtBF).fillna(True)
    sub_labelBFs = sub_labelBF.filter(sub_labelBF).split()
    
    okSFs = []
    for sub_labelBF in sub_labelBFs:
        myStm = Stream.from_SF(sub_labelBF, cap)

        okSF, breakBool = classify_event(myStm, 'all', 'uncaught_check', VERSION, fps = 100)
        okSFs.append(okSF)
                
        if breakBool:
            break
    
    okSF = SFList(okSFs).concatenate()
    okSF.save(vid, sto)
    
    if breakBool:
        break





