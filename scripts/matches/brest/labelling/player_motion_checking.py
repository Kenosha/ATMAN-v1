# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:34:57 2023

@author: gianl
"""


#### READ IN ######
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Features import BoolFeatures, StreamFeatures
from ATMANpy.types.Streams import Stream, StreamList
from ATMANpy.funcs.label_funcs import get_event_dicts, record_events, group_events, check_group

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

label_names = ['labelled_top_hit_Flag', 'labelled_bot_hit_Flag', 
               'labelled_top_serve_Flag', 'labelled_bot_serve_Flag']

jointDict = get_event_dicts()

sfDict = {}
countDict = {'top_hit': [],
             'bot_hit': [],
             'top_serve': [],
             'bot_serve': []}
for vid in vidList:
    
    ### Check which player and which event has been worked on the least!
    labSF = StreamFeatures.from_store(vid, sto, featnameList = label_names)
    sfDict[vid] = labSF
    
    for feat in label_names:
        _, player, event, _ = feat.split('_')
        if feat in labSF.get_features():
            
            vcSer = labSF[feat].value_counts()
            count = vcSer[event + ' ' + 'begin swinging']
            
            countDict[player + '_' + event].append(count)
        else:
            countDict[player + '_' + event].append(0)
 
    

#%% CHECK INCOMPLETE OR BROKEN EVENTS 

### TODO: Mark out incomplete or broken events
# Broken: Has a hole in duplicate event (impact - none - impact)
# Broken: Wrong order
# Incomplete: Not finished or not started (i.e. good order but missing beginning or end)

########### MAIN CHECK LOOP ###########

redo_margin_in_frames = 100

for event in ['hit', 'serve']:
    eventDict = jointDict[event]
    
    if event == 'hit':
        dispensable_events = ['begin loading']
    else:
        dispensable_events = []
    
    for player in ['bot', 'top']:
        for vid in vidList:
            ### Get the labelled StreamFeatures for video, event, player
            labSF = sfDict[vid]
            
            try:
                singleSF = labSF[f'labelled_{player}_{event}_Flag'].dropna()
            except KeyError:
                continue
            
            groupList = group_events(singleSF, event)

            incomplete_groups = [group for group in groupList if not check_group(group, eventDict, dispensable_events = dispensable_events)]

            if incomplete_groups != []:
                
                print('_________')
                print(f'{len(incomplete_groups)} Incomplete groups in: {vid}')
                print(f'  ({event}, {player})')
                
                cap = ut.get_video(vid)
                
                for group in incomplete_groups:
                    frames = sorted(group.keys())
                    fs, ls = frames[0], frames[-1]
                    
                    print({frame: eventDict[key] for frame, key in group.items()})
                    
                    ### Re-record!
                    myStm = Stream(cap, fs - redo_margin_in_frames, ls + redo_margin_in_frames).attach(singleSF)
                    evtSF, breakBool = record_events(myStm, 
                                                     which_player = player, 
                                                     which_event = event,
                                                     zoom_factor = zoom_factor)
                    evtSF.save(vid, sto)
                    
                    if breakBool:
                        break
