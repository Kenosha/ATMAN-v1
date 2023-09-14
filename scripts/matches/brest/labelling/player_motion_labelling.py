# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:34:04 2023

@author: gianl
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 17:17:30 2023

@author: gianl
"""


#### READ IN ######
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Features import BoolFeatures, StreamFeatures
from ATMANpy.types.Streams import Stream, StreamList
from ATMANpy.funcs.label_funcs import get_event_dict, record_events

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
 
#%% RECORD NEW MATERIAL

hit_to_serve_ratio = 10 ### serves are easier to classify, so we do more hits!        
sumDict = {k: sum(v) for k, v in countDict.items()}
weightDict = {k: v / hit_to_serve_ratio if 'hit' in k else v for k, v in sumDict.items()} 

todo_player_event = [k for k, v in weightDict.items() if v == min(weightDict.values())][0] ### take first, at random
player_event_arg = np.argmin(countDict[todo_player_event])
todo_video = vidList[player_event_arg]
todo_player, todo_event = todo_player_event.split('_')

courtBF = BoolFeatures.from_store(todo_video, sto, featnameList = ['isCourt']).squeeze()
courtBFs = courtBF.filter(courtBF).split()   

cap = ut.get_video(todo_video)
myStms = StreamList.from_SFList(courtBFs, cap)

readSF = StreamFeatures.from_store(todo_video, sto, featnameList = [f'labelled_{todo_player}_{todo_event}_Flag']).squeeze()

### check which have been done, and which havent
is_done = []
if not readSF.is_empty():
    for idx, myStm in enumerate(myStms):
        frameList = list(range(myStm.start_frame, myStm.stop_frame + 1))
        partSF = readSF.conform(frameList).dropna()
        if not partSF.is_empty():
            is_done.append(idx)        
   
print('___________')
print(f'Doing {todo_event} for {todo_player} ...')
print(f'(for which {sumDict[todo_player_event]} have already been done.)')
print(f'... in {todo_video} ...')
print(f'(for which {countDict[todo_player_event][player_event_arg]} have already been done.)')
print(f'Stream Indices that we are skipping: {is_done}')
    
### Record Loop
for idx, myStm in enumerate(myStms):
    print(f'Stream {idx}...')
    
    if idx in is_done:
        print('Previously Done! Skipping.')
        continue
    
    evtSF, breakBool = record_events(myStm, 
                                     which_player = todo_player, 
                                     which_event = todo_event,
                                     zoom_factor = zoom_factor)
    
    evtSF.save(todo_video, sto)
    
    if breakBool:
        break
    