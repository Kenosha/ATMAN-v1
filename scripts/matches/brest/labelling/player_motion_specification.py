# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:34:55 2023

In this file, we further specify the hits and serves labelled earlier.
For hits, we add info on:
    - side: forehand/backhand
    - spin: topspin/slice/flat
    - type: groundstroke/volley/overhead
For serves, we add info on:
    - side: ad/deuce
    - (first/second)

@author: gianl
"""


#### READ IN ######
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Features import BoolFeatures, StreamFeature, StreamFeatures, SFList
from ATMANpy.types.Streams import Stream, StreamList
from ATMANpy.funcs.label_funcs import get_event_dict, record_events, group_events, check_group, classify_event

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
class_names = ['labelled_top_hit_side_Flag', 'labelled_bot_hit_side_Flag',
               'labelled_top_hit_spin_Flag', 'labelled_bot_hit_spin_Flag',
               'labelled_top_hit_type_Flag', 'labelled_bot_hit_type_Flag',
               'labelled_top_serve_side_Flag', 'labelled_bot_serve_side_Flag']

##### READ IN #######

clDict = {}
sfDict = {}
countDict = {'top_hit_side': [],
             'bot_hit_side': [],
             'top_hit_spin': [],
             'bot_hit_spin': [],
             'top_hit_type': [],
             'bot_hit_type': [],
             'top_serve_side': [],
             'bot_serve_side': []}
for vid in vidList:
    
    labSF = StreamFeatures.from_store(vid, sto, featnameList = label_names)
    clsSF = StreamFeatures.from_store(vid, sto, featnameList = class_names)
    
    clDict[vid] = clsSF    
    sfDict[vid] = labSF
    
#%%## Main Loop 

breakBool = False

### -> easier to do categories in one big batch
# -> first sides, for serve and hit
# -> spin, for hit
# -> type, for hit

show_margins_dict = {'side': (0, 0),
                     'spin': (0, 15),
                     'type': (15, 0)} ### how many frames before and after to show

#################
##### SIDE ######
#################
todo_category = 'side'
bef, aft = show_margins_dict[todo_category]

for todo_video in vidList:
    
    cap = ut.get_video(todo_video)
    
    ### SERVES ###
    todo_event = 'serve'
    for todo_player in ['top', 'bot']:
        try:
            labSF = sfDict[todo_video][f'labelled_{todo_player}_{todo_event}_Flag'].dropna()
        except:
            continue
        
        try:
            clsSF = clDict[todo_video][f'labelled_{todo_player}_{todo_event}_{todo_category}_Flag'].dropna()
        except:
            clsSF = StreamFeature({})
            
        #### CHECK WHAT HAS BEEN DONE ###
        catSFs = []
        if not clsSF.is_empty():
            todoBF = clsSF.conform(labSF).isna()
            labSF = labSF.filter(todoBF)
        
        ### CYCLE THROUGH INDIVIDUAL SHOTS!
        for groupDict in group_events(labSF, todo_event):
            frames = sorted(groupDict.keys())
            fs, ls = frames[0], frames[-1]
            
            myStm = Stream(cap, fs, ls + 1)
            
            catSF, breakBool = classify_event(myStm, todo_player, todo_event, todo_category, zoom_factor = zoom_factor)
            catSFs.append(catSF)
        
            if not catSF.is_empty():
                print(catSF.unique()[0])
        
            if breakBool:
                break
        
        catSF = SFList([sf for sf in catSFs if not sf.is_empty()]).concatenate()
        catSF.save(todo_video, sto)
        
        if breakBool:
            break
            
    if breakBool:
        break  
    
    ### HITS ###
    todo_event = 'hit'
    for todo_player in ['top', 'bot']:
    
        try:
            labSF = sfDict[todo_video][f'labelled_{todo_player}_{todo_event}_Flag'].dropna()
        except:
            continue
        
        try:
            clsSF = clDict[todo_video][f'labelled_{todo_player}_{todo_event}_{todo_category}_Flag'].dropna()
        except:
            clsSF = StreamFeature({})
    
        #### CHECK WHAT HAS BEEN DONE ###
        catSFs = []
        if not clsSF.is_empty():
            todoBF = clsSF.conform(labSF).isna()
            labSF = labSF.filter(todoBF)
        
        ### CYCLE THROUGH INDIVIDUAL SHOTS!
        for groupDict in group_events(labSF, todo_event):
            frames = sorted(groupDict.keys())
            fs, ls = frames[0] - bef, frames[-1] + aft
            
            myStm = Stream(cap, fs, ls + 1)
            
            catSF, breakBool = classify_event(myStm, todo_player, todo_event, todo_category, zoom_factor = zoom_factor)
            catSFs.append(catSF)
        
            if not catSF.is_empty():
                print(catSF.unique()[0])
        
            if breakBool:
                break
        
        catSF = SFList([sf for sf in catSFs if not sf.is_empty()]).concatenate()
        catSF.save(todo_video, sto)
        
        if breakBool:
            break
        
    if breakBool:
        break
        
###############
##### SPIN ####
###############

todo_category = 'spin'
bef, aft = show_margins_dict[todo_category]

for todo_video in vidList:
    
    cap = ut.get_video(todo_video)
    
    ### HITS ###
    todo_event = 'hit'
    for todo_player in ['top', 'bot']:
    
        try:
            labSF = sfDict[todo_video][f'labelled_{todo_player}_{todo_event}_Flag'].dropna()
        except:
            continue
        
        try:
            clsSF = clDict[todo_video][f'labelled_{todo_player}_{todo_event}_{todo_category}_Flag'].dropna()
        except:
            clsSF = StreamFeature({})
    
        #### CHECK WHAT HAS BEEN DONE ###
        catSFs = []
        if not clsSF.is_empty():
            todoBF = clsSF.conform(labSF).isna()
            labSF = labSF.filter(todoBF)
        
        ### CYCLE THROUGH INDIVIDUAL SHOTS!
        for groupDict in group_events(labSF, todo_event):
            frames = sorted(groupDict.keys())
            fs, ls = frames[0] - bef, frames[-1] + aft
            
            myStm = Stream(cap, fs, ls + 1)
            
            catSF, breakBool = classify_event(myStm, todo_player, todo_event, todo_category, zoom_factor = zoom_factor)
            catSFs.append(catSF)
        
            if not catSF.is_empty():
                print(catSF.unique()[0])
        
            if breakBool:
                break
        
        catSF = SFList([sf for sf in catSFs if not sf.is_empty()]).concatenate()
        catSF.save(todo_video, sto)
        
        if breakBool:
            break
        
    if breakBool:
        break
      
###############
##### TYPE ####
###############

todo_category = 'type'
bef, aft = show_margins_dict[todo_category]

for todo_video in vidList:
    
    cap = ut.get_video(todo_video)
    
    ### HITS ###
    todo_event = 'hit'
    for todo_player in ['top', 'bot']:
    
        try:
            labSF = sfDict[todo_video][f'labelled_{todo_player}_{todo_event}_Flag'].dropna()
        except:
            continue
        
        try:
            clsSF = clDict[todo_video][f'labelled_{todo_player}_{todo_event}_{todo_category}_Flag'].dropna()
        except:
            clsSF = StreamFeature({})
    
        #### CHECK WHAT HAS BEEN DONE ###
        catSFs = []
        if not clsSF.is_empty():
            todoBF = clsSF.conform(labSF).isna()
            labSF = labSF.filter(todoBF)
        
        ### CYCLE THROUGH INDIVIDUAL SHOTS!
        for groupDict in group_events(labSF, todo_event):
            frames = sorted(groupDict.keys())
            fs, ls = frames[0] - bef, frames[-1] + aft
            
            myStm = Stream(cap, fs, ls + 1)
            
            catSF, breakBool = classify_event(myStm, todo_player, todo_event, todo_category, zoom_factor = zoom_factor)
            catSFs.append(catSF)
        
            if not catSF.is_empty():
                print(catSF.unique()[0])
        
            if breakBool:
                break
        
        catSF = SFList([sf for sf in catSFs if not sf.is_empty()]).concatenate()
        catSF.save(todo_video, sto)
        
        if breakBool:
            break
        
    if breakBool:
        break
    
 
    
 
 
    
 
    