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

## TODO: ADD BEGIN UNIT TURN --> maybe even instead of begin loading....

#### READ IN ######
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut

from ATMANpy.types.Features import BoolFeatures, StreamFeatures
from ATMANpy.types.Streams import Stream, StreamList
from ATMANpy.funcs.label_funcs import get_event_dict, record_events

vidDict = {
            '2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa': (0, 2),
            '2021-7341-ms002-brest-brandon-nakashima-vs-alexandre-muller': (0, 1),
            '2021-7341-ms003-brest-joao-sousa-vs-maxime-janvier': (0, 2),
            '2021-7341-ms004-brest-alexandre-muller-vs-luca-van-assche': (0, 2),
            '2021-7341-ms005-brest-brandon-nakashima-vs-henri-laaksonen': (0, 2),
            '2021-7341-ms007-brest-joao-sousa-vs-federico-gaio': (0, 2),
            '2021-7341-ms008-brest-luca-van-assche-vs-manuel-guinard': (0, 1),
            '2021-7341-ms009-brest-alexandre-muller-vs-stefano-travaglia': (1, 1), ### has warm up as first isCourt segment!
            '2021-7341-ms010-brest-brandon-nakashima-vs-bernabe-zapata-miralles': (1, 2), ### has warm up as first isCourt segment!
            '2021-7341-ms013-brest-richard-gasquet-vs-thomas-fabbiano': (0, 3)
           }
sto = 'segmentation_and_labelling'

zoom_factor = 2

label_names = ['labelled_top_hit_Flag', 'labelled_bot_hit_Flag', 
               'labelled_top_serve_Flag', 'labelled_bot_serve_Flag']
 
###### READ IN #######
VIDNUM = 0
EVENT = 'serve'
vid = list(vidDict.keys())[VIDNUM]   
cap = ut.get_video(vid) 

### Check which player and which event has been worked on the least!
labSF = StreamFeatures.from_store(vid, sto, featnameList = label_names)
segBF = BoolFeatures.from_store(vid, sto, featnameList = ['isCourt']).squeeze()


#%% RECORD NEW MATERIAL

fs, ls = vidDict[vid]
jntSFs = segBF.filter(segBF).merge(labSF).split()[fs: ls + 1]

for jntSF in jntSFs:
    
    crtBF = jntSF['isCourt'].to_bool()
    jntSF = jntSF.drop('isCourt')
    
    for feat in label_names:
        try: ### in this case ...
            featSF = jntSF[feat].dropna() # ... the feature is in the features of labSF ...
            do_feat = featSF.is_empty() # ... but not done for this specific jntSF!
        except: ### in this case, the feature is not in any jntSFs (and therefore not in labSF.)
            do_feat = True  
            
        if do_feat:
            print(vid, feat)
            
            myStm = Stream.from_SF(jntSF, cap)
            _, todo_player, todo_event, _ = feat.split('_')
            if todo_event == EVENT:
                
                evtSF, breakBool = record_events(myStm, 
                                                 which_player = todo_player, 
                                                 which_event = todo_event,
                                                 zoom_factor = zoom_factor)
                
                evtSF.save(vid, sto)
                
                if breakBool:
                    break
        
        else:
            ### use currently available features to further split the segment, and then re-check these!
            pass
        
        

        
        
        
        
        
        
        
        
        
    