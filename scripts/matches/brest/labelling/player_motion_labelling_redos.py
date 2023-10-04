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

# label_names = ['labelled_top_hit_Flag', 'labelled_bot_hit_Flag', 
#                'labelled_top_serve_Flag', 'labelled_bot_serve_Flag']

VIDNUM = 9
MARGIN = 0
REDO_VERSION = 'v3'
REDO_NAME = f'labelled_all_uncaught_check_{REDO_VERSION}_Flag'
EVENT = 'serve'
PLAYER = 'bot'
label_name = f'labelled_{PLAYER}_{EVENT}_Flag'

vid = list(vidDict.keys())[VIDNUM]
cap = ut.get_video(vid)

redoSF = StreamFeatures.from_store(vid, sto, featnameList = [REDO_NAME]).squeeze()
redoBF = redoSF == 'redo'
if not redoBF.true() == 0:
    redoBFs = redoBF.filter(redoBF).split()
    
    for redoBF in redoBFs:
        labSF = StreamFeatures.from_store(vid, sto, featnameList = [label_name]).conform(redoBF).squeeze()
        
        myStm = Stream.from_SF(labSF, cap)
        myStm = myStm.conform(myStm.start_frame - MARGIN, myStm.stop_frame + MARGIN)
    
        evtSF, breakBool = record_events(myStm, 
                                         which_player = PLAYER, 
                                         which_event = EVENT,
                                         zoom_factor = zoom_factor)
        
        evtSF.save(vid, sto)
            
        if breakBool:
            break
        

        
        
        
        
        
        
        
        
        
    