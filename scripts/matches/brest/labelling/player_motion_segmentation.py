# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:34:55 2023

In this file, we segment play sequences for the labelled video segments, 
so that we can most easily create the player contours.
The beginnings are marked by the beginning of the serve motion (maybe minus a margin of x frames),
The ends are marked by the last shot before the next serve (plus a margin of y frames)

Actually, to include some more walking, other actions -> serve until next serve, except where this would be too long!
When sides change (top serve to bottom), we can leave in a few seconds, then cut. usually, these are the longest intermissions.

@author: gianl
"""

#### READ IN ######
import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.sf_funcs as sf

from ATMANpy.types.Lines import Region, Regions
from ATMANpy.types.Features import BoolFeature, BoolFeatures, StreamFeature, StreamFeatures, SFList, ConstFeatures
from ATMANpy.types.Streams import Stream, StillStream, CachedStream, MulStream, StreamList
from ATMANpy.funcs.label_funcs import group_events, check_group, check_other, fill_in_group, group_to_dataframe, other_to_dataframe

MARGIN = 5

if __name__ == '__main__':
    ### vidDict maps the video id to the first and last isCourt segment we want to label.
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
    
    serve_names = ['labelled_top_serve_Flag', 'labelled_bot_serve_Flag']
    hit_names = ['labelled_top_hit_Flag', 'labelled_bot_hit_Flag']
    
    
    for vid in vidDict:    
        cap = ut.get_video(vid)
        
        srvSF = StreamFeatures.from_store(vid, sto, featnameList = serve_names)
        hitSF = StreamFeatures.from_store(vid, sto, featnameList = hit_names)
        segBF = BoolFeatures.from_store(vid, sto, featnameList = ['isCourt']).squeeze()
    
        ### Create isCourt Segments of Interest
        fs, ls = vidDict[vid]
        segBFs = segBF.filter(segBF).split()[fs: ls + 1]
        segBF = segBFs.concatenate()
    
        jntSFs = segBF.merge(srvSF).merge(hitSF).split()
        for idx, jntSF in enumerate(jntSFs):
            top_serves = group_events(jntSF['labelled_top_serve_Flag'].dropna(), 'serve')
            bot_serves = group_events(jntSF['labelled_bot_serve_Flag'].dropna(), 'serve')
            
            seg_starts = sorted([min(serve.keys()) for serve in top_serves + bot_serves])
            x1, xs = seg_starts[0], seg_starts[1:]
            
            all_segs = []
            for x2 in xs:
                curr_seg = list(range(x1, x2 - MARGIN)) ### TODO: If necessary, implement max length here!
                all_segs.extend(curr_seg)
                x1 = x2
                
            playBF = BoolFeature({frame: frame in all_segs for frame in jntSF.get_frames()}, name = 'isPlay')
            
            playBF.save(vid, sto)
            
            # chkFrm = jntSF.drop('isCourt').to_dataframe().dropna(how = 'all')
            ### TODO: Use this to establish a good segmentation!
            # however, to do this, we need to ensure all serves and all hits are labelled for all of the above segments!



