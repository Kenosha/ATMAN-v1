# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:42:10 2023

The successor to high level funcs. 
These functions are fed to the .extract method of Stream and CachedStream.
They use the streams to extract the desired features.

Function argument names and return names matter, since they are parsed by the .extract method.

If a function does not require a cache (and is thus fed to a Stream instance), 
arguments and values have to be named after the features that they denote.
Frames have to be named cFrm, gFrm or hFrm, 
depending on whether we work with a BGR, Greyscale, or HSV representation.

If a function requires a cache (and is thus fed to a CachedStream instance),
arguments have to be named frameCache and featCache, understood to be of Cache type.
For return values (i.e. the outputted features), names have to correspond to the intended features.

In principle, these names of features can be arbitrary, but should be consistent!
For example, the courtline homography is named 'H'.

@author: gianl
"""

import numpy as np

from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

from ATMANpy.funcs.court_funcs import find_homography, find_homography_225_only, update_homography
from ATMANpy.funcs.scoreboard_funcs import find_scoreboard

from ATMANpy.types.Frames import BGRFrame, GreyFrame

def find_courtlines(H, schemaLines = schemaLines):
    if H is None:
        return None
    else:
        courtLines = schemaLines.transform('apply_homography', H).transform('round_points')
    return courtLines

def track_courtlines(frameCache, featCache,
                     schemaLines = schemaLines, only_225 = True):
    
    ### extract values from cache
    curr_gFrm = BGRFrame(frameCache[0]).to_grey()
    prev_H = featCache[-1]['H']
    prev_config = featCache[-1]['config']
    
    if prev_H is None or prev_config is None:
        if only_225:
            H, config = find_homography_225_only(curr_gFrm)
        else:
            H, config = find_homography(curr_gFrm)
        trackFlag = 'Found'
    else:
        # print('Courtlines found, updating homography...')
        H, config = update_homography(curr_gFrm, prev_H, prev_config)
        trackFlag = 'Tracked'
        
    if not H is None:
        courtLines = find_courtlines(H, schemaLines)
    else:
        trackFlag = 'Not Found'
        courtLines = None
        
    return courtLines, H, config, trackFlag

def redo_not_found(gFrm, courtLines, H, config, trackFlag, **kwargs):
    try:
        H, config = find_homography(gFrm, **kwargs)
    except:
        H = None
        trackFlag = 'Failed'
    else:
        if H is not None:
            trackFlag = 'Found (w. gen. H.)'
            courtLines = find_courtlines(H, schemaLines)

    return courtLines, H, config, trackFlag
            
def iterative_redo(gFrm, courtLines, H, config, trackFlag, 
                   min_votes = 500):
    try:
        H, config = find_homography(gFrm, min_votes = min_votes)
    except Exception as e:
        # print(e)
        if min_votes > 100:
            courtLines, H, config, trackFlag = iterative_redo(gFrm, courtLines, H, config, trackFlag, 
                                                              min_votes = min_votes // 2)
        return courtLines, H, config, trackFlag 
    else:
        if H is not None:
            trackFlag = 'Found (w. gen. H.)'
            courtLines = find_courtlines(H, schemaLines)
        return courtLines, H, config, trackFlag
        
    
def courtlines_to_schema(H, hFrm, schemaFrm = schemaFrm):
    ######### APPLY INVERSE H TO FIT FRAME COLORS RELATIVE TO COURTLINES BACK TO SCHEMA #######
    H_inv = np.linalg.inv(H)
    wrpFrm = hFrm.warp(H, cols = schemaFrm.cols, rows = schemaFrm.rows)
    return wrpFrm