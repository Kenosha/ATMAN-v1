# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:45:37 2023

@author: gianl
"""

import time

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines

import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex

from ATMANpy.types.Streams import Stream, CachedStream

if __name__ == '__main__':
    VIDEO_ID = 'KTCDxjJvs2U'
    cap = get_video(VIDEO_ID)
    
    START_FRAME = 800
    RUN_LENGTH = 4000 ### TODO: Generalize this to take into account length of stream
    stop_frame = START_FRAME + RUN_LENGTH
    
    NUM_WORKERS = 24
    CHUNKSIZE = 100
    
    emptyCtm = CachedStream(cap, START_FRAME, stop_frame)
    emptyStm = Stream(cap, START_FRAME, stop_frame)
    
    ### Track the courtlines
    courtSF = emptyCtm.extract_chunked_mp(NUM_WORKERS, CHUNKSIZE, ex.track_courtlines)
    
    ### tighten the frame at which the match likely starts:
    notBF = courtSF['trackFlag'] != 'Not Found'
    first_frame, _ = courtSF.filter(notBF).first()
    
    ### reestablish the starting frame
    emptyCtm = CachedStream(cap, first_frame, stop_frame)
    emptyStm = Stream(cap, first_frame, stop_frame)
    courtSF = courtSF.conform(list(range(first_frame, stop_frame + 1)))
    
    ### redo frames that were not found
    redoBF = sf.get_redo_frames(courtSF)
    courtSF = courtSF.merge(redoBF)
    courtStm = emptyStm.attach(courtSF)
    redoSF = courtStm.extract_mp(NUM_WORKERS, ex.redo_not_found)
    
    ### find holes in homography
    notBF = redoSF['trackFlag'] != 'Not Found'
    redoBF2 = sf.get_hole_frames(redoSF.filter(notBF))
    redoBF2 = redoBF2.conform(list(range(first_frame, stop_frame + 1))).fillna(False)
    # print(redoBF2.true()) 
    
    ### redo holes
    redoSF = redoSF.merge(redoBF2)
    courtStm = emptyStm.attach(redoSF)
    redoSF2 = courtStm.extract(ex.redo_not_found) # courtStm.extract_mp(NUM_WORKERS, ex.redo_not_found) 
    ### TODO: even though there is not a single frame to be done, extracting in a multiprocessed way takes ages because of the mp overhead
    ### TODO: Maybe a proper where condition in extraction functions would help a lot, actually...

    courtSF = redoSF2[['H', 'courtLines']].dropna()
    sfList = courtSF.split()
    ### Correct Homographies
    cornerList = sfList.apply(sa.get_court_corner_points)
    regionList = cornerList.apply_whole(sf.median_corner_regions)
    mergeList = cornerList.merge(regionList)

    tlList = mergeList.apply(sa.correct_point, assignArgs = ['tl', 'tlReg'], assignVals = ['tl'])
    trList = mergeList.apply(sa.correct_point, assignArgs = ['tr', 'trReg'], assignVals = ['tr'])
    blList = mergeList.apply(sa.correct_point, assignArgs = ['bl', 'blReg'], assignVals = ['bl'])
    brList = mergeList.apply(sa.correct_point, assignArgs = ['br', 'brReg'], assignVals = ['br'])
    correctList = tlList.merge(trList).merge(blList).merge(brList)

        
