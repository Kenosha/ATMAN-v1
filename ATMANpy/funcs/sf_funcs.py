# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:00:53 2023

Here we write functions here that take a stream feature and return a stream feature, 
if the feature in question lends itself to vectorization or other speed ups 
that call for treating all features as a whole, across frames 

@author: gianl
"""

import time

import pandas as pd
import numpy as np

import copy

import ATMANpy.auxil.constants as const

from ATMANpy.types.Features import StreamFeatures, StreamFeature, BoolFeatures, BoolFeature, SFList
from ATMANpy.types.Streams import Stream
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Region
from ATMANpy.types.Contours import Contours, Contour
from ATMANpy.types.Shapes3D import Point3D, Line3D, Region3D, Ball3D, Circle3D

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_apply_funcs as sa

from ATMANpy.types.Lines import CourtLines
schemaLines = CourtLines.make().add_singles_netline()

def get_redo_frames(mySF, redo_threshold = 24): 
    ### below this many consequent frames that were not found,
    ### we will retry all of them.
    
    def list_to_ranges(inList):
        outList = []
        left_elem = None
        last_elem = None
        for i in inList:
            if last_elem is None:
                last_elem = i
                left_elem = i
                continue
                
            if left_elem is None:
                left_elem = i
                continue
            
            if i == last_elem + 1:
                last_elem = i
                continue
            elif i > last_elem + 1:
                outList.append((left_elem, last_elem))
                left_elem = i
                last_elem = i
                continue
            
        return outList
            
    def flatten(list_of_lists):
        if list_of_lists == []:
            return []
        else:
            return [*list_of_lists[0], *flatten(list_of_lists[1:])]
    
    nfSF = mySF.filter(mySF['trackFlag'] == 'Not Found')
    nfList = nfSF.keys()
    
    contiguous = list_to_ranges(nfList)      
    to_retry = [list(range(start, end + 1)) for start, end in contiguous if end - start < redo_threshold]
    to_retry_flat = flatten(to_retry)
    
    streamDict = {frame: frame in to_retry_flat for frame in mySF.keys()}
    
    return BoolFeature(streamDict, 'to_retry')

def get_hole_frames(mySF):
    ### similar to get_redo_frames above    
    
    DISTANCE_THRESHOLD = 15
    inList = mySF.split()
    end_and_begin = [(max(sf_prev), min(sf_succ)) for sf_prev, sf_succ in zip(inList[:-1], inList[1:])]
    end_and_begin = [(end, beg) for end, beg in end_and_begin if beg - end < DISTANCE_THRESHOLD]
    
    to_retry = []
    for end, beg in end_and_begin:
        frames_missing_between = list(range(end + 1, beg))
        to_retry.extend(frames_missing_between)
        
    # streamDict = {frame: frame in to_retry for frame in mySF.keys()}
    streamDict = {frame: True for frame in to_retry}
    
    return BoolFeature(streamDict, name = 'to_retry')

def median_corner_regions(inSF, how_many_unilateral = 8):
    
    required_features = ['tl', 'tr', 'bl', 'br']
    mySF = inSF[required_features]
    myArr = mySF.apply(sa.points_to_coords).to_array()
    numFrames = mySF.length()
    
    ### Auxiliary function for finding outlier courtlines    
    def get_point_bounds(myArr, 
                         lower_q = 0.2, upper_q = 0.8, acceptable_multiple = 3,
                         min_box_diameter = 4):
        
        qs = [lower_q, 0.5, upper_q]
        loPts, mdPts, hiPts = np.round(np.quantile(myArr, qs, axis = 0)).astype('int')
    
        ### interquantile distance
        iq_up, iq_lo = hiPts - mdPts, mdPts - loPts
        
        ### lower bind our interquantile distances, so we don't have tiny boxes.
        iq_up = iq_up.clip(min = min_box_diameter // 2)
        iq_lo = iq_lo.clip(min = min_box_diameter // 2)
    
        lower_bound = mdPts - iq_lo * acceptable_multiple - 1 ### subtract/add one because checking whether
        upper_bound = mdPts + iq_up * acceptable_multiple + 1 ### a point is inside region is checked STRONGLY.
    
        return lower_bound, upper_bound
    
    ### Identify 'normal' corner regions and whether a corner point is an outlier
    dictList = []
    for idx, ptArr in enumerate(myArr):
        featDict = {}
        
        ### calculate quantile court mask
        # a) get the surrounding frames court corner points
        left_slice = max(0, idx - how_many_unilateral) ### frames before!
        right_slice = min(idx + how_many_unilateral + 1, numFrames) ### frames after
        sliceArr = myArr[left_slice:right_slice]
        
        ### establish lower and upper bounds based on IQR for each point
        lower_bounds, upper_bounds = get_point_bounds(sliceArr) ### shape of each: 4 x 2
        
        tl_lo, tr_lo, bl_lo, br_lo = lower_bounds
        tl_hi, tr_hi, bl_hi, br_hi = upper_bounds
        
        featDict['tlReg'] = Region.from_points(Point2D(tl_lo), Point2D(tl_hi))
        featDict['trReg'] = Region.from_points(Point2D(tr_lo), Point2D(tr_hi))
        featDict['blReg'] = Region.from_points(Point2D(bl_lo), Point2D(bl_hi))
        featDict['brReg'] = Region.from_points(Point2D(br_lo), Point2D(br_hi))
        
        # tl, tr, bl, br = ptArr
        
        # featDict['within_tlReg'] = Point2D(tl).within(featDict['tlReg'])
        # featDict['within_trReg'] = Point2D(tr).within(featDict['trReg'])
        # featDict['within_blReg'] = Point2D(bl).within(featDict['blReg'])
        # featDict['within_brReg'] = Point2D(br).within(featDict['brReg'])
    
        dictList.append(featDict)
    
    return StreamFeatures(dict(zip(mySF.get_frames(), dictList)), name = 'median_corner_regions')
    
def get_normal_bool(courtSF, normalized_deviation_threshold = 1, verbose = False):
    
    def court_points(courtLines):
        p1, p2 = courtLines['Top Baseline'].leftPt, courtLines['Top Baseline'].rightPt
        p3, p4 = courtLines['Bottom Baseline'].leftPt, courtLines['Bottom Baseline'].rightPt
        p1, p2, p3, p4 = p1.coords, p2.coords, p3.coords, p4.coords
        # return [p1.coords, p2.coords, p3.coords, p4.coords]
        return p1, p2, p3, p4
    
    pointSF = courtSF['courtLines'].apply(court_points)
    courtpoints = pointSF.to_array()
    
    med_points = np.median(courtpoints, axis = 0)
    std_points = np.std(courtpoints, axis = 0)
    
    normalized_deviation = (courtpoints - med_points) / std_points
    within_bool = np.abs(normalized_deviation) < normalized_deviation_threshold
    all_corners_within = np.all(within_bool, axis = (1, 2))
    
    if verbose:
        print(f'   Number of Court Lines Found:   {len(courtpoints)}')
        print(f'    ... of which judged normal:   {all_corners_within.sum()}')
        print(f'    ... of which judged abnormal: {(~all_corners_within).sum()}')

    boolSF = BoolFeature(dict(zip(courtSF.keys(), all_corners_within)), name = 'all_corners_within')
    return boolSF

def get_nonserve_bool(courtSF, first_seconds_to_disqualify = 5):
    frames = courtSF.get_frames()
    disq_num = 25 * first_seconds_to_disqualify
    disq_frames = frames[0: disq_num]
    
    boolDict = {frame: frame not in disq_frames for frame in courtSF.get_frames()}
    return BoolFeature(boolDict, name = f'not_in_first_{first_seconds_to_disqualify}_seconds')

def get_play_bool(courtSF, last_seconds_to_take = 3, min_length_of_sequence = 13):
    frames = courtSF.get_frames()
    take_num, length_num = 25 * last_seconds_to_take, 25 * min_length_of_sequence
    if len(frames) < length_num:
        take_frames = []
    else:
        take_frames = frames[-take_num:]
    
    boolDict = {frame: frame in take_frames for frame in frames}
    return BoolFeature(boolDict, name = f'last_{last_seconds_to_take}_seconds_of_long_sequence')

def get_median_and_deviation_frame(wrpSF):
    
    from ATMANpy.types.Frames import HSVFrame
    
    wrpArrs = np.array([frm.array for frm in wrpSF.values()])
    
    ### prune away parts that are superfluous because they are completely dark
    ## in the unimaginable case that we have black rows within the picture, (not at the edges), 
    ## we'd be fucked.
    row_is_not_dark = wrpArrs.any(axis = (0, 2, 3))
    wrpArrs = wrpArrs[:, row_is_not_dark]
    
    
    # avgArr = np.average(wrpArrs, axis = 0)
    medArr = np.median(wrpArrs, axis = 0)
    stdArr = np.std(wrpArrs, axis = 0) ### memory requirements for standard deviation are quite high.....
                                         # check task manager while it runs; RAM usage increases a lot.
    
    # avgFrm = BGRFrame(avgArr.astype('uint8'))
    medFrm = HSVFrame(medArr.astype('uint8'), name = 'median_backFrm')
    stdFrm = HSVFrame(stdArr.astype('uint8'), name = 'std_backFrm')
    
    return medFrm, stdFrm

def split_player_candidate_tracks_forward(plSF, 
                                          frame_offset = 1,
                                          critical_number_of_tracks = 2000,
                                          verbose = False, super_verbose = False):
    
    if verbose: 
        start_time = time.time()
        
    ### Clear up offset
    frames = plSF.get_frames()
    frames_of_interest = frames[::frame_offset]
    fsFrm, other_frames = frames_of_interest[0], frames_of_interest[1:]
    
    ### get data from first frame 
    fsDict = plSF[fsFrm]
    fsChildren = fsDict['player_children']
    
    ### initialize a list of tracks
    last_trackList = []
    for currIdx, fsChild in enumerate(fsChildren):
        if fsChild == []:
            trackDict = {'cnt_idx': currIdx, 'child': None}
            last_trackList.append({fsFrm: trackDict})
        else:
            for child in fsChild:
                trackDict = {'cnt_idx': currIdx, 'child': child}
                last_trackList.append({fsFrm: trackDict})
    
    if verbose:
        loop_begin_time = time.time()
        
    for frame in other_frames:
        
        if super_verbose:
            chk = time.time()
            print(f'Frame Number: {frame} (until {plSF.last()[0]})')
            print(f'Number of Tracks: {len(last_trackList)}')
    
        featDict = plSF[frame]
        currChildren = featDict['player_children']
        
        curr_indices = []
        curr_trackList = [] ### this list of tracks is populated in the loop
        
        for last_track in last_trackList:
            
            ### Handle previously finished tracks ###
            try:
                parDict = last_track[frame - frame_offset]
            except KeyError:
                ### in this case, the track was previously completed from end to beginning, so we add it in unmodified
                curr_trackList.append(last_track)
                continue
        
            currIdx = parDict['child']
            
            ### Handle just finished tracks ###
            if currIdx is None: ### in this case, the track is just completed, so we add it in unmodified.
                curr_trackList.append(last_track)
                continue
            else:
                curr_indices.append(currIdx)
                
            currChild = currChildren[currIdx]
            
            if currChild == []: ### in this case, the track has no children, so it terminates here
                curr_track = last_track
                currDict = {'cnt_idx': currIdx, 'child': None}
                curr_track[frame] = currDict
                curr_trackList.append(curr_track)
    
            else:
                for child in currChild:
                    curr_track = dict(last_track) ### need to do this due to mutability and the loop over parents...
                    currDict = {'cnt_idx': currIdx, 'child': child}
                    curr_track[frame] = currDict
                    curr_trackList.append(curr_track)
                    
        if super_verbose:
            print(f'   ... done with track check: took {round(time.time() - chk, 1)} secs')
            chk = time.time()
                
        ### Handle tracks which are just begun: they are not pointed to by a parent.
        current_parless = [idx for idx in range(len(currChildren)) if idx not in curr_indices]
        
        for currIdx in current_parless:
            currChild = currChildren[currIdx]
            
            if currChild == []: ### in this case, the track has no child, and so it terminates here.
                ### it's childless and parentless -> appears and disappears.
                currDict = {'cnt_idx': currIdx, 'child': None}
                curr_track = {frame: currDict}
                curr_trackList.append(curr_track)
            else:
                for child in currChild:
                    currDict = {'cnt_idx': currIdx, 'child': child}
                    curr_track = {frame: currDict}
                    curr_trackList.append(curr_track)  
           
        if super_verbose:
            print(f'   ... done with parentless: took {round(time.time() - chk, 1)} secs')
            chk = time.time()
        
        ### emergency exit
        if len(curr_trackList) > critical_number_of_tracks:
            print(f'Had to exit {plSF} at frame {frame} --> too many tracks! ({len(curr_trackList)})')
            break ### breaking the loop, thereby ensuring that the tracks which are found are not lost, only shortened (at the beginning)
             
        last_trackList = list(curr_trackList)
        del curr_trackList
        
        if super_verbose:
            print(f'   ... done copying: took {round(time.time() - chk, 1)} secs')
            chk = time.time()
              
    if verbose:
        loop_end_time = time.time()
        
    ### keep only the contour index
    small_trackList = []
    for track in last_trackList:
        small_track = {frame: track[frame]['cnt_idx'] for frame in track}
        small_trackList.append(small_track)
    
    del last_trackList
    
    if verbose:
        sort_time = time.time()
        
    serList = [pd.Series(track, name = idx) for idx, track in enumerate(small_trackList)]
    
    if verbose:
        gather_time = time.time()
    
    # trackSF = SFList(stream_features).join()
    trackDF = pd.concat(serList, axis = 1)
    
    if verbose:
        end_time = time.time()
        
        time_before_loop = round(loop_begin_time - start_time, 1)
        time_in_loop = round(loop_end_time - loop_begin_time, 1)
        time_to_sort = round(sort_time - loop_end_time, 1)
        time_to_gather = round(gather_time - sort_time, 1)
        time_to_join = round(end_time - gather_time, 1)
        time_total = round(end_time - start_time, 1)
        
        print(f'{plSF}: Found {len(small_trackList)} tracks.')
        print(f'Took {time_total} seconds:')
        print(f'   Time spent before loop: {time_before_loop} seconds.')
        print(f'   Time spent in loop: {time_in_loop} seconds.')
        print(f'   Time spent sorting: {time_to_sort} seconds.')
        print(f'   Time spent gathering: {time_to_gather} seconds.')
        print(f'   Time spent joining: {time_to_join} seconds.')
        
    return trackDF
    
    
def split_player_candidate_tracks(plSF, 
                                  frame_offset = 1,
                                  critical_number_of_tracks = 2000,
                                  verbose = False, super_verbose = False):

    if verbose: 
        start_time = time.time()

    ### Clear up offset
    frames = plSF.get_frames()
    frames_of_interest = frames[::frame_offset]
    rev_frames_of_interest = list(reversed(frames_of_interest))
    lastFrm, other_frames = rev_frames_of_interest[0], rev_frames_of_interest[1:]
    
    ### get data from last frame
    lastDict = plSF[lastFrm]
    lastPars = lastDict['player_parents']
    
    ### initialize list of tracks
    last_trackList = []
    for currIdx, lastPar in enumerate(lastPars):
        if lastPar == []:
            trackDict = {'cnt_idx': currIdx, 'parent': None}
            last_trackList.append({lastFrm: trackDict})
        else:
            for par in lastPar:
                trackDict = {'cnt_idx': currIdx, 'parent': par}
                last_trackList.append({lastFrm: trackDict})
         
    if verbose:
        loop_begin_time = time.time()

    for frame in other_frames:
        
        if super_verbose:
            chk = time.time()
            print(f'Frame Number: {frame} (until {plSF.first()[0]})')
            print(f'Number of Tracks: {len(last_trackList)}')
        
        featDict = plSF[frame]
        currPars = featDict['player_parents']
        
        curr_indices = []
        curr_trackList = [] ### this list of tracks is populated in the loop
        
        for last_track in last_trackList:
            
            ### Handle previously finished tracks ###
            try:
                childDict = last_track[frame + frame_offset]
            except KeyError: 
                ### in this case, the track was previously completed from end to beginning, so we add it in unmodified
                curr_trackList.append(last_track)
                continue
            
            currIdx = childDict['parent']
            
            ### Handle just finished tracks ###
            if currIdx is None: ### in this case, the track is just completed, so we add it in unmodified.
                curr_trackList.append(last_track)
                continue
            else:
                curr_indices.append(currIdx)
            
            ### Handle tracks with a continuation in next frame (parents to a child)
            currPar = currPars[currIdx]
            
            if currPar == []: ### in this case, the track has no parent, and so it terminates here (seen backwards; seen forwards, it spawns here).
                # curr_track = copy.deepcopy(last_track)
                curr_track = last_track
                currDict = {'cnt_idx': currIdx, 'parent': None}
                curr_track[frame] = currDict
                curr_trackList.append(curr_track)
                
            else:
                for par in currPar:
                    curr_track = dict(last_track) ### need to do this due to mutability and the loop over parents...
                    currDict = {'cnt_idx': currIdx, 'parent': par}
                    curr_track[frame] = currDict
                    curr_trackList.append(curr_track)
               
        if super_verbose:
            print(f'   ... done with track check: took {round(time.time() - chk, 1)} secs')
            chk = time.time()
        
        ### Handle tracks which are just begun: they are not pointed to by a child (seen forwards, they end here!)
        current_childless = [idx for idx in range(len(currPars)) if idx not in curr_indices]
        
        for currIdx in current_childless:
            currPar = currPars[currIdx]
            
            if currPar == []: ### in this case, the track has no parent, and so it terminates here (seen backwards; seen forwards, it spawns here).
                ### it's childless and parentless -> appears and disappears.
                currDict = {'cnt_idx': currIdx, 'parent': None}
                curr_track = {frame: currDict}
                curr_trackList.append(curr_track)
            else:
                for par in currPar:
                    currDict = {'cnt_idx': currIdx, 'parent': par}
                    curr_track = {frame: currDict}
                    curr_trackList.append(curr_track)  
              
        if super_verbose:
            print(f'   ... done with childless: took {round(time.time() - chk, 1)} secs')
            chk = time.time()
        
        ### emergency exit
        if len(curr_trackList) > critical_number_of_tracks:
            print(f'Had to exit {plSF} at frame {frame} --> too many tracks! ({len(curr_trackList)})')
            break ### breaking the loop, thereby ensuring that the tracks which are found are not lost, only shortened (at the beginning)
    
        last_trackList = list(curr_trackList)
        del curr_trackList
        
        if super_verbose:
            print(f'   ... done copying: took {round(time.time() - chk, 1)} secs')
            chk = time.time()
    
    if verbose:
        loop_end_time = time.time()
        
    ### invert the ordering of the track lists, keep only the contour index
    sort_trackList = []
    for track in last_trackList:
        rev_track = {frame: track[frame]['cnt_idx'] for frame in reversed(track)}
        sort_trackList.append(rev_track)
    
    del last_trackList
    
    if verbose:
        sort_time = time.time()
        
    ### gather the contours
    # cntSF = plSF['player_candidates']
    # stream_features = [StreamFeature({frame: cntSF[frame][cnt_idx] for frame, cnt_idx in track.items()}, name = f'track_{idx}') for idx, track in enumerate(sort_trackList)]
    ### gather only the indices
    # stream_features = [StreamFeature(track, name = f'track_{idx}') for idx, track in enumerate(sort_trackList)]
    serList = [pd.Series(track, name = idx) for idx, track in enumerate(sort_trackList)]
    
    if verbose:
        gather_time = time.time()
    
    # trackSF = SFList(stream_features).join()
    trackDF = pd.concat(serList, axis = 1)
    
    if verbose:
        end_time = time.time()
        
        time_before_loop = round(loop_begin_time - start_time, 1)
        time_in_loop = round(loop_end_time - loop_begin_time, 1)
        time_to_sort = round(sort_time - loop_end_time, 1)
        time_to_gather = round(gather_time - sort_time, 1)
        time_to_join = round(end_time - gather_time, 1)
        time_total = round(end_time - start_time, 1)
        
        print(f'{plSF}: Found {len(sort_trackList)} tracks.')
        print(f'Took {time_total} seconds:')
        print(f'   Time spent before loop: {time_before_loop} seconds.')
        print(f'   Time spent in loop: {time_in_loop} seconds.')
        print(f'   Time spent sorting: {time_to_sort} seconds.')
        print(f'   Time spent gathering: {time_to_gather} seconds.')
        print(f'   Time spent joining: {time_to_join} seconds.')
    
    return trackDF
    
def compare_tracklists(last_trackList, curr_trackList, frame): 
    ### a little function to diagnose issues with the above function
    last_trackList[40]
    curr_trackList[40]
    
    cut_trackList = [{f: itemDict for f, itemDict in trackDict.items() if not f == frame} for trackDict in curr_trackList]
    alive_trackList = [trackDict for trackDict in curr_trackList if frame in trackDict]
    
    checkDict = {}
    for orig_idx, orig_track in enumerate(last_trackList):
        new_idxs = [idx for idx, cut_track in enumerate(cut_trackList) if cut_track == orig_track]
        checkDict[orig_idx] = new_idxs
    
    
    extra_tracks = [t for t in cut_trackList if not t in last_trackList]
    len(extra_tracks)
    len(alive_trackList)
    
    frame
    

def evaluate_player_candidates(candSF, splitDF, ### this needs to be passed to apply_starmap method of  SFList!
                               granularity = 10,
                               # max_meter_per_second = 8
                               # fps = 25,
                               area_outlier_iqr_multiple = 1.5,
                               overlap_optimum = 0.9,
                               overlap_strictness = 5,
                               metric_weights = {'dist': 5, 'ovlp': 2, 'area': 1, 'move': 2},
                               ):
    
    ### get court for top and bottom baselines
    courtSF = candSF['courtLines']
    ### get the court roi for maximum acceptable distances from lines
    ROISF = candSF['courtroiReg']
    ### get the candidates
    cntSF = candSF['player_candidates']
    ### get the top and bot points at serve/return
    try:
        topSF, botSF = candSF['topPt'], candSF['botPt']
    except KeyError:
        topSF, botSF = StreamFeature({}), StreamFeature({})
        
    
    ### some auxiliary functions
    def beta(x, optimum, strictness):
        ### Taking in x between 0 and 1, and optima between 0 and 1 and a positive real valued strictness, yields a beta distribution.
        ### the maximum occurs at optimum parameter; strictness indicates how fast the distribution drops around strictness
        ### the beta distribution is then scaled so that the f(optimum) = 1. f(0) = f(1) = 0, if optimum in (0, 1).
        opt, opt_c = optimum, 1 - optimum
        x_c = 1 - x
        
        y = (x ** (opt * strictness)) * (x_c ** (opt_c * strictness)) 
        norm = (opt ** (opt * strictness)) * (opt_c ** (opt_c * strictness)) 
        
        return y / norm
        
    from scipy.stats import iqr
    
    ### some auxiliary variables
    # max_meter_per_frame = max_meter_per_second / fps
    max_conceivable_ratio_outliers = 0.5 / area_outlier_iqr_multiple #### in theory, up to 50% of observations could be outliers, 
                                                                        # but we scale it by the multiple, to make the metric more sensitive.                                              
    
    all_frames = splitDF.index ### all frames, but since there can be an offset, or the track splitting can terminate early (due to too many tracks...), we use splitDF
    all_tracks = splitDF.columns 
    
    ### Build Look-Up Tables
    ## -> this is to speed up the process: we don't want to recalculate stuff for every track that we already know! 
    ## since there can be upwards of 10,000 tracks!
    
    # a) Frame Dependent, Contour Independent Quantities
    FDCI_LUT = {}
    for frame in all_frames:
        ### to calculate normalized distance to baselines...
        
        # a) get points on the baselines 
        topLn, botLn = courtSF[frame]['Top Baseline'], courtSF[frame]['Bottom Baseline']
        topPts, botPts = topLn.subdivide(granularity), botLn.subdivide(granularity) ### sadly, have to divide the line, because 
                                                                                    ### distance between line and point uses weak_intersect, which extends line
        # b) get points at maximum acceptable distance
        roitopLn, roibotLn = ROISF[frame]['Top'], ROISF[frame]['Bottom']
        roitopPt, roibotPt = roitopLn.center(), roibotLn.center()
        
        # c) calculate maximum acceptable distance
        top_dist_max = roitopPt.distance_to(topLn.center())
        bot_dist_max = roibotPt.distance_to(botLn.center())
    
        FDCI_LUT[frame] = {'topPts': topPts, 'botPts': botPts,
                           'top_dist_max': top_dist_max, 'bot_dist_max': bot_dist_max}
    
    # b) Frame Dependent, Contour Dependent Quantities
    FDCD_LUT = {}
    for frame in all_frames:
        frameCnts = cntSF[frame]
        top_srvPt, bot_srvPt = topSF.maybe_get(frame), botSF.maybe_get(frame) ### points at serve and return frames
        
        top_dist_norms = []
        bot_dist_norms = []
        top_serve_dist_norms = []
        bot_serve_dist_norms = []
        areas = []
        for frameCnt in frameCnts:
            
            top_dist_max = FDCI_LUT[frame]['top_dist_max']
            bot_dist_max = FDCI_LUT[frame]['bot_dist_max']
            
            ### to calculate normalized distance to baselines...
            # a) construct a line traversing the contour from top to bottom 
            cntReg = frameCnt.enclosing_region()
            cntLn = Line(cntReg['Top'].center(), cntReg['Bottom'].center())
            cntPts = cntLn.subdivide(granularity)
            
            # b) calculate distances to points on baselines
            topPts = FDCI_LUT[frame]['topPts']
            botPts = FDCI_LUT[frame]['botPts']
            top_dist = min([topPt.distance_to(cntPt) for cntPt in cntPts for topPt in topPts])
            bot_dist = min([botPt.distance_to(cntPt) for cntPt in cntPts for botPt in botPts])
            
            top_dist_norm = top_dist / top_dist_max
            bot_dist_norm = bot_dist / bot_dist_max
            
            top_dist_norms.append(top_dist_norm)
            bot_dist_norms.append(bot_dist_norm)
            
            # c) calculate distance to points during serve
            if top_srvPt is not None:
                top_serve_dist = min([top_srvPt.distance_to(cntPt) for cntPt in cntPts])
                top_serve_dist_norm = top_serve_dist / top_dist_max
                top_serve_dist_norms.append(top_serve_dist_norm)
                
            if bot_srvPt is not None:
                bot_serve_dist = min([bot_srvPt.distance_to(cntPt) for cntPt in cntPts])
                bot_serve_dist_norm = bot_serve_dist / bot_dist_max
                bot_serve_dist_norms.append(bot_serve_dist_norm)
            
            ### to calculate area...
            area = frameCnt.area()
            areas.append(area)
            
        FDCD_LUT[frame] = {'top_dist_norm': top_dist_norms,
                           'bot_dist_norm': bot_dist_norms,
                           'top_serve_dist_norm': top_serve_dist_norms,
                           'bot_serve_dist_norm': bot_serve_dist_norms,
                           'areas': areas}
        
    # c) Frame Dependent, Contour Dependent, Previous Contour Dependent Quantities
    FDCDPD_LUT = {}
    prev_frameCnts = None
    for frame in all_frames:
        frameCnts = cntSF[frame]
        
        ovlp_array = []
        nstill_array = []
        
        for frameCnt in frameCnts:
            
            if prev_frameCnts is None:
                continue
            
            ovlp_row = []
            nstill_row = []
            for prev_frameCnt in prev_frameCnts:
                ovlp = frameCnt.overlap_factor(prev_frameCnt, relative_to = 'self')
                not_still = frameCnt != prev_frameCnt
                    
                ovlp_row.append(ovlp)
                nstill_row.append(not_still)
                
            ovlp_array.append(ovlp_row)
            nstill_array.append(nstill_row)
            
        FDCDPD_LUT[frame] = {'ovlp': np.array(ovlp_array),
                             'nstill': np.array(nstill_array)}
        
        prev_frameCnts = frameCnts
    
    ### For each track, gather the correct metrics from the LUTs.
    metricSers = []
    for track in all_tracks:
        trackSer = splitDF[track].dropna().astype(int)
        
        if len(trackSer) < 2: ### single frame tracks not admissible!
            continue
        
        top_dists = []
        bot_dists = []
        top_serve_dists = []
        bot_serve_dists = []
        ovlps = []
        areas = []
        nstills = []
        
        prev_frame, track_frames = trackSer.index[0], trackSer.index[1:]
        
        for frame in track_frames:
            
            ### figure out relevant track indices!
            frame_idx = trackSer[frame]
            prev_frame_idx = trackSer[prev_frame]
            
            ### get LUT entries for specific frame
            FDCD_entry = FDCD_LUT[frame]
            FDCDPD_entry = FDCDPD_LUT[frame]
            
            ### gather distances
            top_dist_norm = FDCD_entry['top_dist_norm'][frame_idx]
            bot_dist_norm = FDCD_entry['bot_dist_norm'][frame_idx]
            top_dists.append(top_dist_norm)
            bot_dists.append(bot_dist_norm)
            
            if FDCD_entry['top_serve_dist_norm'] != []:
                top_serve_dist_norm = FDCD_entry['top_serve_dist_norm'][frame_idx]
                top_serve_dists.append(top_serve_dist_norm)
            
            if FDCD_entry['bot_serve_dist_norm'] != []:
                bot_serve_dist_norm = FDCD_entry['bot_serve_dist_norm'][frame_idx]
                bot_serve_dists.append(bot_serve_dist_norm)
            
            ### gather areas
            area = FDCD_entry['areas'][frame_idx]   
            areas.append(area)
            
            ### gather overlaps
            ovlp = FDCDPD_entry['ovlp'][frame_idx, prev_frame_idx]
            ovlps.append(ovlp)
            
            ### gather not-stills
            nstill = FDCDPD_entry['nstill'][frame_idx, prev_frame_idx]
            nstills.append(nstill)
            
            prev_frame = frame
            
        ### Build the Evaluation Metrics!
        metricDict = {}
        
        ### if the player is around the baseline at least 50% of the time, this should be close to one! 
        # otherwise, it can get very small, even substantially less than zero! 
        metricDict['top_dist_score'] = 1 - np.median(np.array(top_dists) ** 2) 
        metricDict['bot_dist_score'] = 1 - np.median(np.array(bot_dists) ** 2)
        
        metricDict['top_dist_time_score'] = (np.array(top_dists) < 0.5).mean() ### time spent close to the baseline
        metricDict['bot_dist_time_score'] = (np.array(bot_dists) < 0.5).mean() ### time spent close to the baseline
        
        ### if the player is at the serving/returning spot when serve and return had previously been detected, this should be close to one
        if not top_serve_dists == []: ### this occurs if track is not online during serve/return
            metricDict['top_dist_serve_score'] = 1 - np.median(np.array(top_serve_dists) ** 2)
        else:
            metricDict['top_dist_serve_score'] = 0
        if not bot_serve_dists == []:
            metricDict['bot_dist_serve_score'] = 1 - np.median(np.array(bot_serve_dists) ** 2)
        else:
            metricDict['bot_dist_serve_score'] = 0
        
        ### if the player is moving fast, this should be closer to one!
        # metricDict['top_speed_score'] = np.mean(np.array(top_speeds) ** 2) ### using the median here causes many zeroes and few positive values.
        # metricDict['bot_speed_score'] = np.mean(np.array(bot_speeds) ** 2)  ## especially for the top player, this could make things hard to distinguish...
                                                                                ## also, the typical tendency to move is captured by the still_score!
                         
        ### if the player is moving adequately, the overlap should be close to some optimum value
        ### -> make a beta distribution, which maps 
        metricDict['ovlp_score'] = np.mean(beta(np.array(ovlps), overlap_optimum, overlap_strictness))              
                         
        ### if the contour is not too jittery, closer to one! If it is super jittery, it can be smaller than zero! 
        # metricDict['area_score'] = 1 - (np.std(areas) / np.mean(areas))
        ### since this is kind of a tie breaker, we want to go into the extremes -> how many outliers are there?
        med_area = np.median(areas)
        out_range = area_outlier_iqr_multiple * iqr(areas)
        lo, hi = med_area - out_range, med_area + out_range
        outliers = [a for a in areas if not lo < a < hi]
        metricDict['area_score'] = 1 - (len(outliers) / (max_conceivable_ratio_outliers * len(areas)))
        
        ### if the contour is moving at least a bit, this should be close to 1!
        metricDict['move_score'] = np.mean(nstills)
        
        ### the longer the contour is on screen, the larger this metric
        metricDict['screen_score'] = len(trackSer) / len(all_frames)
        
        metricSers.append(pd.Series(metricDict, name = track))
        
    metricDF = pd.concat(metricSers, axis = 1).T
        
    ### Evaluate all tracks according to metrics, and the weights assigned
    
    # a) gather the metrics that enter linearly
    topDF = metricDF[['top_dist_score', 'top_dist_time_score', 'top_dist_serve_score', 'ovlp_score', 'move_score', 'area_score']]
    topDF.columns = [col.replace('top_', '').replace('_score', '') for col in topDF.columns]
    
    botDF = metricDF[['bot_dist_score', 'bot_dist_time_score', 'bot_dist_serve_score', 'ovlp_score', 'move_score', 'area_score']]
    botDF.columns = [col.replace('bot_', '').replace('_score', '') for col in botDF.columns]
    
    # b) the proportion of frames that a player can be seen in enters multiplicatively!
    screen_mult = metricDF['screen_score']
    
    # c) evaluate the aggregate metric of each track, for top and bot player
    weightSer = pd.Series(metric_weights)
    
    topSer = (topDF * weightSer).sum(axis = 1) * screen_mult
    botSer = (botDF * weightSer).sum(axis = 1) * screen_mult
    
    # topDF['eval'] = topSer
    # botDF['eval'] = botSer
    
    top_choice = topSer.index[topSer.argmax()]
    bot_choice = botSer.index[botSer.argmax()]
        
    #### only now do we actually build the track!!!
    topidxSer = splitDF[top_choice].dropna().astype(int)
    botidxSer = splitDF[bot_choice].dropna().astype(int)
    topSF = StreamFeature({frame: cntSF[frame][topidxSer[frame]] for frame in topidxSer.index}, name = 'topPlayerCnt')
    botSF = StreamFeature({frame: cntSF[frame][botidxSer[frame]] for frame in botidxSer.index}, name = 'botPlayerCnt')
    
    return topSF.merge(botSF, how = 'outer')
    
def interpolate_players(playerSF, frame_offset = 5, which = 'top',
                        minimum_overlap_between_candidate_and_interpolation_position = 0.1):
    candSF = playerSF['foreCnts']
    plSF = playerSF[f'{which}PlayerCnt']
    
    outDict = plSF.streamDict
    
    for left_frame in plSF.dropna():
        leftCnt = plSF[left_frame]
        
        right_frame = left_frame + frame_offset
        rightCnt = plSF.maybe_get(right_frame)
        if rightCnt is None: ### in this case, there is no right player, so we don't interpolate.
            continue
        
        for itpl_frame in range(1, frame_offset): ### itpl points: 1, 2, ..., FRAME_OFFSET - 1  
            itpl_lambda = itpl_frame / frame_offset      
            abs_frame = left_frame + itpl_frame
    
            candCnts = candSF[abs_frame]
            min_ovlp = minimum_overlap_between_candidate_and_interpolation_position
            itplCnt = None

            for candCnt in candCnts:
                ovlp = (1 - itpl_lambda) * candCnt.overlap_factor(leftCnt) + itpl_lambda * candCnt.overlap_factor(rightCnt)
                if ovlp > min_ovlp:
                    itplCnt = candCnt
                    min_ovlp = ovlp
                    
            outDict[abs_frame] = itplCnt
            
    return StreamFeature(outDict, name = plSF.name)
                



def find_ball_candidates(contSF, svmDict = None, use_training_moments = False,
                         probability_threshold = None):
    ballTable = contSF['ballTable'].to_frame()
    
    if len(ballTable) == 0:
        return StreamFeature({}, name = 'candCnts') 

    ### get the SVM and the features used
    svm = svmDict['svm']
    features = svmDict['features']
    
    ### figure out normalising moments
    if use_training_moments:
        mean = svmDict['mean']
        std = svmDict['std']
    else:
        mean = ballTable.mean()
        std = ballTable.std()
        
    normTable = (ballTable - mean) / std
    
    ### Using the SVM estimated, predict whether a contour is a ball.
    normTable['isBall_proba'] = svm.predict_proba(normTable[features].values)[:,1]
    if probability_threshold is None:
        normTable['isBall'] = svm.predict(normTable[features].values)
    else:
        normTable['isBall'] = (normTable['isBall_proba'] > probability_threshold).astype(int)
    
    ### get all True predictions
    trueTable = normTable[normTable.isBall == 1]
    
    trueDict = {}
    for frmNum, cnts in contSF['movCnts'].items():
        try:
            myTable = trueTable.loc[frmNum]
        except KeyError:
            trueDict[frmNum] = None
        else:
            true_contours = []
            for true_idx in myTable.index:
                true_contours.append(cnts[true_idx])
                
            if true_contours == []:
                trueDict[frmNum] = None
            else:
                trueDict[frmNum] = Contours(true_contours)

    return StreamFeature(trueDict, name = 'candCnts')

def find_ball_candidates_tiered(contSF, svmDict = None, use_training_moments = False,
                                threshold_low = 0.025, threshold_mid = 0.1, threshold_hi = None):
    ballTable = contSF['ballTable'].to_frame()
    
    if len(ballTable) == 0:
        return StreamFeatures({})

    ### get the SVM and the features used
    svm = svmDict['svm']
    features = svmDict['features']
    
    ### figure out normalising moments
    if use_training_moments:
        mean = svmDict['mean']
        std = svmDict['std']
    else:
        mean = ballTable.mean()
        std = ballTable.std()
        
    normTable = (ballTable - mean) / std
    
    ### Using the SVM estimated, predict whether a contour is a ball.
    normTable['isBall_proba'] = svm.predict_proba(normTable[features].values)[:,1]

    trueDict = {frame: {} for frame in contSF.get_frames()}
    for tier_idx, probability_threshold in enumerate([threshold_low, threshold_mid, threshold_hi]):

        if probability_threshold is None:
            normTable['isBall'] = svm.predict(normTable[features].values)
        else:
            normTable['isBall'] = (normTable['isBall_proba'] > probability_threshold).astype(int)
        
        ### get all True predictions
        trueTable = normTable[normTable.isBall == 1]
        
        for frmNum, cnts in contSF['movCnts'].items():
            try:
                myTable = trueTable.loc[frmNum]
            except KeyError:
                trueDict[frmNum][f'cand{tier_idx}Cnts'] = None
            else:
                true_contours = []
                for true_idx in myTable.index:
                    true_contours.append(cnts[true_idx])
                    
                if true_contours == []:
                    trueDict[frmNum][f'cand{tier_idx}Cnts'] = None
                else:
                    trueDict[frmNum][f'cand{tier_idx}Cnts'] = Contours(true_contours)

    return StreamFeatures(trueDict)


def find_particles(mySF, 
                   dims = (1920, 1080), pixel_in_meter = (200, 100), N = 1000, fps = 25,
                   normal_propagation = None, close_propagation = None, switch_probability = None):

    import ATMANpy.funcs.particle_funcs_vectorized as pf
    
    indexHistories = None
    closeHistories = None
    modelHistories = None
    particles = pf.initialize_particles(dims, pixel_in_meter = pixel_in_meter, N = N, fps = fps)
    
    partDict = {} ### we need particles, first and foremost
    histDict = {} ### but we also need the final index histories, for smoothing
    for frame, featDict in mySF.items():
        ### write to frame!
        partDict[frame] = particles.astype(int) ### for memory reasons, save as integer
        
        chamferCnts = featDict['chamferCnts']
        ballContours = featDict['candCnts']
        
        propagation_models, closeBools, modelBools = pf.get_motion_model_with_contours(particles, chamferCnts,
                                                                         closeHistories = closeHistories,
                                                                         modelHistories = modelHistories,
                                                                         normal_propagation = normal_propagation, 
                                                                         close_propagation = close_propagation, 
                                                                         switch_probability = switch_probability)    
    
        ########### UPDATE PARTICLES #########
        mDict, wDict = pf.get_means_and_weights(particles, propagation_models, ballContours) ### means_and_weights updates last periods particles, and compares to evidence in this period!
        particles, particleIdxs, _ = pf.resample(mDict, wDict)
        
        ########### UPDATE HISTORIES #########
        indexHistories, closeHistories, modelHistories = pf.update_histories(indexHistories, closeHistories, modelHistories,
                                                                             particleIdxs, closeBools, modelBools) 

    partDict[frame + 1] = particles.astype(int) ### for memory reasons, save as integer
    histDict[frame + 1] = indexHistories ### these are only needed at the very end, 
                                           # since indexHistories update and accumulate
    
    partSF = StreamFeature(partDict, name = 'particles')
    histSF = StreamFeature(histDict, name = 'idxHistories')
    return partSF.merge(histSF)
    

def kitagawa_smooth(mySF):    
    ##### Let's go backwards.
    ## -> u = T - t - 1: Essentially, we now kitagawa-smooth from recent to old.
    ## this is easier since we can get the direct parent recursively.
        
    last_frame, idx_history = mySF['idxHistories'].last()
    T, N = idx_history.shape
    
    # initialize for last period
    k_iT_u = np.arange(start = 0, stop = N)
    
    smoothDict = {}
    for u in range(T):
        curr_frame = last_frame - u
        
        ### select the particles based on previous indices
        s_i = mySF['particles'][curr_frame]
        selected_s_i = s_i[k_iT_u]
        
        smoothDict[curr_frame] = selected_s_i
        
        ### select the corresponding indices based on previous indices
        parentIdxs = idx_history[u]
        k_iT_u = parentIdxs[k_iT_u]
    
    ### reverse the dictionary, so frames are in order
    smoothDict = {frame: smoothDict[frame] for frame in sorted(smoothDict)}
    
    return StreamFeature(smoothDict, name = 'smooth_particles')


def calculate_contiguous_movement(mySF):
    ### Takes: StreamFeature(..., name = 'largeCnts') -> large movement contours
    
    ### INIT ###
    current_movement_index = 0
    
    idxDict = {}
    
    for frmNum, largeCnts in mySF.items():
        
        if largeCnts.length() > 0:
            current_movement_index += 1
        else:
            current_movement_index = 0
            
        idxDict[frmNum] = current_movement_index 
    
    return StreamFeature(idxDict, name = 'contiguous_movement')
    
def drop_movement_frames(mySF, 
                         minimum_length_of_contiguous_movement = 10, 
                         keep_first_and_last_x_frames = 3):
    ### Takes: StreamFeature(idxDict, name = 'contiguous_movement')
    
    ### INIT ###
    prior_movement_index = 0
    frames_to_toss = [] ### the overall list of lists, containing all continuous segments of extraneous movement on court.
    current_frames_to_toss = [] ### a list containing a single contiguous segment of extraneous movement.
    
    for frmNum, current_movement_index in mySF.items():
        
        if current_movement_index > 0:
            current_frames_to_toss.append(frmNum)
            prior_movement_index = current_movement_index
        else:
            if prior_movement_index > minimum_length_of_contiguous_movement:
                ### keep first and last few frames
                start_idx = keep_first_and_last_x_frames
                stop_idx = - keep_first_and_last_x_frames
                current_frames_to_toss = current_frames_to_toss[start_idx:stop_idx]
                ### add to list of lists
                frames_to_toss.append(current_frames_to_toss)
            
            current_frames_to_toss = []
            prior_movement_index = 0
            
    frames_to_toss.append(current_frames_to_toss) ### have to do one final append!
            
    flat_frames_to_toss = [el for ls in frames_to_toss for el in ls]
    
    return BoolFeature({frame: frame in flat_frames_to_toss for frame in mySF.get_frames()}, name = 'frames_to_toss')
 

def find_next_item(mySF, startNum = None):
    frames = mySF.get_frames()
    
    if startNum is None:
        startNum = frames[0]
    
    for frmNum, item in mySF.items():
        if frmNum < startNum:
            continue
        
        if item is None:
            continue
        
        return frmNum, item
    
    return None, None

def find_last_item(mySF, startNum = None):
    frames = mySF.get_frames()

    if startNum is None:
        startNum = frames[-1]
        
    for frmNum in sorted(frames, reverse = True):
        if frmNum > startNum:
            continue
        
        item = mySF[frmNum]
        if item is None:
            continue
        
        return frmNum, item
    
    return None, None

def find_next_N_items(mySF, N = 3, startNum = None): 
    frmNum, item = find_next_item(mySF, 
                                  startNum = startNum)
    
    if not frmNum is None:
        outList = [item]
        frame_count = frmNum
        while len(outList) < N:
            frame_count += 1
            add_item = mySF.maybe_get(frame_count)
            outList.append(add_item)
        return frmNum, outList
    else:
        return None, [None] * N

def find_last_N_items(mySF, N = 3, startNum = None):
    frmNum, item = find_last_item(mySF, 
                                  startNum = startNum)

    if not frmNum is None:
        outList = [item]
        frame_count = frmNum
        while len(outList) < N:
            frame_count -= 1
            add_item = mySF.maybe_get(frame_count)
            outList.append(add_item)
        return frmNum, outList
    else:
        return None, [None] * N    

def get_positions_speeds_and_accelerations_for_interpolation(ballSF, itpl_distance = 25):
    
    def calculate_speeds_and_acceleration(cnt1, cnt2, cnt3, frm1, frm2, frm3):
        ### given three contours, calculate their speed and acceleration per frame
        try:
            speed1 = (cnt2.center() - cnt1.center()) / (frm2 - frm1)
        except:
            speed1 = None
        
        try:
            speed2 = (cnt3.center() - cnt2.center()) / (frm3 - frm2)
        except:
            speed2 = None
        
        try:
            accel = speed2 - speed1
        except:
            accel = None
            
        return speed1, speed2, accel
    
    outDict = {}
    
    ### Get the stretches of play without a ball --> these will have to be interpolated
    emptySFs = ballSF.filter(ballSF.isna()).split()

    for emptySF in emptySFs:
        T = emptySF.length()
        
        if T > itpl_distance:
            segDict = {frame: None for frame in emptySF}
        
        ### find first and last frames of empty stretch
        fsNum, _ = emptySF.first()
        lsNum, _ = emptySF.last()
        
        ##### DETERMINE PREVIOUS BALL POSITIONS
        prevNum, prevCnt = find_last_item(ballSF, startNum = fsNum)   
        
        if prevCnt is None or fsNum - prevNum > itpl_distance:
            prevNum, prevCnt     = None, None ### being over itpl distance is the same as being None 
            prevNum_1, prevCnt_1 = None, None ### in this case, we have no last known position, speed and acceleration
            prevNum_2, prevCnt_2 = None, None                 
        else:
            prevNum_1, prevCnt_1 = find_last_item(ballSF, startNum = prevNum - 1)
        
            if prevCnt_1 is None or fsNum - prevNum_1 > itpl_distance:
                prevNum_1, prevCnt_1 = None, None ### being over itpl distance is the same as being None 
                prevNum_2, prevCnt_2 = None, None ### in this case, we have no speed and acceleration
            else:    
                prevNum_2, prevCnt_2 = find_last_item(ballSF, startNum = prevNum_1 - 1)
        
                if prevCnt_2 is None or fsNum - prevNum_2 > itpl_distance:
                    prevNum_2, prevCnt_2 = None, None  ### being over itpl distance is the same as being None 
                                                 ### in this case, we have no acceleration
            
        ##### DETERMINE NEXT BALL POSITIONS
        nextNum, nextCnt = find_next_item(ballSF, startNum = lsNum)   
        
        if nextCnt is None or nextNum - lsNum > itpl_distance:
            nextNum, nextCnt     = None, None ### being over itpl distance is the same as being None 
            nextNum_1, nextCnt_1 = None, None ### in this case, we have no last known position, speed and acceleration
            nextNum_2, nextCnt_2 = None, None                 
        else:
            nextNum_1, nextCnt_1 = find_next_item(ballSF, startNum = nextNum + 1)
        
            if nextCnt_1 is None or nextNum_1 - lsNum > itpl_distance:
                nextNum_1, nextCnt_1 = None, None ### being over itpl distance is the same as being None 
                nextNum_2, nextCnt_2 = None, None ### in this case, we have no speed and acceleration
            else:    
                nextNum_2, nextCnt_2 = find_next_item(ballSF, startNum = nextNum_1 + 1)
        
                if nextCnt_2 is None or nextNum_2 - lsNum > itpl_distance:
                    nextNum_2, nextCnt_2 = None, None ### being over itpl distance is the same as being None 
                                                      ### in this case, we have no acceleration
    
        ##### DETERMINE SPEEDS AND ACCELERATION
        _, speed_before, accel_before = calculate_speeds_and_acceleration(prevCnt_2, prevCnt_1, prevCnt, prevNum_2, prevNum_1, prevNum)
        speed_after, _, accel_after = calculate_speeds_and_acceleration(nextCnt, nextCnt_1, nextCnt_2, nextNum, nextNum_1, nextNum_2)

        ##### TURN POINTS (i.e. VECTORS) INTO SERIES
        try:
            p_b = prevCnt.center()
            p_b_x, p_b_y = p_b.x, p_b.y
        except AttributeError:
            p_b_x, p_b_y = None, None
            
        try:
            p_a = nextCnt.center()
            p_a_x, p_a_y = p_a.x, p_a.y
        except AttributeError:
            p_a_x, p_a_y = None, None 
           
        try:
            s_b_x, s_b_y = speed_before.x, speed_before.y
        except AttributeError:
            s_b_x, s_b_y = None, None
        
        try:
            s_a_x, s_a_y = speed_after.x, speed_after.y
        except AttributeError:
            s_a_x, s_a_y = None, None
            
        try:
            a_b_x, a_b_y = accel_before.x, accel_before.y
        except AttributeError:
            a_b_x, a_b_y = None, None
        
        try:
            a_a_x, a_a_y = accel_after.x, accel_after.y
        except AttributeError:
            a_a_x, a_a_y = None, None
            
        
        #### WRITE TO DICTIONARY
        frameDict = {'p_b_x': p_b_x, 
                     'p_b_y': p_b_y,
                     's_b_x': s_b_x,
                     's_b_y': s_b_y,
                     'a_b_x': a_b_x,
                     'a_b_y': a_b_y,
                     'p_a_x': p_a_x,
                     'p_a_y': p_a_y,
                     's_a_x': s_a_x,
                     's_a_y': s_a_y,
                     'a_a_x': a_a_x,
                     'a_a_y': a_a_y,
                     'T': T,
                     }

        segDict = {frame: frameDict for frame in emptySF}
        
        outDict = {**outDict, **segDict}
        
    return StreamFeatures(outDict)

def even_smarter_interpolate_ball_contours(ballSF, itpl_distance = 25, 
                                           verbose = False, plot = False, cap = None):
    ### TODO: What if there is no acceleration only? Should probably still try to interpolate, no?
    ### TODO: What do we do with the really pathological cases, where the nearest point pair is far away?
        ## idea 1: use player position to enforce a better point? (i don't like this much - more moving parts)
    
    from mpl_toolkits.mplot3d import Axes3D  
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    import matplotlib.pyplot as plt
    
    dataSF = get_positions_speeds_and_accelerations_for_interpolation(ballSF, itpl_distance = itpl_distance)
    dataSFs = dataSF.split()

    ### initialize search grid
    b_min, b_max = 0, 1 ### without the endpoint!
    N = 40
    betas = np.linspace(b_min, b_max, N, endpoint = False)

    sfList = []

    for segSF in dataSFs:
        fsFrm, dataDict = segSF.first() ### the data is constant across segments, so get the first, arbitrarily.
        
        if pd.Series(dataDict).isna().any():
            pass
        else:
            
            if cap is not None:   ### very helpful if puzzled
                ### show the previous and next position, speed and acceleration by making three contours respectively.
                befPt = Point2D((dataDict['p_b_x'], dataDict['p_b_y']))
                befSp = Point2D((dataDict['s_b_x'], dataDict['s_b_y']))
                befAc = Point2D((dataDict['a_b_x'], dataDict['a_b_y'])) 
                
                aftPt = Point2D((dataDict['p_a_x'], dataDict['p_a_y']))
                aftSp = Point2D((dataDict['s_a_x'], dataDict['s_a_y']))
                aftAc = Point2D((dataDict['a_a_x'], dataDict['a_a_y'])) 
                
                befPt2 = befPt - befSp
                befPt3 = befPt2 - (befSp - befAc)
        
                befPt2 = befPt2.round_point()
                befPt3 = befPt3.round_point()
        
                aftPt2 = aftPt + aftSp
                aftPt3 = aftPt2 + aftSp + aftAc 
        
                aftPt2 = aftPt2.round_point()
                aftPt3 = aftPt3.round_point()
        
                gFrm = Stream.from_SF(segSF, cap, target_color_space = 'grey').get(fsFrm, put = False)
                cFrm = gFrm.to_BGR()
                cFrm = cFrm.put(befPt, color = (0, 255, 0)).put(befPt2, color = (0, 200, 0)).put(befPt3, color = (0, 150, 0))
                cFrm = cFrm.put(aftPt, color = (0, 0, 255)).put(aftPt2, color = (0, 0, 200)).put(aftPt3, color = (0, 0, 150))
                # cFrm.crop(Region.from_points(befPt2.up(20).left(20), befPt2.down(20).right(20))).zoom(20).show()
                cFrm.show()
            
            ### Length of empty stretch to be interpolated
            T = dataDict['T']
            
            ##### Family of Lines emanating from Contours BEFORE
            ### initialize Position and Speed Arrays
            posArr_bef = np.zeros((T + 1, N, 2)) # Time, Betas, Coordinates
            spdArr_bef = np.zeros((T + 1, N, 2)) # Time, Betas, Coordinates
            
            posArr_bef[0] = dataDict['p_b_x'], dataDict['p_b_y']
            spdArr_bef[0, :, 0] = dataDict['s_b_x'] + betas * dataDict['a_b_x']
            spdArr_bef[0, :, 1] = dataDict['s_b_y'] + betas * dataDict['a_b_y']
            
            for t in range(1, T + 1):
                posArr_bef[t] = posArr_bef[t-1] + spdArr_bef[t-1]
                spdArr_bef[t, :, 0] = spdArr_bef[t-1, :, 0] + betas ** (t + 1) * dataDict['a_b_x']
                spdArr_bef[t, :, 1] = spdArr_bef[t-1, :, 1] + betas ** (t + 1) * dataDict['a_b_y']
            
            posArr_bef, spdArr_bef = posArr_bef[1:], spdArr_bef[1:] ### cut out first period, which is initial!

            ##### Family of Lines emanating from Contours AFTER
            ### initialize Position and Speed Arrays
            posArr_aft = np.zeros((T + 1, N, 2)) # Time, Betas, Coordinates
            spdArr_aft = np.zeros((T + 1, N, 2)) # Time, Betas, Coordinates
            
            posArr_aft[T] = dataDict['p_a_x'], dataDict['p_a_y']
            spdArr_aft[T, :, 0] = dataDict['s_a_x'] ### note the difference in timing here! consult last (inlaid) page in blue book.
            spdArr_aft[T, :, 1] = dataDict['s_a_y'] ### in short: speed is a forwards thing -> p(T + 2) = p(T + 1) + s(T + 1) holds with these "initial" (really, terminal) conditions.
            
            for t in reversed(range(1, T + 1)):
                spdArr_aft[t-1, :, 0] = spdArr_aft[t, :, 0] - betas ** (t + 1) * dataDict['a_a_x']
                spdArr_aft[t-1, :, 1] = spdArr_aft[t, :, 1] - betas ** (t + 1) * dataDict['a_a_y']
                posArr_aft[t-1] = posArr_aft[t] - spdArr_aft[t-1]
                
            posArr_aft, spdArr_aft = posArr_aft[:-1], spdArr_aft[:-1] ### cut the last period, which is terminal
            
            if cap is not None:
                gFrm = Stream.from_SF(segSF, cap, target_color_space = 'grey').get(fsFrm, put = False)
                cFrm = gFrm.to_BGR()  
                
                sampleArr_bef = posArr_bef[::2,::10]
                sampleArr_aft = posArr_aft[::2,::10]
            
                for idx, tArr in enumerate(sampleArr_bef):
                    color = (0, 255 - idx * 20, 0)
                    for coord in tArr:
                        myPt = Point2D(tuple(coord))
                        cFrm = cFrm.put(myPt, color = color)
                        
                for idx, tArr in enumerate(sampleArr_aft):
                    color = (0, 0, 255 - idx * 20)
                    for coord in tArr:
                        myPt = Point2D(tuple(coord))
                        cFrm = cFrm.put(myPt, color = color)
                        
                cFrm.show()
            
            ##### Calculate sum of squares, and minimize!
            posArr_bef_exp = posArr_bef[:, :, np.newaxis, :] ### expand, so as to allow betas to differ!
            posArr_aft_exp = posArr_aft[:, np.newaxis, :, :] ### expand, so as to allow betas to differ!
            
            sum_squares = ((posArr_bef_exp - posArr_aft_exp) ** 2).sum(axis = -1)
            
            T_star_idx, B_bef_star_idx, B_aft_star_idx = np.argwhere(sum_squares == np.min(sum_squares))[0] ### get the first argmin (should be unique)
            T_star, B_bef_star, B_aft_star = T_star_idx + 1, betas[B_bef_star_idx], betas[B_aft_star_idx]
            
            opt_distance = sum_squares[T_star_idx, B_bef_star_idx, B_aft_star_idx] ** 0.5
            if verbose: 
                print(f'Pixel distance between two lines at meeting point: {round(opt_distance, 3)}')
                print(f'Meeting point at time {T_star} (counting from 1 to {T}), at beta_bef = {B_bef_star} and beta_aft = {B_aft_star}')
            
            if plot:
                LIMIT = 1000
                limit_sum_squares = np.where(sum_squares > LIMIT, np.nan, sum_squares)
                
                ### Plot
                X, Y = np.meshgrid(betas, range(1, T + 1))
                
                fig = plt.figure(figsize = (20, 20))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, limit_sum_squares, cmap='viridis', edgecolor='none')
    
                ax.set_xlabel('B')
                ax.set_ylabel('T')

            ##### Create the interpolated paths, using the optimal model parameter beta_star and the switching point T_star
            meetPt_bef = posArr_bef[T_star_idx, B_bef_star_idx]
            meetPt_aft = posArr_aft[T_star_idx, B_aft_star_idx]
            meetPt = (meetPt_bef + meetPt_aft) / 2
            
            befPts = posArr_bef[:T_star_idx, B_bef_star_idx]
            aftPts = posArr_aft[T_star_idx + 1:, B_aft_star_idx]
            
            ptArr = np.array([*befPts, meetPt, *aftPts]).astype(int)
            
            streamDict = {frame: Point2D(tuple(ptCoords)) for frame, ptCoords in zip(segSF.get_frames(), ptArr)}
            
            sfList.append(StreamFeature(streamDict, name = 'ball'))
        
    itplSF = SFList(sfList).concatenate()
    ### TODO: ITPLSF OBJECTS ARE POINT2D, ballSF OBJECTS MIGHT BE CONTOURS! HARMONIZE.
    ### --> turn ballSF contours to center points.
    
    outSF = ballSF.apply(sa.maybe_center).fillna(itplSF).rename('ball_itplPt')
    
    return outSF

def smarter_interpolate_ball_contours(ballSF, itpl_distance = 25, verbose = False):
    ### Instead of looking at three consecutive prior or past contours, these may now be spaced apart.
    
    ####### AUXILIARY 
    import ATMANpy.funcs.particle_funcs_vectorized as pf
    
    def calculate_speeds_and_acceleration(cnt1, cnt2, cnt3, frm1, frm2, frm3):
        ### given three subsequent contours, calculate their speed and acceleration per frame
        try:
            speed1 = (cnt2.center() - cnt1.center()) / (frm2 - frm1)
        except:
            speed1 = None
        
        try:
            speed2 = (cnt3.center() - cnt2.center()) / (frm3 - frm2)
        except:
            speed2 = None
        
        try:
            accel = speed2 - speed1
        except:
            accel = None
            
        return speed1, speed2, accel
    
    ####### LOOP OVER BALL SF
    
    outDict = {}
    for frmNum, ballCnt in ballSF.items():
        
        if ballCnt is not None:
            outDict[frmNum] = ballCnt
            continue
        
        ##### DETERMINE PREVIOUS BALL POSITIONS
        prevNum, prevCnt = find_last_item(ballSF, startNum = frmNum)   
        
        if prevCnt is None or frmNum - prevNum > itpl_distance:
            prevNum, prevCnt     = None, None ### being over itpl distance is the same as being None 
            prevNum_1, prevCnt_1 = None, None ### in this case, we have no last known position, speed and acceleration
            prevNum_2, prevCnt_2 = None, None                 
        else:
            prevNum_1, prevCnt_1 = find_last_item(ballSF, startNum = prevNum - 1)
        
            if prevCnt_1 is None or frmNum - prevNum_1 > itpl_distance:
                prevNum_1, prevCnt_1 = None, None ### being over itpl distance is the same as being None 
                prevNum_2, prevCnt_2 = None, None ### in this case, we have no speed and acceleration
            else:    
                prevNum_2, prevCnt_2 = find_last_item(ballSF, startNum = prevNum_1 - 1)
        
                if prevCnt_2 is None or frmNum - prevNum_2 > itpl_distance:
                    prevNum_2, prevCnt_2 = None, None  ### being over itpl distance is the same as being None 
                                                 ### in this case, we have no acceleration
            
        ##### DETERMINE NEXT BALL POSITIONS
        nextNum, nextCnt = find_next_item(ballSF, startNum = frmNum)   
        
        if nextCnt is None or nextNum - frmNum > itpl_distance:
            nextNum, nextCnt     = None, None ### being over itpl distance is the same as being None 
            nextNum_1, nextCnt_1 = None, None ### in this case, we have no last known position, speed and acceleration
            nextNum_2, nextCnt_2 = None, None                 
        else:
            nextNum_1, nextCnt_1 = find_next_item(ballSF, startNum = nextNum + 1)
        
            if nextCnt_1 is None or nextNum_1 - frmNum > itpl_distance:
                nextNum_1, nextCnt_1 = None, None ### being over itpl distance is the same as being None 
                nextNum_2, nextCnt_2 = None, None ### in this case, we have no speed and acceleration
            else:    
                nextNum_2, nextCnt_2 = find_next_item(ballSF, startNum = nextNum_1 + 1)
        
                if nextCnt_2 is None or nextNum_2 - frmNum > itpl_distance:
                    nextNum_2, nextCnt_2 = None, None ### being over itpl distance is the same as being None 
                                                      ### in this case, we have no acceleration

        ##### DETERMINE SPEEDS AND ACCELERATION
        _, speed_before, accel_before = calculate_speeds_and_acceleration(prevCnt_2, prevCnt_1, prevCnt, prevNum_2, prevNum_1, prevNum)
        speed_after, _, accel_after = calculate_speeds_and_acceleration(nextCnt, nextCnt_1, nextCnt_2, nextNum, nextNum_1, nextNum_2)

        
        ##### DETERMINE TIME FROM LAST BALL AND TO NEXT BALL
        if prevNum is None or nextNum is None:
            continue
        else:
            time_since_prev_ball = frmNum - prevNum
            time_to_next_ball = nextNum - frmNum
        
        ##### INTERPOLATE
        if speed_before is not None and speed_after is not None:
            interpolation_points = 6
            acceptable_lambda_difference = 3
            itplBall = pf.interpolate_ball_features_smart_circle(prevCnt, nextCnt,
                                                                 speed_before = speed_before, 
                                                                 speed_after = speed_after,
                                                                 accel_before = accel_before,
                                                                 accel_after = accel_after,
                                                                 time_since_prev_ball = time_since_prev_ball,
                                                                 time_to_next_ball = time_to_next_ball,
                                                                 interpolation_points = interpolation_points,
                                                                 acceptable_lambda_difference = acceptable_lambda_difference,
                                                                 verbose = verbose)
        # else:
        #     itplBall = pf.interpolate_ball_features_circle(prevCnt, nextCnt, 
        #                                                    frames_before = time_from_ball,
        #                                                    frames_after = time_to_ball)
        else:
            itplBall = None
        
        outDict[frmNum] = itplBall

    return StreamFeature(outDict, name = 'ballCnt')


def smart_interpolate_ball_contours(ballSF, verbose = False):
    
    import ATMANpy.funcs.particle_funcs_vectorized as pf
    
    def calculate_speeds_and_acceleration(cnt1, cnt2, cnt3):
        ### given three subsequent contours, calculate their speed and acceleration per frame
        try:
            speed1 = cnt2.center() - cnt1.center()
        except:
            speed1 = None
        
        try:
            speed2 = cnt3.center() - cnt2.center()
        except:
            speed2 = None
        
        try:
            accel = speed2 - speed1
        except:
            accel = None
            
        return speed1, speed2, accel
    
    outDict = {}
    for frmNum, ballCnt in ballSF.items():

        if ballCnt is not None:
            outDict[frmNum] = ballCnt
            continue
        
        ### determine the last found ball position
        last_frame_with_ball, last_three_cnts = find_last_N_items(ballSF, N = 3,
                                                                  startNum = frmNum)
        
        prevCnt, prevCnt_1, prevCnt_2 = last_three_cnts
        
        if prevCnt is None:
            outDict[frmNum] = None
            continue
            
        ### determine the next found ball position
        next_frame_with_ball, next_three_ctrs = find_next_N_items(ballSF, N = 3,
                                                                  startNum = frmNum)
        
        nextCnt, nextCnt_1, nextCnt_2 = next_three_ctrs
        
        if nextCnt is None:
            outDict[frmNum] = None
            continue
        
        ### determine speeds and accelerations before and after
        _, speed_before, accel_before = calculate_speeds_and_acceleration(prevCnt_2, prevCnt_1, prevCnt)
        speed_after, _, accel_after = calculate_speeds_and_acceleration(nextCnt, nextCnt_1, nextCnt_2)

        ### determine the time elapsed since the last ball and until the next ball
        time_since_prev_ball = frmNum - last_frame_with_ball
        time_to_next_ball = next_frame_with_ball - frmNum
        
        if speed_before is not None and speed_after is not None:
            itplBall = pf.interpolate_ball_features_smart_circle(prevCnt, nextCnt,
                                                                 speed_before = speed_before, 
                                                                 speed_after = speed_after,
                                                                 accel_before = accel_before,
                                                                 accel_after = accel_after,
                                                                 time_since_prev_ball = time_since_prev_ball,
                                                                 time_to_next_ball = time_to_next_ball,
                                                                 verbose = verbose)
        else:
            itplBall = pf.interpolate_ball_features_circle(prevCnt, nextCnt, 
                                                           time_since_prev_ball = time_since_prev_ball,
                                                           time_to_next_ball = time_to_next_ball)
        outDict[frmNum] = itplBall
        
    return StreamFeature(outDict, name = 'ballCnt')



def get_observation_frame(ballSF, velocity_shift = 3, acceleration_shift = 1):
    import pandas as pd
    
    def get_ball(ballCnt):
        if not ballCnt is None:
            coords = ballCnt.center().coords
        else:
            coords = (np.nan, np.nan)
        return coords
    
    ballFrm = pd.DataFrame.from_dict(ballSF.apply(get_ball).streamDict, orient = 'index', columns = ['x', 'y'])
    
    ###### SPEED #######
    speedFrm = ballFrm.shift(-velocity_shift) - ballFrm.shift(velocity_shift) 
    
    ### help out with nans at beginning and end by using an approximate speed and accel formula
    # a) only using information from earlier (for end of sequences.)
    secondary_speedFrm = (ballFrm - ballFrm.shift(velocity_shift)) * 2
    speedFrm = speedFrm.fillna(secondary_speedFrm)
    
    # b) only using information from later (for beginning of sequences.)
    secondary_speedFrm = (ballFrm.shift(-velocity_shift) - ballFrm) * 2
    speedFrm = speedFrm.fillna(secondary_speedFrm)
    
    #### ACCELERATION ####
    accelFrm = speedFrm.shift(-acceleration_shift) - speedFrm.shift(acceleration_shift)
    
    ### help out with nans at beginning and end by using an approximate speed and accel formula
    # a) only using information from earlier (for end of sequences.)
    secondary_accelFrm = (speedFrm - speedFrm.shift(acceleration_shift)) * 2
    accelFrm = accelFrm.fillna(secondary_accelFrm)
    
    # b) only using information from later (for beginning of sequences.)
    secondary_accelFrm = (speedFrm.shift(-acceleration_shift) - speedFrm) * 2
    accelFrm = accelFrm.fillna(secondary_accelFrm)
    
    #### PROJECTED DISTANCE TO COURTLINE ###
    # idea: this is additional info for bounces: we give the perpendicular distance to the nearest courtline
    # of the point, in 2D. In this way, the bounce HMMs will better be able to infer whether a ball is in or out.
    ### for each distinct bounce event, we have to make a distance metric
    # -> for BINSL/BONS(L), we have to make a distance of ball to the region of the near left service box
    # -> for BINSR/BONS(R), we have to make a distance of ball to the region of the near right service box
    # -> for BIN/BON, we have to make a distance of ball to the region of the near singles court
    distance_features = ['dN', 'dF']
    distFrm = pd.concat([ballSF[feat].to_series() for feat in distance_features], axis = 1)
    
    #### LABEL AND CONCAT
    speedFrm.columns = ['v_x', 'v_y']
    accelFrm.columns = ['a_x', 'a_y']
    
    obsFrm = pd.concat([ballFrm, speedFrm, accelFrm, distFrm], axis = 1).dropna()

    return obsFrm

def narrow_down_instantaneous_events(evtSF):
        
    def count_consecutive(evt, evtList):
        i = 0
        for other in evtList:
            if evt == other:
                i += 1
            else:
                return i, other
        return i, None

    frames = evtSF.get_frames()
    min_frame = min(frames)
    max_frame = max(frames)
    
    MAX_REACH = 15

    narrow_event_function = lambda x: x.startswith('B') or x.startswith('H')
    
    narrow_expanded_events = {}
    for frame in frames:
        curr_event = evtSF[frame]
        
        ### figure out left and right look
        comparison_reach = min(frame - min_frame, min(max_frame - frame, MAX_REACH))
        
        isNarrow = narrow_event_function(curr_event)
        
        if not isNarrow:
            narrow_expanded_events[frame] = curr_event
        else:
            events_before = [evtSF[frame - comp] for comp in range(1, comparison_reach + 1)]
            events_after  = [evtSF[frame + comp] for comp in range(1, comparison_reach + 1)]
            
            count_bef, prev_event = count_consecutive(curr_event, events_before)
            count_aft, succ_event = count_consecutive(curr_event, events_after)
            
            if count_bef + 1 < count_aft: ### there are substantially more of our narrow event after than before
                narrow_expanded_events[frame] = prev_event ### -> insert the previous event!
            elif count_bef - 1 > count_aft:
                narrow_expanded_events[frame] = succ_event 
            else:
                narrow_expanded_events[frame] = curr_event 
                
    return StreamFeature(narrow_expanded_events, name = 'exp_eventFlag')
        

def interpolate_matrices(matSF):
    from ATMANpy.funcs.util_funcs import find_last_and_next_list_member
    
    all_frames = matSF.get_frames()
    available_frames = [frame for frame in all_frames if not np.isnan(matSF[frame]).any()]
    
    outDict = {}
    for frame in all_frames:
        if frame in available_frames:
            outDict[frame] = matSF[frame]
        else:
            ls_frame, nx_frame = find_last_and_next_list_member(frame, available_frames)
            
            if nx_frame is None and ls_frame is None:
                outDict[frame] = None ### -> no values anywhere! fill with None
            elif nx_frame is None:
                outDict[frame] = matSF[ls_frame]
            elif ls_frame is None:
                outDict[frame] = matSF[nx_frame]
            else:
                dist = (frame - ls_frame) / (nx_frame - ls_frame)
                outDict[frame] = matSF[ls_frame] * (1 - dist) + matSF[nx_frame] * dist
        
    return StreamFeature(outDict, name = matSF.name)
    
def interpolate_player(plSF):
    from ATMANpy.funcs.util_funcs import find_last_and_next_list_member
    
    available_frames = plSF.dropna().get_frames()
    
    outDict = {}
    for frame, plPt_3D in plSF.items():
        if frame in available_frames:
            outDict[frame] = plSF[frame]
        else:
            ls_frame, nx_frame = find_last_and_next_list_member(frame, available_frames)
            
            if nx_frame is None and ls_frame is None:
                outDict[frame] = None ### -> no values anywhere! fill with None
            elif nx_frame is None:
                outDict[frame] = plSF[ls_frame]
            elif ls_frame is None:
                outDict[frame] = plSF[nx_frame]
            else:
                dist = (frame - ls_frame) / (nx_frame - ls_frame)
                outDict[frame] = plSF[ls_frame] * (1 - dist) + plSF[nx_frame] * dist
        
    return StreamFeature(outDict, name = plSF.name)
            


def interpolate_ball_positions_3D_smart(mySF,
                                        max_iterations = 150, correction_strength = 0.25,
                                        acceptable_standard_deviations = 3, 
                                        acceptable_absolute_deviation = 10,
                                        prop_of_z_needed = 0.5):
    ##### Needs StreamFeatures with...
    ### ball_3D: a Point3D
    ### ballCnt: a ball contour
    ### M_int: internal camera matrices
    ### M_ext: external camera matrices
    
    from ATMANpy.types.Points import Point2D
    from ATMANpy.types.Shapes3D import Point3D
    from ATMANpy.funcs.util_funcs import find_last_and_next_list_member
    import ATMANpy.funcs.HMM.HPost as HPost
    
    known_frames = mySF['ball_3D'].dropna().get_frames()
    
    ### FIRST PASS: FIGURE OUT LAMBDAS
    lambdaDicts = []
    argDicts = []
    sequence_lambdas = {}
    interpolation_arguments = {}
    for frame, featDict in mySF.items():
        ballPt_3D = featDict['ball_3D']
        ballPt_2D = featDict['ballCnt'].center()
        M_int = featDict['M_int']
        M_ext = featDict['M_ext']
        if ballPt_3D is None:
            ### here, we need to interpolate!
            ls_frame, nx_frame = find_last_and_next_list_member(frame, known_frames)
            # print(f'{frame} -- last: {ls_frame}, next: {nx_frame}')
            if ls_frame is not None and nx_frame is not None:
                ### if both last and next position are known, we can interpolate
                lsPt = mySF['ball_3D'][ls_frame]
                nxPt = mySF['ball_3D'][nx_frame]
                   
                ### Derive the ray on which the ball has to lie
                m_c, b_c = HPost.get_camera_ray_params(ballPt_2D, M_int)
                m_w, b_w = HPost.get_world_ray_params(m_c, M_ext)
            
                ### together with knowledge about the last and the next point that is known, 
                ### we can interpolate!
                ### UPDATE: We have to do two passes, because the interpolation can be slightly off:
                    # 1) lambdas can be smaller than zero, and larger than one
                    # 2) lambdas are not necessarily monotonously increasing.
                    ### -> both of these issues are best fixed knowing the entire sequence of lambdas.
                sequence_lambdas[frame] = HPost.calculate_interpolation_lambda(m_w, b_w, lsPt, nxPt)
                # print(f'{frame}: {calculate_interpolation_lambda(m_w, b_w, lsPt, nxPt)}')
                
                ### save the necessary items in another dictionary, so we don't need to re-create
                interpolation_arguments[frame] = (m_w, b_w, lsPt, nxPt)
                # ball_3D.loc[frame] = calculate_interpolated_point(m_w, b_w, lsPt, nxPt)
        else:
            if not sequence_lambdas == {}:
                lambdaDicts.append(sequence_lambdas)
                argDicts.append(interpolation_arguments)
                sequence_lambdas = {}
                interpolation_arguments = {}
        
    ### FIX LAMBDAS
    fixed_lambdaDicts = [HPost.fix_lambda_sequence_iterative(l, 
                                                             max_iterations = max_iterations,
                                                             correction_strength = correction_strength) for l in lambdaDicts]
        
    ### SECOND PASS: PERFORM INTERPOLATIONS
    interDicts = []
    interpolated_points = {}
    for lambdaDict, argDict in zip(fixed_lambdaDicts, argDicts):
        for frame, lmbd in lambdaDict.items():
            interpolated_points[frame] = HPost.calculate_interpolated_point(*argDict[frame], lmbd = lmbd)    
        interDicts.append(interpolated_points)
        interpolated_points = {}
    
    ### THIRD PASS: FIX OUTLIERS IN Z-DIMENSION
    # note that x and y dimensions are okay as long as interpolation fix with lambdas works...
    # but this leaves the z dimension open still. We can look at the arc, estimate it, and remove outliers!
    fixed_interDicts = []
    for lambdaDict, interDict in zip(lambdaDicts, interDicts):
        lambdas = np.array(list(lambdaDict.values()))
        z_values = np.array([coords[2] for coords in interDict.values()])
        
        fit, sd = HPost.fit_parabola_intervals(lambdas, z_values)
    
        replace_because_negative = (z_values < 0) 
        replace_because_weird = np.abs(z_values - fit) > np.maximum(acceptable_absolute_deviation, 
                                                                    acceptable_standard_deviations * sd)
        replace = replace_because_negative | replace_because_weird
        
        if replace.sum() == 0:
        #     print(f'{z_values}: Replacing in {replace.sum()} positions')
            fixed_interDicts.append(interDict)
        elif replace.sum() / len(replace) > prop_of_z_needed:
            # print('Too many issues, can not fix z-coordinates.')
            fixed_interDicts.append(interDict) 
        else:
            x_values, y_values = [coords[0] for coords in interDict.values()], [coords[1] for coords in interDict.values()]
            # new_z_values = np.where(replace, fit, z_values) ### replacing with fit is easy, but choppy
            ### instead: maybe linearly interpolate?
            gap_z_values = np.where(replace, np.nan, z_values)
            idx = np.arange(len(gap_z_values))
            idx_not_nan = np.isnan(gap_z_values) == False
            new_z_values = np.interp(idx, idx[idx_not_nan], gap_z_values[idx_not_nan])
            new_coords = list(zip(x_values, y_values, new_z_values))
            new_interDict = dict(zip(list(interDict.keys()), new_coords))
            fixed_interDicts.append(new_interDict)
    
    ### ADD FORWARD AND BACKWARD INTERPOLATIONS
    ## TODO: ?
    
    ### JOIN INTERDICTS AND TURN TO BALL 
    joinDict = {frame: Point3D(interDict[frame]) for interDict in interDicts for frame in interDict}
    
    return StreamFeature(joinDict, name = 'ball_3D')

def find_ballkid_sequences(cntSF, namedROIs, fill_limit = 3,
                           distance_adjacent = 30, distance_near = 125,
                           relevance_irrelevant = 0.25, relevance_conditional = 0.8):
    import ATMANpy.funcs.ballkid_funcs as bk
    
    ### from movement contours and ballkid ROIs (with descriptive names!) ...
    ### ... produce a scalar between 0 and 1, indicating the proportion of the contour within the ROI ...
    intSF = bk.contour_proportion_in_ROI(cntSF, namedROIs)
    ### ... and turn it into a dataframe ...
    intDF = bk.prop_sf_to_dataframe(intSF, fill_limit = fill_limit)
    ### ... where the proportion values are transformed into strings explaining the motion of the contour relative to the ROI.
    strDF = bk.prop_frame_to_motion(intDF)
    
    ### Remove nan rows and split into contiguous chunks, called BLOCKS.
    strDF = strDF.dropna(axis = 0, how = 'all')
    strDFs = ut.split(strDF)
        
    ### Look at these blocks and join those which together represent a single interruption of play due to ballkids.
    outList = bk.segment_play(strDFs, 
                              distance_adjacent = distance_adjacent, 
                              distance_near = distance_near,
                              relevance_irrelevant = relevance_irrelevant,
                              relevance_conditional = relevance_conditional)
    
    ### Transform into a BoolFeature.
    ballkidBF = bk.outlist_to_bool(outList).conform(cntSF).fillna(False)

    ### TODO: Widen the segments of play towards the back -> often, they cut out a bit too early.
    return ballkidBF

def find_player_candidate_links(candSF, frame_offset = 5, relative_to = 'other'):
    from ATMANpy.funcs.player_funcs import find_player_candidate_contours_with_overlap
    
    sfDict = {}
    
    player_candidates_prev = None
    for idx, frame in enumerate(candSF):
        if idx % frame_offset != 0:
            featDict = {'player_candidates': None,
                        'player_parents': None}
        else:
            candCnts = candSF[frame]
            
            if player_candidates_prev is None:
                player_candidates, player_parents = candCnts, [[]] * candCnts.length()
            else:
                player_candidates, player_parents = find_player_candidate_contours_with_overlap(candCnts, player_candidates_prev, 
                                                                                                relative_to = relative_to)
            player_candidates_prev = player_candidates
                
            featDict = {'player_candidates': player_candidates,
                        'player_parents': player_parents}
                
        sfDict[frame] = featDict
        
    return StreamFeatures(sfDict)

def find_player_candidate_links_forward(candSF, frame_offset = 5, relative_to = 'self'):
    from ATMANpy.funcs.player_funcs import find_player_candidate_contours_with_overlap
        
    sfDict = {}
    player_candidates_next = None
    
    frames = candSF.get_frames()
    idxs_rev = reversed(range(len(frames)))
    frames_rev = reversed(frames)
    
    for idx, frame in zip(idxs_rev, frames_rev):
        if idx % frame_offset != 0:
            featDict = {'player_candidates': None,
                        'player_children': None}
        else:
            candCnts = candSF[frame]
            
            if player_candidates_next is None:
                player_candidates, player_children = candCnts, [[]] * candCnts.length()
            else:
                player_candidates, player_children = find_player_candidate_contours_with_overlap(candCnts, player_candidates_next, 
                                                                                                 relative_to = relative_to)
            player_candidates_next = player_candidates
                
            featDict = {'player_candidates': player_candidates,
                        'player_children': player_children}
        
        sfDict[frame] = featDict
        
    return StreamFeatures(sfDict).sort()


def find_matching_tracks(candSF, splitDF, frame_offset = 5, num_points = 6, threshold_distance = 20):
    dicts = []
    for col in splitDF:
        idx = splitDF[col].dropna().index
        fs, ls = idx[0], idx[-1]
        myDict = {'first': fs, 'last': ls}
        dicts.append(myDict)
    fslsFrm = pd.DataFrame(dicts, index = splitDF.columns)
        
    fsSer = fslsFrm['first'].sort_values()
    lsSer = fslsFrm['last'].sort_values() 
        
    ### for each track, find a track that would complement it (if possible)
    ### i.e, another track that begins right after that track ends.
    tracks_to_add = []
    for track_id, frame_ls in lsSer.iteritems():
        frame_after_ls = frame_ls + frame_offset
        
        begin_after_ls_series = fsSer == frame_after_ls
        tracks_beginning_after_ls = begin_after_ls_series[begin_after_ls_series].index.tolist()
    
        for track_after_id in tracks_beginning_after_ls:
            myCnt_idx = splitDF.loc[frame_ls, track_id].astype(int)
            aftCnt_idx = splitDF.loc[frame_after_ls, track_after_id].astype(int)
            
            myCnt = candSF['player_candidates'][frame_ls][myCnt_idx]
            aftCnt = candSF['player_candidates'][frame_after_ls][aftCnt_idx]
            
            myPts, aftPts = myCnt.sample_points(samples = num_points), aftCnt.sample_points(samples = num_points)
            
            d = min([myPt.distance_to(aftPt) for myPt in myPts for aftPt in aftPts])
            if d < threshold_distance:
                mySer = splitDF[track_id].dropna()
                aftSer = splitDF[track_after_id].dropna()
                
                jointSer = pd.concat([mySer, aftSer])
                jointSer.name = str(mySer.name) + '_' + str(aftSer.name)
                tracks_to_add.append(jointSer)
    
    if tracks_to_add == []:
        return None
    else:
        addFrm = pd.concat(tracks_to_add, axis = 1)
        dup_cols = addFrm.columns.duplicated(keep = 'first')
        addFrm = addFrm.loc[:, ~dup_cols]
        return addFrm
    
def iteratively_match_tracks(candSF, splitDF, frame_offset = 5, num_points = 6, threshold_distance = 20):
    addFrm = find_matching_tracks(candSF, splitDF, 
                                  frame_offset = frame_offset, 
                                  num_points = num_points, 
                                  threshold_distance = threshold_distance)
    if addFrm is None:
        return splitDF
 
    new_columns = [c for c in addFrm if c not in splitDF]
    if new_columns == []:
        return splitDF
    else:
        new_splitDF = splitDF.merge(addFrm[new_columns], left_index = True, right_index = True, how = 'left')
        return iteratively_match_tracks(candSF, new_splitDF, 
                                        frame_offset = frame_offset, 
                                        num_points = num_points, 
                                        threshold_distance = threshold_distance)
        
def add_in_hit_point(mySF, height_in_feet = 5, resolution = 5, radius_in_feet = 15):
    ### get last hit frame
    ls_hit_frame = mySF['hitFlag'].dropna().get_frames()[-1]
    
    hitFlag = mySF['hitFlag'][ls_hit_frame]
    topPlayerCnt = mySF['topPlayerCnt'][ls_hit_frame]
    botPlayerCnt = mySF['botPlayerCnt'][ls_hit_frame]
    P = mySF['P'][ls_hit_frame]
    
    ### figure out how long we see a ball for
    fs_frame, ls_frame = mySF.fs_ls()
    last_flight_frames = list(range(ls_hit_frame, ls_frame + 1))
    ball_choicePt = mySF['ball_choicePt'].conform(last_flight_frames)
    
    steepness = 0.1
    thresh_frame = 25
    last_flight_frames_rect = np.array(last_flight_frames) - ls_hit_frame
    is_flight_prior = 1 - 1 / (1 + np.exp(-steepness * (last_flight_frames_rect - thresh_frame)))
    
    ball_vicinity = 3
    isballArr = np.array(list((ball_choicePt != None).values()))
    
    shiftArr = np.stack([ut.shift_array(isballArr, offset) for offset in range(-ball_vicinity, 1)], axis = -1)
    ball_pct = np.nanmean(shiftArr, axis = 1)
    
    threshold = 0.05
    flight_probs = ball_pct * is_flight_prior
    flight_probs_enveloped = np.array([np.max([flight_probs[i:]]) for i in range(len(flight_probs))])
    
    in_flight = flight_probs_enveloped > threshold
    if in_flight.all():
        last_flight_index = -1
    else:
        ### first value at which the ball is not in flight anymore, bounded by 
            # a minimum flight length = 10, 
            # and the length of possible flight frames
        last_flight_index = min(max(10, np.argmin(in_flight)), len(last_flight_frames) - 1)
    ls_flight_frame = last_flight_frames[last_flight_index]
    
    try:
        side, hit = hitFlag.split('_') ### if we have a serve
        is_serve = True
    except:
        side, stroke, kind = hitFlag.split('_') ### if we have a groundstroke
        is_serve = False
        
    topPt3D, botPt3D = sa.find_player_positions_3D(P, topPlayerCnt, botPlayerCnt)
    
    basePt3D = topPt3D if side == 'top' else botPt3D
    height_in_axis_units = height_in_feet * const.SCALE
    hingePt3D = basePt3D.up(height_in_axis_units)
    
    if side == 'top':
        theta_rad_min, theta_rad_max = 0, 1
    elif side == 'bot':
        theta_rad_min, theta_rad_max = 1, 2
    
    limit_theta = theta_rad_min * np.pi, theta_rad_max * np.pi

    radius_in_vx = radius_in_feet * const.SCALE
    hypothetical_circle = Circle3D(hingePt3D.coords, radius = radius_in_vx, limit_theta = limit_theta)
    hypothetical_points = hypothetical_circle.rasterize(resolution = resolution)
    
    x, y, z = hypothetical_points.T
    cldPt3D = Point3D((x, y, z))
    
    streamDict = {ls_flight_frame: {'cldPt3D': cldPt3D, 'hitFlag': 'inferred_flight_end'}}
    
    return StreamFeatures(streamDict)
    
def intersect_ball_observations_and_planes(mySF, matchCamera = None, P = None):
    from ATMANpy.types.ShapesTennis3D import Net
    myNet = Net.make_singles()
    netReg = myNet.get_rectangular_net_region()
    
    anchorSF = mySF['cldPt3D']
    shapeSF = mySF['ball_choicePt']
    camlnSF = shapeSF.apply(sa.make_camline, myCamera = matchCamera)
    
    ### the flight_plane height quantities below are irrelevant to trajectory finding, hence they're not parameters
    flight_plane_size = 100 
    flight_plane_heigth = flight_plane_size / 2 ### half of it is overground
    
    anchor_frames = anchorSF.dropna().get_frames()
    
    outDict = {}
    for idx in range(len(anchor_frames) - 1): ### dont do the last one -> the hypothetical flight cloud is not a hit cloud
        hit_idx_is_last = idx == len(anchor_frames) - 2
    
        frm_currhit, frm_nexthit = anchor_frames[idx], anchor_frames[idx + 1]
        anchorPts_curr, anchorPts_next = anchorSF[frm_currhit], anchorSF[frm_nexthit]
        ballPt_curr, ballPt_next = shapeSF[frm_currhit], shapeSF[frm_nexthit]
        
        flight_frames = list(range(frm_currhit, frm_nexthit + 1))
        non_na_frmArr = np.array([(i, f) for i, f in enumerate(flight_frames) if f in shapeSF.dropna().get_frames()])
        if len(non_na_frmArr) == 0:
            non_na_idxs = np.array([], dtype = int)
            print('NO BALL CANDIDATES OBSERVED')
        else:
            non_na_idxs = non_na_frmArr[:, 0]
        
        ptArr_curr = np.stack([anchorPts_curr.x, anchorPts_curr.y, anchorPts_curr.z], axis = -1)
        ptArr_next = np.stack([anchorPts_next.x, anchorPts_next.y, anchorPts_next.z], axis = -1)
        
        ptArr_curr_ground = np.copy(ptArr_curr)
        ptArr_curr_ground[:, 2] = 0
        ptArr_next_ground = np.copy(ptArr_next)
        ptArr_next_ground[:, 2] = 0
            
        num_curr_points = len(ptArr_curr_ground)
        num_next_points = len(ptArr_next_ground)
        num_flight_frames = len(flight_frames)
        
        pointArr = np.full((num_curr_points, num_next_points, 2, 3), fill_value = np.nan) ### use this to store the current and next point
        coordArr = np.full((num_curr_points, num_next_points, num_flight_frames, 3), fill_value = np.nan)
        paramArr = np.full((num_curr_points, num_next_points, num_flight_frames, 3), fill_value = np.nan)
        planeArr = np.full((num_curr_points, num_next_points, num_flight_frames, 2), fill_value = np.nan)
        
        ###### DEAL WITH NET COORDINATES, IF APPLICABLE #######
        if hit_idx_is_last:
            net_coordArr = np.full((num_flight_frames, 3), fill_value = np.nan)
            net_paramArr = np.full((num_flight_frames, 3), fill_value = np.nan)

            for frame_idx, frame_num in enumerate(flight_frames):
                camLn = camlnSF[frame_num]
                if camLn is not None:
                    net_ballPt3D, net_params = camLn.intersect(netReg, rectify_plane = True)
                else:
                    net_ballPt3D = None 

                if net_ballPt3D is not None:
                    x_net, y_net, z_net = net_ballPt3D.coords
                    l_net, m_net, n_net = net_params
                    net_coordArr[frame_idx] = x_net, y_net, z_net
                    net_paramArr[frame_idx] = l_net, m_net, n_net
        else:
            net_coordArr = None
            net_paramArr = None
            
        ###### DEAL WITH FLIGHT COORDINATES #######
        for curr_idx in range(num_curr_points):
            currptArr = ptArr_curr[curr_idx]
            currptArr_ground = ptArr_curr_ground[curr_idx]
            currPt_ground = Point3D(tuple(currptArr_ground))
            
            pointArr[curr_idx, :, 0] = currptArr 
            
            for next_idx in range(num_next_points):
                nextptArr = ptArr_next[next_idx]
                nextptArr_ground = ptArr_next_ground[next_idx]
                nextPt_ground = Point3D(tuple(nextptArr_ground))
        
                pointArr[curr_idx, next_idx, 1] = nextptArr
        
                ### make flight plane
                flightLn = Line3D(currPt_ground, nextPt_ground)
                top_bottom = '+y' if flightLn.to_vector().y > 0 else '-y'
                left_right = '+x' if flightLn.to_vector().x > 0 else '-x'
                flightReg = Region3D.from_line3D(flightLn, flight_plane_size, dim = 'z', top_bottom = top_bottom, left_right = left_right)
                
                flightLen = flightLn.to_vector().length()
                
                for frame_idx, frame_num in enumerate(flight_frames):
                    camLn = camlnSF[frame_num]
                    if camLn is not None:     
                        ### 1) Get the intersection of camera line and flight region
                        ballPt3D, params = camLn.intersect(flightReg, rectify_plane = True)
                        
                        ### 2) Build arrays
                        x, y, z = ballPt3D.coords
                        l, m, n = params
                        
                        coordArr[curr_idx, next_idx, frame_idx] = x, y, z
                        paramArr[curr_idx, next_idx, frame_idx] = l, m, n
                        planeArr[curr_idx, next_idx, frame_idx] = flightLen * l, z
                        
        
        outDict[frm_currhit] = {'coordArr': coordArr, 
                                'paramArr': paramArr, 
                                'planeArr': planeArr, 
                                'pointArr': pointArr, 
                                'net_coordArr': net_coordArr,
                                'net_paramArr': net_paramArr}
        
    outDict[frm_nexthit] = {'coordArr': None, 
                            'paramArr': None, 
                            'planeArr': None, 
                            'pointArr': None, 
                            'net_coordArr': None,
                            'net_paramArr': None}
        
    return StreamFeatures(outDict)

def best_colinear_planes(mySF, matchCamera = None, num_angles = 1001, lim_pi = np.pi / 4, max_slope = 10 ** 6,
                         w_obs = 0.8, w_cld = 0.2):
    # The colinear projected line in this function is the 2D line created when projecting 
    # a point in 3D space (ang_vecs are the set of these points for different angles)
    # and a point straight above that point onto the screen.

    # EDIT: this is not exactly what we want: we want to find the line a point makes when it flies
    # along the camera line through the hit point, but changes only its height in 3D (since the camera line)
    # goes below ground fast if the camera is well above the playing field, and the ball is assumed
    # to travel in a straight plane in (x, y) coordinates, with changing heights (i.e: no sidespin)
    # this is done in a different, better function.
    
    # now, the when adjusting the height of the point above the camera line arbitrarily,
    # we usually get not just a line, but a plane 
    # (except in the case where the camera rotation is perfectly aligned with the vertical flight plane)
    # (i.e, for a perfectly straight camera looking down the middle, the ball straight ahead will always appear)
    # to be on a line after projecting, not on a plane.
    
    # Hence, we need to restrict the height, otherwise the entire screen (?) can be projected on 
    # for some z and some point along the camera line.

    # in general, the second point has to be above ground. more strictly, we can say that such projections
    # are close to the projected line of a ball flying dead straight, 
    # with z adjusting so that the overall height is the same as the height of the original point.

    ### check the equation for the colinear lines
    # (Red Notebook, 09.03.2025)
    d = 500 ### this parameter is pretty irrelevant, hence not a kwarg
    M_proj = matchCamera.get_P()
    
    ### construct angle array
    fs_pi, ls_pi = -lim_pi, lim_pi
    angles = np.linspace(fs_pi, ls_pi, num = num_angles)
    d_sin_angles, d_cos_angles = d * np.sin(angles), d* np.cos(angles)
    ang_vecs = np.array([d_sin_angles, -d_cos_angles, np.zeros(num_angles), np.zeros(num_angles)]).T # shape: angles x 4

    ### get the A, B, C --> these are dependent on theta, but not on the height
    A0, B0, C0 = np.einsum('ij, nj -> in', M_proj, ang_vecs)
    A1, B1, C1 = A0 + M_proj[0, 2], B0 + M_proj[1, 2], C0 + M_proj[2, 2]

    ### dehomogenize
    ux, uy = A0 / C0, B0 / C0
    vx, vy = A1 / C1, B1 / C1
    u, v = np.array([ux, uy]).T, np.array([vx, vy]).T

    ### get the slope of the colinear projected line
    # -> points flying colinearly with the camera will be projected close to one of these lines.
    m = (vy - uy) / (vx - ux) ### these are needed for minimising distance of observed points
                                # if vx - ux = 0, then there is an easier formula for minimal distance
    # correct extreme slopes (vx - ux almost 0) to avoid numerical issues
    m = np.where(np.abs(m) > max_slope, np.nan, m)

    ### calculate normal vector coordinates
    diff_vecs = v - u ### shape: angles x 2
    dx, dy = diff_vecs[:, 0], diff_vecs[:, 1]
    nx, ny = dy, -dx
    len_n = (nx ** 2 + ny ** 2) ** 0.5
    nx_unit, ny_unit = nx / len_n, ny / len_n

    ### some broadcasting for later. shape: angles, obs
    nx_unit_brdc = nx_unit[:, np.newaxis] 
    ny_unit_brdc = ny_unit[:, np.newaxis]
    m_brdc = m[:, np.newaxis]
    ux_brdc = ux[:, np.newaxis]
    uy_brdc = uy[:, np.newaxis]

    def calculate_optimal_gamma(px_brdc, py_brdc):
        # this calculates the gamma (the parameter of the unit normal vector (nx, ny) of our co-linear projected line)
        # for which (px, py) + gamma * (nx, ny) intersects the co-linear projected line. 
        # in effect, the gamma is the shortest signed distance to the co-linear projected line.
        # the distance is then simply the absolute value of the gamma
        # px_brdc and py_brdc are of shape: choice_dim, obs (choice_dim = angles, or hit points)
        
        ### optimal gamma -> usual formula where ux != vx
        opt_gamma_numer = m_brdc * (px_brdc - ux_brdc) - (py_brdc - uy_brdc)
        opt_gamma_denom = ny_unit_brdc - m_brdc * nx_unit_brdc
        opt_gamma_usual = opt_gamma_numer / opt_gamma_denom

        ### optimal gamma -> fallback formula where ux = vx (and therefore m is dangerously large in absolute terms)
        opt_gamma_fallback = (px_brdc - ux_brdc) / nx_unit_brdc

        ### join the two formulae
        opt_gamma = np.where(np.isnan(opt_gamma_usual), opt_gamma_fallback, opt_gamma_usual)
        
        return opt_gamma

    anchorSF = mySF['cldPt3D']
    shapeSF = mySF['ball_choicePt']

    anchor_frames = anchorSF.dropna().get_frames()

    outDict = {}
    for idx in range(len(anchor_frames) - 1):
        hit_idx_is_last = idx == len(anchor_frames) - 2
    
        frm_currhit, frm_nexthit = anchor_frames[idx], anchor_frames[idx + 1]
        flight_frames = list(range(frm_currhit, frm_nexthit + 1))
        
        ### figure out observed projected point vectors
        observed_and_none_points = list(shapeSF[flight_frames].values())
        obsArr = np.array([obsPt.coords if obsPt is not None else (np.nan, np.nan) for obsPt in observed_and_none_points]) ### shape obs, 2
        px, py = obsArr[:, 0], obsArr[:, 1]
        px_brdc = px[np.newaxis, :] # shape: angles, obs
        py_brdc = py[np.newaxis, :] # shape: angles, obs

        ### figure out anchor point projections
        anchorPts3D_curr, anchorPts3D_next = anchorSF[frm_currhit], anchorSF[frm_nexthit] 
        anchorArr_curr, anchorArr_next = anchorPts3D_curr.project(M_proj), anchorPts3D_next.project(M_proj) ### shapes for either: cloud_points, 2

        ax_curr = anchorArr_curr[:, 0]
        ay_curr = anchorArr_curr[:, 1]
        
        ax_curr_brdc = ax_curr[np.newaxis, :]
        ay_curr_brdc = ay_curr[np.newaxis, :]

        if not hit_idx_is_last:
            ax_next = anchorArr_next[:, 0]
            ay_next = anchorArr_next[:, 1]
            
            ax_next_brdc = ax_next[np.newaxis, :]
            ay_next_brdc = ay_next[np.newaxis, :]

        ### calculate optimal distances for observed balls, candidate anchor points at moment of hit
        opt_gamma_ball = calculate_optimal_gamma(px_brdc, py_brdc) ### shape: angles, obs
        opt_gamma_curr = calculate_optimal_gamma(ax_curr_brdc, ay_curr_brdc) ### shape: angles, cloud_points

        if not hit_idx_is_last:
            opt_gamma_next = calculate_optimal_gamma(ax_next_brdc, ay_next_brdc) ### shape: angles, cloud_points
        else:
            opt_gamma_next = np.zeros_like(opt_gamma_curr) ### for the last hit, we have no indication of where it went

        opt_dist_ball = np.abs(opt_gamma_ball)
        opt_dist_curr = np.abs(opt_gamma_curr)
        opt_dist_next = np.abs(opt_gamma_next)

        ### the distance of every observation to every line is the absolute value of the gamma (the parameter along the normalized normal vector)
        opt_dist_ball_avg = np.nanmean(opt_dist_ball, axis = 1)
        opt_dist_curr_avg = np.nanmean(opt_dist_curr, axis = 1)
        opt_dist_next_avg = np.nanmean(opt_dist_next, axis = 1)

        if hit_idx_is_last:
            opt_dist_anch_avg = opt_dist_curr_avg ### since opt_dist_next_brdc is useless, and all zeroes.  
        else:
            opt_dist_anch_avg = (opt_dist_curr_avg + opt_dist_next_avg) / 2

        ### question: are the thetas of the co-linear projected for the observed points and the anchor cloud points consistent?
        best_angle_ball_idx = np.argmin(opt_dist_ball_avg)
        best_angle_anch_idx = np.argmin(opt_dist_anch_avg)
        
        # -> cross check the deviations of the projected balls evaluated at the best planes
        best_angle_ball_avg = opt_dist_ball_avg[best_angle_ball_idx]
        best_angle_anch_avg = opt_dist_anch_avg[best_angle_anch_idx]
        anch_angle_ball_avg = opt_dist_ball_avg[best_angle_anch_idx]
        ### -> report both of these for us to use as metrics when deciding if the co-linear plane should be taken
        # a) best_angle_ball_avg is the mean deviation of projected points from the best co-linear plane
            # -> indication if the projected points may be co-linear
        # b) best_angle_anch_avg is the mean deviation of anchor points from the co-linear plane that fits the anchor points best
            # -> indication if the anchor points may be co-linear
        # c) anch_angle_ball_avg is the mean deviation of projected points from the co-linear plane that fits the anchor points best
            # -> indication if the balls and the anchor points are consistently co-linear

        ### best combined angle -> this is what we'll use at the end, if we accept that the flight of the ball is co-linear
        opt_dist_avg = w_obs * opt_dist_ball_avg + (w_cld / 2) * opt_dist_curr_avg + (w_cld / 2) * opt_dist_next_avg
        if hit_idx_is_last:
           opt_dist_avg = opt_dist_avg / (w_obs + w_cld / 2) ### since opt_dist_next_brdc is useless, and all zeroes.  

        best_angle_idx = np.argmin(opt_dist_avg)
        best_angle_avg = opt_dist_avg[best_angle_idx]
        ### d) report this too: an overall metric of co-linearity

        ### we can now choose the best starting and ending point
        best_curr_idx = np.argmin(opt_dist_curr[best_angle_idx])
        
        # a) in 2D
        ax_curr_best = ax_curr[best_curr_idx]
        ay_curr_best = ay_curr[best_curr_idx]

        # b) in 3D
        anchor3DArr_curr = np.array([anchorPts3D_curr.x, anchorPts3D_curr.y, anchorPts3D_curr.z]).T
        curr_coords_3D = anchor3DArr_curr[best_curr_idx]
        # -> this are forced to be projected onto the line
        
        if not hit_idx_is_last:
            best_next_idx = np.argmin(opt_dist_next[best_angle_idx])
            
            # a) in 2D
            ax_next_best = ax_curr[best_next_idx]
            ay_next_best = ay_curr[best_next_idx]

            # b) in 3D
            anchor3DArr_next = np.array([anchorPts3D_next.x, anchorPts3D_next.y, anchorPts3D_next.z]).T
            next_coords_3D = anchor3DArr_next[best_next_idx]
            # -> this are forced to be projected onto the line

        ### get the gammas at the optimal angle, and correct the the observed points to be on the co-linear projected line
        best_unit_nx = nx_unit[best_angle_idx]
        best_unit_ny = ny_unit[best_angle_idx]
        
        best_opt_gamma_ball = opt_gamma_ball[best_angle_idx]
        px_shift = px + best_opt_gamma_ball * best_unit_nx
        py_shift = py + best_opt_gamma_ball * best_unit_ny
        shifted_ballArr = np.array([px_shift, py_shift]).T

        best_opt_gamma_curr = opt_gamma_curr[best_angle_idx, best_curr_idx]
        ax_curr_shift = ax_curr_best + best_opt_gamma_curr * best_unit_nx
        ay_curr_shift = ay_curr_best + best_opt_gamma_curr * best_unit_ny
        
        if not hit_idx_is_last:
            best_opt_gamma_next = opt_gamma_next[best_angle_idx, best_next_idx]
            ax_next_shift = ax_next_best + best_opt_gamma_next * best_unit_nx
            ay_next_shift = ay_next_best + best_opt_gamma_next * best_unit_ny
        else:
            ax_next_shift = np.nan
            ay_next_shift = np.nan
        
        original_anchArr = np.array([[ax_curr_best, ay_curr_best], 
                                     [ax_next_best, ax_next_best]])
        shifted_anchArr = np.array([[ax_curr_shift, ay_curr_shift],
                                    [ax_next_shift, ay_next_shift]])
        
        ### get the lambdas at the optimal angle.        
        best_ux, best_uy = u[best_angle_idx]
        best_vx, best_vy = v[best_angle_idx]
        best_dy = best_vy - best_uy

        best_opt_lambda_ball = (py - best_uy + best_opt_gamma_ball * best_unit_ny) / best_dy
        
        ### from the lambdas, calculate the height in 3D where the camera line cuts the original 3D line
        # -> this is not the height of the ball, but a point on the vertical 3D line such that
        #      the ball is projected onto the same point as that point on the vertical 3D line.
        
        best_opt_height_ball_numer = best_opt_lambda_ball * best_dy * C0[best_angle_idx]
        best_opt_height_ball_denom = M_proj[1, 2] - M_proj[2, 2] * (best_uy + best_opt_lambda_ball * best_dy)
        best_opt_height_ball = best_opt_height_ball_numer / best_opt_height_ball_denom
        # -> this height is relative to the camera!

        best_opt_height_ball_3D = np.array([np.zeros_like(best_opt_height_ball),
                                            np.zeros_like(best_opt_height_ball),
                                            best_opt_height_ball]).T

        best_ang_vec = ang_vecs[best_angle_idx][:3] ### drop homogenous coordinate (= 0 in this case)
        best_world_basis_point = best_ang_vec + matchCamera.get_cw()
        # best_world_basis_point = best_world_basis_point[np.newaxis, :]
        
        best_point_on_vertical = best_world_basis_point + best_opt_height_ball_3D        


        ### move the current point in 3D onto our projection plane
        sin_theta, neg_cos_theta, _ = best_ang_vec / d
        cos_theta = - neg_cos_theta
        sx, sy, _ = curr_coords_3D
        cx, cy, _ = matchCamera.get_cw()

        gamma_curr = sin_theta * (cy - sy) + cos_theta * (cx - sx)

        sx_on, sy_on = np.array([sx, sy]) + gamma_curr * np.array([cos_theta, sin_theta])        

        outDict[frm_currhit] = {'shifted_ballArr': shifted_ballArr, 'shifted_anchArr': shifted_anchArr,
                                'original_ballArr': obsArr, 'original_anchArr': original_anchArr,
                                'vertical_3D_points': best_point_on_vertical,
                                'u': (best_ux, best_uy), 'v': (best_vx, best_vy), 
                                's': (sx, sy), 's_on': (sx_on, sy_on),
                                'pure_ball_mean_dev': best_angle_ball_avg, 'pure_anchor_mean_dev': best_angle_anch_avg,
                                'cross_val_mean_dev': anch_angle_ball_avg, 'final_mean_dev': best_angle_avg}

        ### from the best angle we immediately get a flight plane!
        # this is the flight plane given that the ball flies co-linear with the camera
        # in this flight plane, we can not locate the ball by intersecting plane and camera line through observation (-> they're identical)
        # instead, we use the height of the ball on the line (which can easily be calculated with the other optimization parameter, lambda)
        # and then find a trajectory consistent with these projected heights.
        # also consistent with bounce info, etc., of course.

    outDict[frm_nexthit] = {'shifted_ballArr': None, 'shifted_anchArr': None,
                            'original_ballArr': None, 'original_anchArr': None,
                            'vertical_3D_points': None,
                            'u': None, 'v': None,
                            's': None, 's_on': None,
                            'pure_ball_mean_dev': None, 'pure_anchor_mean_dev': None,
                            'cross_val_mean_dev': None, 'final_mean_dev': None}
        
    return StreamFeatures(outDict)



def filter_trajectories(trajDict, penalty_quantile = 0.05):
    
    outDict = {}
    for frame in trajDict:
        coordArr, paramArr, planeArr, pointArr, penalty = trajDict[frame]
        
        if coordArr is None: ### at last hypothetical flight frame after last hit.
            outDict[frame] = (None, None, None, None)
        
        else:
            ### Compare the 'best' trajectories with theoretical trajectories
            # -> since this is computationally intensive, we don't do it for all, but only for those that pass the first filter.
            penalty_threshold = np.quantile(penalty.flatten(), penalty_quantile)
            penalty_filter = penalty <= penalty_threshold
            
            coordArr_filt = coordArr[penalty_filter]
            paramArr_filt = paramArr[penalty_filter]
            planeArr_filt = planeArr[penalty_filter]
            pointArr_filt = pointArr[penalty_filter]
        
            outDict[frame] = (coordArr_filt, paramArr_filt, planeArr_filt, pointArr_filt)

    return outDict
    

def combine_bounce_priors_and_probs(mySF, schemaLines = schemaLines):
    ### TODO: priors strongly influence the selection of the final trajectory.
    ### 1) use also the deviation of the trajectory to exclude bogus trajectories
    ### 2) make priors more informed 
        # -> player movement, 
        # -> umpire shouts
        # -> number of hits after the current hit (point can not be over if there are a bunch more hits after the current one)
        # ...
    
    def prior_dist(lmb):
        ### given a lambda (the depth of the receiving player over the depth of the court),
        ### calculate the probability of zero, one, two bounces (more bounces are not calculated)
        ### TODO: add in position of receiving player at hit 
            ### -> if he moves from baseline to ball at net, he was dropshotted
            ### -> if he was already at the net, likely a volley!
        ### TODO: Incorporate length of time between the two hits (the shorter, the fewer bounces.)
        p1 = (2 / (1 + np.exp(-3 * lmb)) - 1)
        p0, p2 = 0.5 * (1 - p1), 0.5 * (1 - p1)
        return p0, p1, p2
        
    def set_broad_diagonal_to_zero(arr, offset = 5):
        
        idxs = np.arange(len(arr))
        
        new_arr = []
        for idx, row in enumerate(arr):
            to_zero = np.abs(idxs - idx) < offset
            new_row = row * ~to_zero
            new_arr.append(new_row)
        
        return np.array(new_arr)
    
    bncSer = mySF['bounce'].to_series()
    
    outDict = {}
    prev_hit_frame, prev_hit_flag, prev_shape = None, None, None
    for next_hit_frame, next_hit_flag in mySF['hitFlag'].dropna().items():
        which = next_hit_flag[:3]
        
        if which == 'top':    
            which_long = 'Top'
            next_shape = mySF[f'topPlayer_3D'][next_hit_frame]
        elif which == 'bot':
            which_long = 'Bottom'
            next_shape = mySF[f'botPlayer_3D'][next_hit_frame]
        else: ### in case of inferred hit point
            which, which_long = None, None    
            next_shape = None
            
        if prev_hit_frame is not None:
            frames = range(prev_hit_frame, next_hit_frame + 1) 
            
            if which in ['top', 'bot']:
                hit_point = prev_shape.get_ground_point()
                rec_point = next_shape.get_ground_point()
                
                baseLn, netLn = schemaLines[f'{which_long} Baseline'], schemaLines['Net Line']
                crt_dist = netLn.distance_to(baseLn.center())
                ply_dist = netLn.distance_to(rec_point)
                
                ### calculate priors
                lmb = ply_dist / crt_dist    
                p0, p1, p2 = prior_dist(lmb)
            else:
                p0, p1, p2 = 0.1, 0.7, 0.2 ### this happens if the next hit frame is the inferred end of flight frame
                
            ### Conditional bounce probabilities for one and two bounces: given that it bounces once or twice...
            ### ... what is the probability that it bounces in this exact frame?
            pArr1 = bncSer.reindex(frames).values
            pArr1[0] = 0 ### assumption being -> can not bounce in hit frame (after hit)
            pArr2 = pArr1[np.newaxis, :] * pArr1[:, np.newaxis]
            
            ### make it so that two bounces can not occur within offset frames of each other
            pArr2 = set_broad_diagonal_to_zero(pArr2, offset = 5)
            
            ### normalize
            pArr1_norm = pArr1 / pArr1.sum()
            pArr2_norm = pArr2 / pArr2.sum()
        
            ### reduce to a single dimension -> len: frames!
            pArr2_norm = pArr2_norm.sum(axis = 1)    
            
            frmArr = np.array([pArr1_norm, pArr2_norm]).T
            pFrm = pd.DataFrame(frmArr, index = frames, columns = ['bounce_prob_cond1', 'bounce_prob_cond2'])
            pFrm['prior0'] = p0
            pFrm['prior1'] = p1
            pFrm['prior2'] = p2
        
            ### EDIT: Instead of a dense stream feature, we return a sparse stream feature
            ### with arrays pertaining to the current hit at the hit frame
            outDict[prev_hit_frame] = pFrm.values
            
        prev_hit_frame, prev_hit_flag, prev_shape = next_hit_frame, next_hit_flag, next_shape
    
    return StreamFeature(outDict, name = 'bounceArr')
  
    
def viterbi_trajectories(prob_coordSF, 
                         dst_factor = 1/6, dev_factor = 1/6, 
                         bnc_factor = 1/6, net_factor = 1/6,
                         fair_factor = 1/6, spin_factor = 1/6):
    
    ### figure out where we have gaps in the trajectory finder 
        # can only apply viterbi between these gaps, so that prev_lsArr and curr_fsArr are well defined
    all_hit_frames = prob_coordSF.get_frames()[:-1]
    frames_with_measurement = prob_coordSF.dropna().get_frames()
    hit_frames_without_measurement = [f for f in all_hit_frames if f not in frames_with_measurement]
    
    split_hit_frames_with_measurement = []
    curr_uninterrupted_frames = []
    for frame in all_hit_frames:
        if frame in frames_with_measurement:
            curr_uninterrupted_frames.append(frame)
        else:
            split_hit_frames_with_measurement.append(curr_uninterrupted_frames)
            curr_uninterrupted_frames = []
    
    if len(curr_uninterrupted_frames) > 0:
        split_hit_frames_with_measurement.append(curr_uninterrupted_frames)

    if len(frames_with_measurement) > 0:
        traj_coord_shapes = [arr.shape for arr in prob_coordSF['traj_coordArr'].values() if arr is not None]
        traj_coord_planes = [shape[0] for shape in traj_coord_shapes]
        traj_coord_initial_conditions = [shape[1] for shape in traj_coord_shapes]
        plane, traj = max(traj_coord_planes), max(traj_coord_initial_conditions)

    outDicts = []
    ### apply viterbi to each of these sub-blocks of uninterrupted rallying
    for hit_frames in split_hit_frames_with_measurement:
    
        num_hits = len(hit_frames)
        
        prev_lsArr = None
        dist_prob_dict = {}

        ### FORWARD PASS ###
        for idx, frame in enumerate(hit_frames):
            ### get arrays
            featDict = prob_coordSF[frame]
            bounceArr = featDict['bounce_probs']
            devArr = featDict['dev_probs']
            netArr = featDict['net_probs']
            fairArr = featDict['fair_probs']
            spinArr = featDict['spin_probs']
            coordArr = featDict['traj_coordArr']
            
            if prev_lsArr is None: ### initialize trellis
            
                trellis = np.full((num_hits, plane, traj), fill_value = np.nan)
                pointers = np.full((num_hits, plane, traj, 2), fill_value = np.nan, dtype = int) ### need two coordinates at the end to keep track of the previous plane, traj pair index
                
                log_devArr = np.log(devArr)
                log_bounceArr = np.log(bounceArr)
                log_netArr = np.log(netArr)
                log_fairArr = np.log(fairArr)
                log_spinArr = np.log(spinArr)
                
                dist_prob_dict[frame] = None

                dist_missing_correction = 1 / (1 - dst_factor)

                trellis[0] = dist_missing_correction * (dev_factor * log_devArr + bnc_factor * log_bounceArr + net_factor * log_netArr + fair_factor * log_fairArr + spin_factor * log_spinArr)
                # pointers[0] = np.zeros((plane, traj, 2)) 
            else:
                curr_fsArr = coordArr[:, :, 0, :]
                
                ### calculate the distance between 
                    # the end point of the previous trajectory
                    # the start point of the current trajectory
                distArr = prev_lsArr[:, :, np.newaxis, np.newaxis] - curr_fsArr[np.newaxis, np.newaxis, :, :]    
                distArr = (distArr ** 2).sum(axis = -1) ** 0.5
                ### -> shape: plane, traj, plane, traj
                
                ### calculate the 'probability of transition' from the distances
                stretch_factor = 1 / const.SCALE
                likelihood_norm_in_px = 24 ### this is a yard (three feet) in voxel
                dist_probArr = 1 / (1 + np.exp(-stretch_factor * -(distArr - likelihood_norm_in_px)))
                dist_prob_dict[frame] = dist_probArr ### have to keep this for later, since we don't have it in a featDict

                log_dist_probArr = np.log(dist_probArr)
                log_devArr = np.log(devArr)
                log_bounceArr = np.log(bounceArr)
                log_netArr = np.log(netArr)
                log_fairArr = np.log(fairArr)
                log_spinArr = np.log(spinArr)
                
                ### combine the distance probabilities with the current bounce audio and deviation probabilities
                log_probArr = dst_factor * log_dist_probArr + dev_factor * log_devArr + bnc_factor * log_bounceArr + net_factor * log_netArr + fair_factor * log_fairArr + spin_factor * log_spinArr
                
                ### combine it with the trellis likelihood: -> what is the likelihood of being in the previous trajectory in the first place
                trellis_brdcst = trellis[idx - 1, :, :, np.newaxis, np.newaxis]
                probArr = log_probArr + trellis_brdcst

                ### add some dummy log probabilities in case some of the planes have been culled 
                ### (because they don't contain enough observations)
                plane_prev, plane_curr = prev_lsArr.shape[0], curr_fsArr.shape[0]
                prev_to_add = plane - plane_prev
                curr_to_add = plane - plane_curr

                prev_to_add_probArr = np.full((prev_to_add, traj, plane_curr, traj), -np.inf) ### fill by -np.inf -> will never be chosen!
                probArr = np.concatenate([probArr, prev_to_add_probArr], axis = 0)

                curr_to_add_probArr = np.full((plane, traj, curr_to_add, traj), -np.inf) ### fill by -np.inf -> will never be chosen!
                probArr = np.concatenate([probArr, curr_to_add_probArr], axis = 2)

                ### now, we can reshape to consistent plane/traj shape, since plane prev and plane curr have been rectified.
                probArr_flat = probArr.reshape((plane * traj, plane, traj))
                maxima = probArr_flat.max(axis = 0)
                
                argmaxs_flat = probArr_flat.argmax(axis = 0)
                plane_idxs, traj_idxs = np.unravel_index(argmaxs_flat, (plane, traj))
                maxima_indices = np.stack([plane_idxs, traj_idxs], axis = -1)
        
                trellis[idx] = maxima
                pointers[idx] = maxima_indices
        
            prev_lsArr = coordArr[:, :, -1, :]
        
        ### BACKWARD PASS ###
        best_trajectories = np.full((num_hits, 2), fill_value = np.nan, dtype = int)
        for hit in range(num_hits, 0, -1):
            if hit == num_hits:
                best_plane, best_traj = ut.argmax_multidim(trellis[-1])
            else:
                best_plane_before, best_traj_before = best_trajectories[hit]
                best_plane, best_traj = pointers[hit, best_plane_before, best_traj_before]
            best_trajectories[hit-1] = best_plane, best_traj
            
        ### CREATE THE UNINTERRUPTED HIT SEGMENT DICTIONARIES (best indices and probabilities)
        outDict = {}
        for idx, frame in enumerate(hit_frames):
            best_plane, best_traj = best_trajectories[idx]
            outDict[frame] = {'best_plane': best_plane, 'best_traj': best_traj}
            
            featDict = prob_coordSF[frame]
            bounceArr = featDict['bounce_probs']
            devArr = featDict['dev_probs']
            netArr = featDict['net_probs']
            fairArr = featDict['fair_probs']
            spinArr = featDict['spin_probs']

            outDict[frame]['bounce_prob'] = bounceArr[best_plane, best_traj]
            outDict[frame]['dev_prob'] = devArr[best_plane, best_traj]
            outDict[frame]['net_prob'] = netArr[best_plane, best_traj]
            outDict[frame]['fair_prob'] = fairArr[best_plane, best_traj]
            outDict[frame]['spin_prob'] = spinArr[best_plane, best_traj]
            
            if idx > 0:
                dist_probArr = dist_prob_dict[frame]
                prev_best_plane, prev_best_traj = best_trajectories[idx - 1]
                outDict[frame]['dist_prob'] = dist_probArr[prev_best_plane, prev_best_traj, best_plane, best_traj]
            else:
                outDict[frame]['dist_prob'] = None

        outDicts.append(outDict)
        
    allDict = {k: v for outDict in outDicts for k, v in outDict.items()}
    
    ### add in None for frames without measurement
    for f in hit_frames_without_measurement:
        allDict[f] = {'best_plane': None,
                      'best_traj': None,
                      'bounce_prob': None,
                      'dev_prob': None,
                      'net_prob': None,
                      'fair_prob': None,
                      'spin_prob': None,
                      'dist_prob': None}
    
    return StreamFeatures(allDict)    
    

def make_trajectory_frame(traj_evtSF, courtLines = None):
    ### features: outArr, columns, srvFlag, hitFlag
    
    if courtLines is None:
        courtLines = CourtLines.make().add_singles_netline()
    
    hit_frames = traj_evtSF.get_frames()
    
    hitDFs = []
    for idx, frame in enumerate(hit_frames):
        featDict = traj_evtSF[frame]
        outArr, columns, srvFlag, hitFlag = featDict['outArr'], featDict['columns'], featDict['srvFlag'], featDict['hitFlag']
        
        if outArr is not None:
            ### Initialize DataFrame with outArr from sa.make_trajectory_arrays
            hitDF = pd.DataFrame(outArr, columns = columns, index = range(frame, frame + outArr.shape[0]))
            
            ### Initialize net and bounce booleans
            hitDF['is_net'] = hitDF['is_net'].astype(bool)
            hitDF['is_bounce'] = np.where(hitDF.z_t == 0, True, False)
            
            ### Determine Hit and Serve Side
            side, desc = hitFlag.split('_', 1)
            if srvFlag is not None:
                srv_side = srvFlag[2]
            else:
                srv_side = None
    
            ### Check if bounce was in or out
            if srv_side is None:
                inReg = courtLines.get_half_region('Bottom' if side == 'top' else 'Top', width = 'Singles')
            else:
                inReg = courtLines.get_service_box('Bottom' if side == 'top' else 'Top', which = 'Deuce' if srv_side == 'D' else 'Ad')
                
            bounce_coords = hitDF.loc[hitDF['is_bounce'], ['x_t', 'y_t']]
            
            if not bounce_coords.empty:
                def coords_in_region(coords):
                    return Point2D(coords).within(inReg)
            
                hitDF['is_bounce_in'] = bounce_coords.apply(coords_in_region, axis = 1, raw = True)
            else:
                hitDF['is_bounce_in'] = np.nan
                    
            ### Construct Event Column
            hitDF['event'] = np.where(hitDF.is_bounce, 
                                        np.where(hitDF.is_bounce_in, 'BI', 'BO'),
                                        np.where(hitDF.is_net, 'N', 'F'))
            if srvFlag is not None:
                hitDF.loc[frame, 'event'] = srvFlag
            else:
                hitDF.loc[frame, 'event'] = 'H' + hitFlag[0].capitalize()
    
            ### Add event descriptor
            hitDF['event_descriptor'] = desc
            
            ### add hand and player name
            if side == 'top':
                player_name, player_hand = featDict['top_name_Flag'], featDict['top_hand_Flag']
            elif side == 'bot':
                player_name, player_hand = featDict['bot_name_Flag'], featDict['bot_hand_Flag']

            hitDF['player_name'] = player_name
            hitDF['player_hand'] = player_hand

            ### drop unnecessary booleans
            hitDF = hitDF.drop(columns = ['is_net', 'is_bounce', 'is_bounce_in'])
            
            ### add a shot index to identify the shot
            hitDF['shot_idx'] = idx
            
            ### kill last shot in index -> it is the next hit frame
            hitDF = hitDF.iloc[:-1]
            
            ### Append to list to be returned
            hitDFs.append(hitDF)
    
    if len(hitDFs) > 0:
        return pd.concat(hitDFs)

def create_propagation_models(hitSF, courtLines = None):
    
    def calculate_speeds_and_acceleration(cnt1, cnt2, cnt3, frm1, frm2, frm3):
        ### given three subsequent contours, calculate their speed and acceleration per frame
        try:
            speed1 = (cnt2.center() - cnt1.center()) / (frm2 - frm1)
        except:
            speed1 = None
        
        try:
            speed2 = (cnt3.center() - cnt2.center()) / (frm3 - frm2)
        except:
            speed2 = None
        
        try:
            accel = speed2 - speed1
        except:
            accel = None
            
        return speed1, speed2, accel
    
    topPt = courtLines['Top Baseline'].center()
    botPt = courtLines['Bottom Baseline'].center()
    topCnt = Contour.from_point(topPt)
    botCnt = Contour.from_point(botPt)

    mySF = hitSF.dropna()
    cntSF = mySF['hitCnt']
    flagSF = mySF['hitFlag']

    bDict = {}
    hit_frames = mySF.get_frames()
    prev_hit_frame, prev_hit_cnt, prev_hit_flag = None, None, None
    curr_hit_frame, curr_hit_cnt, curr_hit_flag = None, None, None
    for next_hit_frame in hit_frames:
        ### Initialize Next Hit Variables
        next_hit_cnt = cntSF[next_hit_frame]
        next_hit_flag = flagSF[next_hit_frame]
        
        ### if the next hit contour is not available, replace it with the default hit position:
        ###    -> the middle of the baseline on the other side.
        next_hit_side = next_hit_flag.split('_')[0]
        if next_hit_cnt is None:
            next_hit_cnt = topCnt if next_hit_side == 'top' else botCnt

        ### Calculate the Propagation Matrices
        if curr_hit_frame is not None:
            v1, v2, a = calculate_speeds_and_acceleration(prev_hit_cnt, curr_hit_cnt, next_hit_cnt, 
                                                            prev_hit_frame, curr_hit_frame, next_hit_frame)

            ### Edit: numerically unsafe to work with v1 -> can be zero in a dimension (esp x), and then we divide by zero
            ### -> therefore, do all matrices A like the serve matrix, independent of v1 and dependent instead on hit point
            ### Edit 2: Instead of passing in As, which suffer in general numerical issues of multiplication, we adjust the
            ### particle filter so that it allows taking As and Bs like so: S1 = A * S0 + B.
            ### In this way, at model change points, we feed in Bs, and As are all zero!
            ### At flight points, we feed in As as usual, and Bs are zero
            B = np.array([0, 0, v2.x, v2.y])
            bDict[curr_hit_frame] = B

        prev_hit_frame, prev_hit_cnt, prev_hit_flag = curr_hit_frame, curr_hit_cnt, curr_hit_flag
        curr_hit_frame, curr_hit_cnt, curr_hit_flag = next_hit_frame, next_hit_cnt, next_hit_flag
        del next_hit_frame, next_hit_cnt, next_hit_flag ### not strictly necessary, but cleaner

    ### Deal with Last Shot in Rally
        # -> no next hit!
    next_hit_frame = curr_hit_frame + 25
    curr_hit_side = curr_hit_flag.split('_')[0]
    next_hit_cnt = topCnt if curr_hit_side == 'bot' else botCnt

    v1, v2, a = calculate_speeds_and_acceleration(prev_hit_cnt, curr_hit_cnt, next_hit_cnt, 
                                                    prev_hit_frame, curr_hit_frame, next_hit_frame)
    B = np.array([0, 0, v2.x, v2.y])
    bDict[curr_hit_frame] = B

    return StreamFeature(bDict, name = 'B')

def widen_propagation_models(bSF, share_kernel = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]):
    ### The model switch may not just occur at the exact hit frame, it may also occur some frames around it.
    ### Therefore, we widen the scope of the switch propagation matrices A and B to the frames around it. 
    ### We also have to make sure that the widened frames don't overlap 
        # -> for the sake of the particle filter, there has to be at least one normal flying motion frame between potential model switches.
    hit_frames = bSF.get_frames()
    

    A_still = np.zeros((4, 4))
    A_still[0, 0] = 1
    A_still[1, 1] = 1

    len_k = len(share_kernel)
    central_idx = len_k // 2


    wideDict = {}
    for hit_index, hit_frame in enumerate(hit_frames):
        kernel_array = np.array(share_kernel)
        
        ### Check Distances to Previous and Next Hit, and de-overlap the kernels if necessary!
        ### Note that this way, each kernel is de-overlapped twice, symmetrically 
            # -> once from before to current, and once from current to next.
        if hit_index != 0:
            hit_bef = hit_frames[hit_index - 1]

            distance_between_kernels = hit_frame - hit_bef - len_k
            if distance_between_kernels > 0:
                pass
            elif distance_between_kernels in [0, -1]: 
                ### for a kernel of length 7, this happens when consecutive hits are 7 or 6 frames apart respectively.
                ### in this case, it suffices to shift the leftmost kernel value inward.
                kernel_array = kernel_array[1:] + kernel_array[0] / (len_k - 1)
            elif distance_between_kernels in [-2, -3]:
                ### for a kernel of length 7, this happens when consecutive hits are 5 or 4 frames apart respectively.
                ### in this case, we shift the two leftmost kernel values inward
                kernel_array = kernel_array[2:] + kernel_array[:2].sum() / (len_k - 2)
            else:
                raise Exception(f'Hit Frames {hit_bef} and {hit_frame} to close to each other!')
                
        if hit_index != len(hit_frames) - 1:
            hit_aft = hit_frames[hit_index + 1]

            distance_between_kernels = hit_aft - hit_frame - len_k
            if distance_between_kernels > 0:
                pass
            elif distance_between_kernels in [0, -1]: 
                ### for a kernel of length 7, this happens when consecutive hits are 7 or 6 frames apart respectively.
                ### in this case, it suffices to shift the rightmost kernel value inward.
                kernel_array = kernel_array[:-1] + kernel_array[-1] / (len_k - 1)
            elif distance_between_kernels in [-2, -3]:
                ### for a kernel of length 7, this happens when consecutive hits are 5 or 4 frames apart respectively.
                ### in this case, we shift the two rightmost kernel values inward
                kernel_array = kernel_array[:-2] + kernel_array[-2:].sum() / (len_k - 2)
            else:
                raise Exception(f'Hit Frames {hit_frame} and {hit_aft} to close to each other!')
        
        B = bSF[hit_frame]
        for idx, val in enumerate(kernel_array):
            curr_frame = hit_frame - central_idx + idx
            wideDict[curr_frame] = {'A': A_still, 'B': B, 'weight': val} 

    return StreamFeatures(wideDict)


def viterbi_hit_events(conv_hitSF, 
                        initial_state_probabilities = [0.9], hit_state_probabilities = [0, 0, 0],
                        flight_state_probabilities = [0.9], exit_state_probabilities = [1],
                        initial_emission_exponents = [1], flight_emission_exponents = [1], exit_emission_exponents = [1],
                        high_hit_emission_exponents = [1, 1, 1], low_hit_emission_exponents = [1, 1, 1]):
    np.seterr(divide = 'ignore')

    def get_A(initial_state_probabilities, hit_state_probabilities, flight_state_probabilities, exit_state_probabilities):
        ### for each event class in init, hit, flight, state, we supply a list containing probabilities.
        ### these probabilities denote the probability of remaining in a state associated with an event class, as opposed to moving to the next
        ### a list of [0.9, 0.25, 0] denotes a three state system where 
            # a process will remain in the first state with p = 0.9, or move to the second state with p = 0.1, upon which
            # the process will remain in the second state with p = 0.25, or move to the third state with p = 0.75, upon which
            # the process will move from the third state automatically out.
        ### this function builds the transition matrix A resulting from these lists automatically.
        ### as far as I can see, if the emission probabilities for init, hit, flight and exit are the same across the constituent states,
        ###     having more than one state is useful only to add buffer states that are traversed quickly (even with p = 0),
        ###     to basically enforce a minimum number of frames that a state has to linger somewhere.
        ###     in this sense, having different emission probabilities for different states within an event class might be useful!
        num_init = len(initial_state_probabilities)
        num_hit = len(hit_state_probabilities)
        num_flight = len(flight_state_probabilities)
        num_exit = len(exit_state_probabilities) 

        ### ordering of event classes: init, hit_top, hit_bot, flight_up, flight_down, exit
        num_states = num_init + num_hit * 2 + num_flight * 2 + num_exit 
        A = np.zeros((num_states, num_states))

        def transitions_within_event_class(subA, state_probabilities, col_offset):
            ps, ps_c = state_probabilities, [1 - p for p in state_probabilities]
            rows, cols = subA.shape
            idxs = np.arange(rows)
            subA[idxs, idxs + col_offset] = ps ### probabilities to remain in state
            subA[idxs[:-1], idxs[:-1] + col_offset + 1] = ps_c[:-1] ### transition probabilities within event classes 
            return subA

        
        col_offset = 0
        ### INIT ###
        # a) make partial matrix
        A_init = np.zeros((num_init, num_states))
        A_init = transitions_within_event_class(A_init, initial_state_probabilities, col_offset)
        A_init[-1, [num_init, num_init + num_hit]] = (1 - initial_state_probabilities[-1]) / 2 ### into hit_bot or hit_top
        # b) make statedict
        init_dict = {i + col_offset: f'init_{i}' for i in range(len(initial_state_probabilities))}
        # c) increment row offset
        col_offset += num_init
        
        ### HIT TOP ###
        # a) make partial matrix
        A_hit_top = np.zeros((num_hit, num_states))
        A_hit_top = transitions_within_event_class(A_hit_top, hit_state_probabilities, col_offset)
        A_hit_top[-1, num_init + 2 * num_hit + num_flight] = (1 - hit_state_probabilities[-1]) ### into initial state of flight down
        # b) make statedict
        hit_top_dict = {i + col_offset: f'hittop_{i}' for i in range(len(hit_state_probabilities))}
        # c) increment row offset
        col_offset += num_hit

        ### HIT BOT ###
        # a) make partial matrix
        A_hit_bot = np.zeros((num_hit, num_states))
        A_hit_bot = transitions_within_event_class(A_hit_bot, hit_state_probabilities, col_offset)
        A_hit_bot[-1, num_init + 2 * num_hit] = (1 - hit_state_probabilities[-1]) ### into initial state of flight down
        # b) make statedict
        hit_bot_dict = {i + col_offset: f'hitbot_{i}' for i in range(len(hit_state_probabilities))}
        # c) increment row offset
        col_offset += num_hit

        ### FLIGHT UP ###
        # a) make partial matrix
        A_flight_up = np.zeros((num_flight, num_states))
        A_flight_up = transitions_within_event_class(A_flight_up, flight_state_probabilities, col_offset)
        A_flight_up[-1, num_init] = (1 - flight_state_probabilities[-1])
        # b) make statedict
        flight_up_dict = {i + col_offset: f'flightup_{i}' for i in range(len(flight_state_probabilities))}
        # c) increment row offset
        col_offset += num_flight

        ### FLIGHT DOWN ###
        # a) make partial matrix
        A_flight_down = np.zeros((num_flight, num_states))
        A_flight_down = transitions_within_event_class(A_flight_down, flight_state_probabilities, col_offset)
        A_flight_down[-1, num_init + num_hit] = (1 - flight_state_probabilities[-1])
        # b) make statedict
        flight_down_dict = {i + col_offset: f'flightdown_{i}' for i in range(len(flight_state_probabilities))}
        # c) increment row offset
        col_offset += num_flight

        ### EXIT
        # a) make partial matrix
        A_exit = np.zeros((num_exit, num_states))
        A_exit = transitions_within_event_class(A_exit, exit_state_probabilities, col_offset)
        # b) make statedict
        exit_dict = {i + col_offset: f'exit_{i}' for i in range(len(exit_state_probabilities))}

        ### JOIN
        A = np.vstack((A_init, A_hit_top, A_hit_bot, A_flight_up, A_flight_down, A_exit))

        ### STATEDICT
        stateDict = {**init_dict, **hit_top_dict, **hit_bot_dict, **flight_up_dict, **flight_down_dict, **exit_dict}
        
        return A, stateDict

    def get_emission_probabilities(q_top, q_bot, 
                                   initial_emission_exponents, flight_emission_exponents, exit_emission_exponents,
                                   high_hit_emission_exponents, low_hit_emission_exponents):
        
        q_nohit = (1 - q_top) * (1 - q_bot)

        ### HITS ###
        if q_top > q_bot:
            top_es = [1 - (1 - q_top) ** exp for exp in high_hit_emission_exponents]
            bot_es = [1 - (1 - q_bot) ** exp for exp in low_hit_emission_exponents]
        else:
            top_es = [1 - (1 - q_top) ** exp for exp in low_hit_emission_exponents]
            bot_es = [1 - (1 - q_bot) ** exp for exp in high_hit_emission_exponents]

        ### INIT ###       
        init_es = [q_nohit ** exp for exp in initial_emission_exponents]

        ### FLIGHT ###
        flight_es = [q_nohit ** exp for exp in flight_emission_exponents]

        ### EXIT ###
        exit_es = [q_nohit ** exp for exp in exit_emission_exponents]

        return [*init_es, *top_es, *bot_es, *flight_es, *flight_es, *exit_es]

    conv_hitFrm = conv_hitSF.to_dataframe()
    frames = conv_hitFrm.index.to_list()

    A, stateDict = get_A(initial_state_probabilities, hit_state_probabilities, flight_state_probabilities, exit_state_probabilities)
    logA = np.log(A)        
    
    T, S = len(frames), len(A)

    ### FORWARD PASS ###
    trellis = np.full((T, S), fill_value = np.nan)
    pointers = np.full((T, S), fill_value = np.nan)
    for frame_idx, frame in enumerate(frames):

        pSer = conv_hitFrm.loc[frame]

        if frame_idx == 0: ### initialize
            initial_probability_distribution = np.zeros(S)
            initial_probability_distribution[0] = 1
            trellis[frame_idx] = np.log(initial_probability_distribution)
            pointers[frame_idx] = np.zeros(S)
        else:
            q_top, q_bot = pSer['top_hit_p'], pSer['bot_hit_p']
            p_emit = get_emission_probabilities(q_top, q_bot, 
                                                initial_emission_exponents, flight_emission_exponents, exit_emission_exponents,
                                                high_hit_emission_exponents, low_hit_emission_exponents)
            log_p_emit = np.log(p_emit)

            ### Get the Log Probabilities of Transition and Emission
            log_p_trans = trellis[frame_idx - 1][:, np.newaxis] + logA
            log_p_all = log_p_trans + log_p_emit[np.newaxis, :]

            # ### Probability that a trellis probability transition between periods i and j = i + 1
            # p_trans = np.einsum('i, ij -> ij', trellis[frame_idx - 1], A) ### shape: 
            # ### Probability that the new state emits B
            # p_all = np.einsum('ij, j -> ij', p_trans, p_emit)

            trellis[frame_idx] = np.max(log_p_all, axis = 0)
            pointers[frame_idx] = np.argmax(log_p_all, axis = 0)

    ### BACKWARD PASS ###
    states = np.full(T, fill_value = 999, dtype = int)
    for t in range(T, 0, -1):
        if t == T:
            s_before = np.argmax(trellis[-1]) ### initialization with highest final trellis probability state
        else:    
            s_before = pointers[t, states[t]]
        states[t-1] = s_before

    states_mapped = {frames[i]: stateDict[s] for i, s in enumerate(states)}
    stateSer = pd.Series(states_mapped)
    stateSer = stateSer[stateSer.str.split('_', expand = True)[0].isin(['hittop', 'hitbot'])].rename('hitFlag')

    np.seterr(divide = 'warn') 

    outSF = StreamFeature.from_series(stateSer)
    return outSF

def specify_hit_events(mySF):
    myFrm = mySF.to_dataframe()
    myFrms = ut.split(myFrm)

    extra_cols = ['bot_name_Flag', 'bot_hand_Flag', 'top_name_Flag', 'top_hand_Flag']

    outDict = {}
    for frm in myFrms:
        unspecific_events = frm['hitFlag'].str.split('_', expand = True)[0].unique()
        assert len(unspecific_events) == 1
        unspecific_event = unspecific_events[0]
        side = unspecific_event[3:]
        side_columns = [c for c in frm if side in c and not 'oth' in c and not c in extra_cols]
        hand_column = [c for c in frm if side in c and c in extra_cols and 'hand' in c]

        assert len(hand_column) == 1
        hand_column = hand_column[0]
        hand = frm[hand_column].unique()
        assert len(hand) == 1
        hand = hand[0]

        ### average event probabilities over the duration of the hit
        average_event_probabilities = frm[side_columns].mean() 

        ### depending on the hand, we have to flip the wrong-handed strokes and combine them with their correctly-handed counterpart
        lf_probs = average_event_probabilities[average_event_probabilities.index.str.contains('_lf')]
        rt_probs = average_event_probabilities[average_event_probabilities.index.str.contains('_rt')]

        lf_probs.index = lf_probs.index.str.replace('_lf', '')
        rt_probs.index = rt_probs.index.str.replace('_rt', '')

        if hand == 'right':
            base_probs = rt_probs.copy()
            repl_probs = lf_probs.copy()
            repl_probs.index = repl_probs.index.str.replace('_bh', '_chng').str.replace('_fh', '_bh').str.replace('_chng', '_fh') ### triple swap to avoid replacing same row twice

        elif hand == 'left':
            base_probs = lf_probs.copy()
            repl_probs = rt_probs.copy()
            repl_probs.index = repl_probs.index.str.replace('_bh', '_chng').str.replace('_fh', '_bh').str.replace('_chng', '_fh') ### triple swap to avoid replacing same row twice

        probs = base_probs + repl_probs
        
        ### create the event aggregators -> have to map the very granular events to something more usable
        srv_or_ov = [i for i in probs.index if '_ov' in i or '_srv' in i]
        gs_or_vl = [i for i in probs.index if i not in srv_or_ov] 

        p_srv_or_ov = probs[srv_or_ov].sum()
        
        ps_gs_or_vl = probs[gs_or_vl]
        ps_gs_or_vl.index = ps_gs_or_vl.index.str.replace('_gs', '').str.replace('_vl', '')
        ps_gs_or_vl = ps_gs_or_vl.groupby(level = 0).sum()

        max_gs_or_vl, p_gs_or_vl = ps_gs_or_vl.index[ps_gs_or_vl.argmax()], ps_gs_or_vl.max()
        
        if len(outDict) == 0: ### need a serve!
            if p_srv_or_ov > p_gs_or_vl:
                outDict.update({frame: f'{side}_srv' for frame in frm.index})
            else:
                print(f'{mySF} -> did not see serve at beginning!')
        else: ### anything that looks like a serve is an overhead!
            if p_srv_or_ov > p_gs_or_vl:
                outDict.update({frame: f'{side}_ovh' for frame in frm.index})
            else:
                outDict.update({frame: max_gs_or_vl for frame in frm.index})

    return StreamFeature(outDict, name = 'hitFlag')

def count_remaining_shots(mySF):
    frames = mySF.get_frames()
    num_shots = len(frames) - 1 ### noting that the last frame in the SF is an added frame for the calculation of the last shot
    remaining_shots = {f: num_shots - i for i, f in enumerate(frames)}
    return StreamFeature(remaining_shots, name = 'remaining_shots')