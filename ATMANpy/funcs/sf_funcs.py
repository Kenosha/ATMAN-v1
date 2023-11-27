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
        
        top_dist_norms = []
        bot_dist_norms = []
        areas = []
        for frameCnt in frameCnts:
            
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
            
            # c) normalize
            top_dist_max = FDCI_LUT[frame]['top_dist_max']
            bot_dist_max = FDCI_LUT[frame]['bot_dist_max']
            top_dist_norm = top_dist / top_dist_max
            bot_dist_norm = bot_dist / bot_dist_max
    
            top_dist_norms.append(top_dist_norm)
            bot_dist_norms.append(bot_dist_norm)
    
            ### to calculate area...
            area = frameCnt.area()
            areas.append(area)
            
        FDCD_LUT[frame] = {'top_dist_norm': top_dist_norms,
                           'bot_dist_norm': bot_dist_norms,
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
        # top_speeds = []
        # bot_speeds = []
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
    topDF = metricDF[['top_dist_score', 'ovlp_score', 'move_score', 'area_score']]
    topDF.columns = [col.replace('top_', '').replace('_score', '') for col in topDF.columns]
    
    botDF = metricDF[['bot_dist_score', 'ovlp_score', 'move_score', 'area_score']]
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
        try:
            rightCnt = plSF[right_frame]
        except KeyError: ### in this case, there is no right player, so we don't interpolate.
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

    
