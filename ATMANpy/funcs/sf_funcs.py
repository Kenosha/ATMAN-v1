# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:00:53 2023

Here we write functions here that take a stream feature and return a stream feature, 
if the feature in question lends itself to vectorization or other speed ups 
that call for treating all features as a whole, across frames 

@author: gianl
"""

import pandas as pd
import numpy as np

import copy

import ATMANpy.auxil.constants as const

from ATMANpy.types.Features import StreamFeatures, StreamFeature, BoolFeatures, BoolFeature, SFList
from ATMANpy.types.Streams import Stream
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Region
from ATMANpy.types.Contours import Contours

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

def split_player_candidate_tracks(plSF):

    ### get data from last frame
    lastFrm, lastDict = plSF.last()    
    lastCnts, lastPars = lastDict['player_candidates'], lastDict['player_parents']
    
    ### initialize list of tracks
    last_trackList = []
    for lastCnt, lastPar in zip(lastCnts, lastPars):
        if lastPar == []:
            trackDict = {'contour': lastCnt, 'parent': None}
            last_trackList.append({lastFrm: trackDict})
        else:
            for par in lastPar:
                trackDict = {'contour': lastCnt, 'parent': par}
                last_trackList.append({lastFrm: trackDict})
         
    ### iterate over frames, backward, from penultimate
    other_frames = list(reversed(plSF.get_frames()[:-1]))

    for frame in other_frames:
        featDict = plSF[frame]
        currCnts = featDict['player_candidates']
        currPars = featDict['player_parents']
        
        curr_indices = []
        curr_trackList = [] ### this list of tracks is populated in the loop
        for last_track in last_trackList:
            
            ### Handle previously finished tracks ###
            try:
                childDict = last_track[frame + 1]
            except KeyError: 
                ### in this case, the track was previously completed from end to beginning, so we add it in unmodified
                curr_trackList.append(last_track)
                continue
            
            currIdx = childDict['parent']
            curr_indices.append(currIdx)
            
            ### Handle just finished tracks ###
            if currIdx is None: ### in this case, the track is just completed, so we add it in unmodified.
                curr_trackList.append(last_track)
                continue
            
            ### Handle tracks with a continuation in next frame (parents to a child)
            currCnt = currCnts[currIdx]
            currPar = currPars[currIdx]
            
            if currPar == []: ### in this case, the track has no parent, and so it terminates here (seen backwards; seen forwards, it spawns here).
                curr_track = copy.deepcopy(last_track)
                currDict = {'contour': currCnt, 'parent': None}
                curr_track[frame] = currDict
                curr_trackList.append(curr_track)
                
            else:
                for par in currPar:
                    curr_track = copy.deepcopy(last_track)
                    currDict = {'contour': currCnt, 'parent': par}
                    curr_track[frame] = currDict
                    curr_trackList.append(curr_track)
                
        ### Handle tracks which are just begun: they are not pointed to by a child (seen forwards, they end here!)
        current_childless = [idx for idx in range(currCnts.length()) if idx not in curr_indices]
        
        for currIdx in current_childless:
            currCnt = currCnts[currIdx]
            currPar = currPars[currIdx]
            
            if currPar == []: ### in this case, the track has no parent, and so it terminates here (seen backwards; seen forwards, it spawns here).
                ### it's childless and parentless -> appears and disappears.
                currDict = {'contour': currCnt, 'parent': None}
                curr_track = {frame: currDict}
                curr_trackList.append(curr_track)
            else:
                for par in currPar:
                    currDict = {'contour': currCnt, 'parent': par}
                    curr_track = {frame: currDict}
                    curr_trackList.append(curr_track)
                    
        last_trackList = copy.deepcopy(curr_trackList)
        
    

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
            
            if cap is not None: 
                ### show the previous and next position, speed and acceleration by making three contours respectively.
                befPt = Point2D((dataDict['p_b_x'], dataDict['p_b_y']))
                befSp = Point2D((dataDict['s_b_x'], dataDict['s_b_y']))
                befAc = Point2D((dataDict['a_b_x'], dataDict['a_b_y'])) 
                
                aftPt = Point2D((dataDict['p_a_x'], dataDict['p_a_y']))
                aftSp = Point2D((dataDict['s_a_x'], dataDict['s_a_y']))
                aftAc = Point2D((dataDict['a_a_x'], dataDict['a_a_y'])) 
                
                befPt2 = befPt - befSp
                befPt3 = befPt2 - befSp - befAc
                
                aftPt2 = aftPt + aftSp
                aftPt3 = aftPt2 + aftSp + aftAc 
            
                gFrm = Stream.from_SF(segSF, cap, target_color_space = 'grey').get(fsFrm, put = False)
                cFrm = gFrm.to_BGR()
                cFrm = cFrm.put(befPt, color = (0, 255, 0)).put(befPt2, color = (0, 200, 0)).put(befPt3, color = (0, 150, 0))
                cFrm = cFrm.put(aftPt, color = (0, 0, 255)).put(aftPt2, color = (0, 0, 200)).put(aftPt3, color = (0, 0, 150))
                
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
    return ballSF.fillna(itplSF)

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

    
