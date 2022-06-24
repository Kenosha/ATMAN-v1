# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:03:54 2022

After the creation of play sequences and stream features, 
stream features are post processed 

@author: gianl
"""

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

# image operation
import cv2 as cv
import numpy as np

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

###### Court Line Normalization

from copy import deepcopy
from ATMANpy.funcs.court_funcs import find_courtlines

def get_point_arrays(sf):
    tlList, trList, blList, brList = [], [], [], []
    
    for frame, featDict in sf.items():
        myLines = find_courtlines(featDict['H'], schemaLines)
        tl, tr, bl, br = myLines.get_court_region().to_points()

        tlList.append(tl.coords)
        trList.append(tr.coords)
        blList.append(bl.coords)
        brList.append(br.coords)

    return np.array(tlList), np.array(trList), np.array(blList), np.array(brList)


def get_point_bounds(pointArr, 
                     lower_q = 0.2, upper_q = 0.8, acceptable_multiple = 3,
                     min_box_diameter = 4):
    
    qs = [lower_q, 0.5, upper_q]
    loPt, mdPt, hiPt = np.round(np.quantile(pointArr, qs, axis = 0)).astype('int')
    loPt, mdPt, hiPt = Point(loPt), Point(mdPt), Point(hiPt)

    ### interquantile distance
    iq_up, iq_lo = hiPt - mdPt, mdPt - loPt
    
    ### lower bind our interquantile distances, so we don't have tiny boxes.
    iq_up = iq_up.bind_below(min_box_diameter // 2)
    iq_lo = iq_lo.bind_below(min_box_diameter // 2)

    lower_bound = mdPt - iq_lo * acceptable_multiple - 1 ### subtract/add one because checking whether
    upper_bound = mdPt + iq_up * acceptable_multiple + 1 ### a point is inside region is checked STRONGLY.

    return lower_bound, upper_bound

def stream_features_median_corner_regions(sf, how_many_unilateral = 8):
    
    tlArr, trArr, blArr, brArr = get_point_arrays(sf)
    
    assert tlArr.shape[0] == trArr.shape[0] == blArr.shape[0] == brArr.shape[0]
    numFrames = tlArr.shape[0]
    
    out_sf = deepcopy(sf)
    
    for idx, frame in enumerate(out_sf):
        featDict = out_sf[frame]
        
        myLines = find_courtlines(featDict['H'], schemaLines)
        tl, tr, bl, br = myLines.get_court_region().to_points()

        ### calculate quantile court mask
        left_slice = max(0, idx - how_many_unilateral)
        right_slice = min(idx + how_many_unilateral + 1, numFrames)

        tlSlice = tlArr[left_slice:right_slice]
        trSlice = trArr[left_slice:right_slice]
        blSlice = blArr[left_slice:right_slice]
        brSlice = brArr[left_slice:right_slice]
        
        tl_lo, tl_hi = get_point_bounds(tlSlice)
        tr_lo, tr_hi = get_point_bounds(trSlice)
        bl_lo, bl_hi = get_point_bounds(blSlice)
        br_lo, br_hi = get_point_bounds(brSlice)
        
        featDict['tlReg'] = Region.from_points(tl_lo, tl_hi)
        featDict['trReg'] = Region.from_points(tr_lo, tr_hi)
        featDict['blReg'] = Region.from_points(bl_lo, bl_hi)
        featDict['brReg'] = Region.from_points(br_lo, br_hi)
        
        featDict['within_tlReg'] = tl.within(featDict['tlReg'])
        featDict['within_trReg'] = tr.within(featDict['trReg'])
        featDict['within_blReg'] = bl.within(featDict['blReg'])
        featDict['within_brReg'] = br.within(featDict['brReg'])
    
    return out_sf
    
def stream_features_correct_homography(sf):
    out_sf = deepcopy(sf)
    
    for idx, frame in enumerate(out_sf):
        featDict = out_sf[frame]
        
        within_tlReg = featDict['within_tlReg']
        within_trReg = featDict['within_trReg']
        within_blReg = featDict['within_blReg']
        within_brReg = featDict['within_brReg']
        
        #### check if maybe all points are within - then, we don't have to correct anything.
        if within_tlReg and within_trReg and within_blReg and within_brReg:
            continue
        
        myLines = find_courtlines(featDict['H'], schemaLines)
        tlPt, trPt, blPt, brPt = myLines.get_court_region().to_points()
        
        tlReg = featDict['tlReg']
        trReg = featDict['trReg']
        blReg = featDict['blReg']
        brReg = featDict['brReg']
        
        tlPt_med = tlReg.centroid.round_point()
        trPt_med = trReg.centroid.round_point()
        blPt_med = blReg.centroid.round_point()
        brPt_med = brReg.centroid.round_point()
        
        ### replace faulty points with the median point
        newPts = []
        
        # Top Left
        if not within_tlReg:
            newPts.append(tlPt_med)
        else:
            newPts.append(tlPt)
            
        ### Top Right
        if not within_trReg:
            newPts.append(trPt_med)
        else:
            newPts.append(trPt)   
        
        ### Bottom Left
        if not within_blReg:
            newPts.append(blPt_med)
        else:
            newPts.append(blPt)
        
        ### Bottom Right
        if not within_brReg:
            newPts.append(brPt_med)
        else:
            newPts.append(brPt)
        
        corrected_courtLines = Region.from_points(*newPts)
        corrected_H = schemaRegion.get_homography(corrected_courtLines)
    
        featDict['H'] = corrected_H
        
    return out_sf
    


#%% Split the loaded stream features if there is movement in frame.


def get_frames_warped(sf):    
    frmType = GreyFrame
    
    frmList = []
    for frmNum in sf:
        myFrm = frmType.from_video(cap, frmNum)
    
        H = sf[frmNum]['H']
        myFrm = myFrm.warp(np.linalg.inv(H), cols = schemaFrm.cols, rows = schemaFrm.rows)
    
        frmList.append(myFrm.array)
    
    frames = np.stack(frmList, axis = 0)
    # frames = np.swapaxes(np.swapaxes(frames, 2, 1), 1, 0) ### move last axis to front.
    
    return frames


def roll_safe(frames, offset):
    ### rolling without rolling over -> we shift the frames by offset, 
    ### but we do not reinsert at the other end
    ### instead, we repeat the first or the last frame.
    offsetSlice = - offset ## if we roll forward by 6, then we want
                                # i) the first frame, six times
                                # ii) all other frames except the last six.
    if offset == 0:
        return frames
    if offset > 0:
        return np.concatenate([np.stack([frames[0] for i in range(offset)]), frames[:offsetSlice]])
    else:
        return np.concatenate([frames[offsetSlice:], np.stack([frames[-1] for i in range(offsetSlice)])])


def diff_frames(frames, offsetList):
    
    min_offset = min(offsetList)
    max_offset = max(offsetList)
    
    outList = []
    for offset in offsetList:
        shift_frames = roll_safe(frames, offset)
        diffFrm = cv.absdiff(frames, shift_frames)
        outList.append(diffFrm)
    return np.stack(outList, axis = -1)
     

def detect_movement_contours(sf, cap = cap, save_all = True,
                             verbose = False):
        
    out_sf = deepcopy(sf)
    
    ###### PARAMETERS ######
    OFFSETLIST = [-8, -6, -4, 4, 6, 8]
    MOVEMENT_INTENSITY_THRESHOLD = 20 
    
    ### PREP DIFFERENCE FRAME ###
    frames = get_frames_warped(out_sf)   
    diffs = diff_frames(frames, OFFSETLIST)           

    joint_diff = (diffs > MOVEMENT_INTENSITY_THRESHOLD).all(axis = 3)  
    ####

    from ATMANpy.funcs.court_funcs import find_courtlines
    from ATMANpy.funcs.high_level_funcs import match_specific_burn_in_func    
        
    for idx, frmNum in enumerate(out_sf):
        featDict = out_sf[frmNum]
        
        cFrm = BGRFrame.from_video(cap, frmNum)
        hFrm = cFrm.to_HSV()
        
        H = featDict['H']
        courtLines = find_courtlines(H, schemaLines)
        playerTop = featDict['playerTop']
        playerBot = featDict['playerBot']
        ball = featDict['ball']
        scoreBoard = featDict['scoreBoard']
        
        motionMask_warped = BoolFrame((joint_diff[idx] * 255).astype('uint8'))
        motionMask = motionMask_warped.warp(H, cols = cFrm.cols, rows = cFrm.rows)
        
        courtlineMask = motionMask.zero().put(courtLines, thickness = 10) > 0
        
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
        # courtlineMask.dilate(kernel).show()
        
        def cntMask(frm, cnt, kernelsize = (20, 10)):
            ### Dilated player mask
            dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernelsize)
            try:
                return frm.within_contour(cnt).dilate(dil_kernel)
            except:
                return frm.within_region(cnt).dilate(dil_kernel) 
        
        if not playerTop is None:
            topMask = cntMask(cFrm, playerTop, kernelsize = (23, 11))
        else:
            topMask = BoolFrame(cFrm.to_grey().zero().array)
            # print('Lost Top Player Tracking!')
            
        if not playerBot is None:
            botMask = cntMask(cFrm, playerBot, kernelsize = (27, 13))
        else:
            botMask = BoolFrame(cFrm.to_grey().zero().array)
            # print('Lost Bottom Player Tracking!')
        
        if not ball is None:
            ballMask = cntMask(cFrm, ball, kernelsize = (7, 7))
        else:
            ballMask = BoolFrame(cFrm.to_grey().zero().array)
            # print('Lost Ball Tracking!')
        
        if not scoreBoard is None:
            scoreMask = cntMask(cFrm, scoreBoard, kernelsize = (10, 10))
        else:
            scoreMask = BoolFrame(cFrm.to_grey().zero().array)
            # print('Lost Scoreboard Tracking!')
        
        
        # ballMask.show()
        
        motionMask2 = motionMask & ~courtlineMask & ~topMask & ~botMask & ~ballMask & ~scoreMask
        # if verbose: motionMask2.show()
        
        # ### Adding a blur stage!
        # motionMask2 = motionMask2.gaussian_blur((7, 7))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        motionMask2 = motionMask2.open(kernel) >= 255
        # if verbose: motionMask2.show()
        
        ### Adding a closing stage
        ### large kernel, so that we join blobs belonging to the same ball kid.
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
        motionMask2 = motionMask2.close(kernel)
        # if verbose: motionMask2.show()
        
        burnMask = match_specific_burn_in_func(hFrm, H)
        
        #####
        finalMask = motionMask2 & burnMask
        # if verbose: finalMask.show()
        
        cts, _ = finalMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        myCts = [Contour(ct) for ct in cts]
        
        if save_all:
            featDict['movContours'] = myCts
        else:
            sortCts = sorted(myCts, key = lambda el: el.area())
            try:
                largestCnt = sortCts[-1]
            except:
                largestCnt = None
            featDict['movContours'] = largestCnt
            
    return out_sf

def calculate_contiguous_movement(sf):

    out_sf = deepcopy(sf)    

    ### PARAMETERS ###
    AREA_THRESHOLD = 600

    ### INIT ###
    current_movement_index = 0
    
    for frmNum, featDict in out_sf.items():
        
        myCts = featDict['movContours']
        
        if isinstance(myCts, Contour):
            myCts = [myCts]
        
        if not myCts is None:
            myCts = [ct for ct in myCts if ct.area() > AREA_THRESHOLD]
            # myCts.sort(key = lambda el: el.area())
            # try:
            #     largestCnt = myCts[-1].area()
            # except:
            #     largestCnt = 'None'

            if myCts != []:
                current_movement_index += 1
            else:
                current_movement_index = 0
        else:
            current_movement_index = 0
            
        featDict['movInt'] = current_movement_index 
    
    return out_sf
    

def drop_movement_frames(sf, verbose = False):
    
    ### PARAMETERS ###
    MINIMUM_LENGTH_OF_CONTINUOUS_MOVEMENT = 10
    KEEP_FIRST_AND_LAST_X_FRAMES = 3
        
    ### INIT ###
    prior_movement_index = 0
    frames_to_toss = [] ### the overall list of lists, containing all continuous segments of extraneous movement on court.
    current_frames_to_toss = [] ### a list containing a single contiguous segment of extraneous movement.
    
    for frmNum, featDict in sf.items():
        
        current_movement_index = featDict['movInt']
        
        ### SELECT FRAME FOR REMOVAL ###
        if current_movement_index > 0:
            current_frames_to_toss.append(frmNum)
            prior_movement_index = current_movement_index
        else:
            if prior_movement_index > MINIMUM_LENGTH_OF_CONTINUOUS_MOVEMENT:
            
                start_idx = KEEP_FIRST_AND_LAST_X_FRAMES
                stop_idx = - KEEP_FIRST_AND_LAST_X_FRAMES
                current_frames_to_toss = current_frames_to_toss[start_idx:stop_idx]
            
                if verbose: print(f'Tossing: {current_frames_to_toss}...')
                frames_to_toss.append(current_frames_to_toss)
            
            current_frames_to_toss = []
            prior_movement_index = 0
            
                
    flat_frames_to_toss = [el for ls in frames_to_toss for el in ls]
    
    return {frmNum: featDict for frmNum, featDict in sf.items() if frmNum not in flat_frames_to_toss}
          

def separate_out_contiguous_segments(sf):
    
    ### PARAMETERS ###
    ACCEPTABLE_LENGTH_OF_SEGMENT = 25 ### a segment below X frames will be ignored.
    
    outList = [] ### the list of stream features which are contiguous (do not skip frames.)
    
    keys = sorted(sf.keys())
    k_prev, ks = keys[0], keys[1:]
    contiguous_sf = {k_prev: sf[k_prev]}
    
    for k in ks:
        if k == k_prev + 1:
            contiguous_sf[k] = sf[k]
        elif k > k_prev + 1:
            
            ### determine whether the segment should be kept!!!
            if len(contiguous_sf) >= ACCEPTABLE_LENGTH_OF_SEGMENT:
                outList.append(contiguous_sf)
            
            contiguous_sf = {k: sf[k]}
        
        k_prev = k
    
    return outList

#%% Plug holes in homographies

from ATMANpy.funcs.feature_funcs import split_joint_features, join_split_features
import ATMANpy.funcs.high_level_funcs as hlf

def plug_hole_in_homographies(sf, cap):
        
    out_sf = deepcopy(sf)
    
    DISTANCE_THRESHOLD = 15
    inList = split_joint_features(sf)
    end_and_begin = [(max(sf_prev), min(sf_succ)) for sf_prev, sf_succ in zip(inList[:-1], inList[1:])]
    end_and_begin = [(end, beg) for end, beg in end_and_begin if beg - end < DISTANCE_THRESHOLD]
    
    to_redo = []
    for end, beg in end_and_begin:
        frames_missing_between = list(range(end + 1, beg))
        to_redo.extend(frames_missing_between)
    
    for frm in to_redo:
        gFrm = GreyFrame.from_video(cap, frm)
        courtLines, netLines, H, config, outFlag = hlf.track_courtlines(gFrm, None, None, only_225 = False)
        
        if not H is None:
            out_sf[frm] = {'H': H}
            
    return out_sf