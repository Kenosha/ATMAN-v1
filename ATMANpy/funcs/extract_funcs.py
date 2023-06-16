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
import cv2 as cv

from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

from ATMANpy.funcs.court_funcs import find_homography, find_homography_225_only, update_homography
from ATMANpy.funcs.scoreboard_funcs import find_scoreboard
from ATMANpy.funcs.player_funcs import find_best_expansion, stop_criterion, find_player_contour_with_overlap

from ATMANpy.types.Frames import Frame, BoolFrame, GreyFrame, BGRFrame, HSVFrame
from ATMANpy.types.Contours import Contour

def find_courtlines(H, schemaLines = schemaLines):
    if H is None:
        return None
    else:
        courtLines = schemaLines.transform('apply_homography', H).transform('round_points')
    return courtLines

def track_courtlines(gCache, featCache,
                     width_of_line = 8, 
                     int_diff_thresh = 30,
                     white_thresh = 150,
                     rho = 3, theta = np.pi / 180, min_votes = 500,
                     minLineLength = 8 + 1, maxLineGap = 1,
                     merge_threshold = 2 * 8, 
                     filter_margin = 40,
                     points_on_line = 100, acceptable_failure_rate = 0.5,
                     schemaLines = schemaLines, only_225 = True):
    
    ### extract values from cache
    curr_gFrm = gCache[0]
    prev_H = featCache[-1]['H']
    prev_config = featCache[-1]['config']
    
    if prev_H is None or prev_config is None:
        if only_225:
            H, config = find_homography_225_only(curr_gFrm,
                                                 width_of_line = width_of_line,
                                                 int_diff_thresh = int_diff_thresh,
                                                 white_thresh = white_thresh,
                                                 rho = rho, theta = theta, min_votes = min_votes,
                                                 minLineLength = minLineLength, maxLineGap = maxLineGap,
                                                 merge_threshold = merge_threshold, 
                                                 filter_margin = filter_margin)
        else:
            H, config = find_homography(curr_gFrm,
                                        width_of_line = width_of_line,
                                        int_diff_thresh = int_diff_thresh,
                                        white_thresh = white_thresh,
                                        rho = rho, theta = theta, min_votes = min_votes,
                                        minLineLength = minLineLength, maxLineGap = maxLineGap,
                                        merge_threshold = merge_threshold, 
                                        filter_margin = filter_margin)
        trackFlag = 'Found'
    else:
        # print('Courtlines found, updating homography...')
        H, config = update_homography(curr_gFrm, prev_H, prev_config,
                                      points_on_line = points_on_line, 
                                      acceptable_failure_rate = acceptable_failure_rate)
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
                   min_votes = 500, vote_threshold = 100):
    try:
        H, config = find_homography(gFrm, min_votes = min_votes)
    except Exception as e:
        # print(e)
        if min_votes > vote_threshold:
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
    wrpFrm = hFrm.warp(H_inv, cols = schemaFrm.cols, rows = schemaFrm.rows)
    return wrpFrm


def get_burn_in_mask_iterative(hFrm, H, median_backFrm, std_backFrm, 
                               minimum_proportion = 0.01, minimum_step_gain = 0.015,
                               verbose = False):
    
    """
    medFrm and stdFrm are the standard deviations of the court color, 
    projected onto the court schema during burn in.
    They are basically what is produced and saved by the burn in script
    """
    
    # H_inv = np.linalg.inv(H)
    wrpFrm = median_backFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
    # if verbose: wrpFrm.show()
    
    stdwrpFrm = std_backFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
    # if verbose: stdwrpFrm.show()
    
    ### Background Differencing ###
    normdiffFrm = (hFrm - wrpFrm) / stdwrpFrm.clip_lo(1)
    
    ### exclude everything_outside wrpFrms scope. this way, it will not bleed into the frame if we gaussian blur.
    normdiffFrm = normdiffFrm.mask((wrpFrm > 0).any_channel())
    
    ### Adding a blur stage!
    normdiffFrm = normdiffFrm.gaussian_blur((21, 21))
    
    ### Iterative discovery of best background parameters
    histFrm = normdiffFrm.hist(['H', 'V'], normalize = True)
    
    # return histFrm

    previous_expansion = (0, 0)
    current_expansion = find_best_expansion(histFrm, 
                                            previous_expansion = previous_expansion)
    while not stop_criterion(histFrm, previous_expansion, current_expansion,
                              minimum_proportion = minimum_proportion, 
                              minimum_step_gain = minimum_step_gain,
                              verbose = verbose):
        
        previous_expansion = current_expansion
        current_expansion = find_best_expansion(histFrm, 
                                                previous_expansion = previous_expansion)
    else:
        best_expansion = previous_expansion ### since the current expansion triggered the stop criterion.
    
    best_h, best_v = best_expansion
    
    backgroundMask = (normdiffFrm.H() <= best_h) & (normdiffFrm.V() <= best_v)
    foregroundMask = ~backgroundMask
    
    if verbose:
        pixel_count = (foregroundMask.rows * foregroundMask.cols)
        proportion_foreground = (foregroundMask.sum((0, 1)) / 255) / pixel_count
        print(proportion_foreground)
    
    ### Adding a closing stage
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 18))
    foregroundMask = foregroundMask.close(kernel)
    
    if verbose: hFrm.mask(foregroundMask).show()
    return foregroundMask

def get_shadow_mask(hFrm, H, median_backFrm,
                    h_threshold = 5, s_threshold = 10, 
                    v_threshold_lower = 0.75,  ### the brighter the sun, the starker the shadow, the lower this value has to be.
                    v_threshold_upper = 0.97,
                    verbose = False):
    
    wrpFrm = median_backFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
    if verbose: wrpFrm.show()

    diffFrm = hFrm - wrpFrm

    ### H channel -> affected only slightly!
    H_THRESHOLD = 5
    HBool = diffFrm.H() < H_THRESHOLD

    ### S channel -> affected slightly, but usually lowered!
    S_THRESHOLD = 10
    S_is_lowered = (hFrm.S() < wrpFrm.S())
    SBool = (diffFrm.S() < S_THRESHOLD) & S_is_lowered
    
    ### V channel -> lowered by 10-20% typically, according to cucchiara et al 2001
    ratioArr = hFrm.V().array.astype(float) / np.maximum(wrpFrm.V().array.astype(float), 1)
    ratioFrm = Frame(ratioArr)
    VBool = (ratioFrm > v_threshold_lower) & (ratioFrm < v_threshold_upper)
    
    shadowMask = HBool & SBool & VBool 
    
    if verbose: hFrm.mask(~shadowMask).show()
    
    ### close the candidate shadows, since shadows of a player are contiguous.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    shadowMask = shadowMask.close(kernel)
    
    if verbose: hFrm.mask(~shadowMask).show()
    
    return shadowMask

def adaptive_player_ROI(playerContour, initialROI, H, gFrm,
                        player_tracking_scale_factor = 1.2,
                        min_bounding_height = 100,
                        min_bounding_width = 40,
                        halfcourtLine = None, keep_below_or_above = None):
        
    def get_projected_ROI(ROI, H):
        return ROI.transform('apply_homography', H).transform('round_points')

    if playerContour is None:
        initial_projROI = get_projected_ROI(initialROI, H)
    
    if playerContour is None:
        roiMask = gFrm.within_region(initial_projROI)
    else:
        enclReg = playerContour.enclosing_region()
        updated_projROI = enclReg.scale_equal(player_tracking_scale_factor)
        updated_projROI = updated_projROI.enlarge(to_min_height = min_bounding_height, to_min_width = min_bounding_width)
        roiMask = gFrm.within_region(updated_projROI)
        
    # if not halfcourtROI is None:
    #     ### to prevent adaptive ROIs to spill over into the other player's court
    #     ### we can impose a half court mask
    #     proj_halfcourtROI = get_projected_ROI(halfcourtROI, H)
    #     halfcourtMask = gFrm.within_region(proj_halfcourtROI)
    #     roiMask = roiMask & halfcourtMask
        
    if not halfcourtLine is None:
        ### to prevent adaptive ROIs to spill over into the other player's court
        ### we can impose a half court mask
        if not keep_below_or_above in ['below', 'above']:
            raise Exception('When specifying halfcourt boundary in player selection, specify whether to keep everything above or below.')
        else:
            proj_halfcourtLine = halfcourtLine.apply_homography(H).round_points()
            halfMask = getattr(gFrm, keep_below_or_above)(proj_halfcourtLine)
            roiMask = roiMask & halfMask
        
    return roiMask

def find_both_players_smart(gCache, featCache,
                            foregroundMask, shadowMask,
                            initial_ROI_upper = None, initial_ROI_lower = None,
                            halfcourtLine_upper = None, halfcourtLine_lower = None,
                            player_tracking_scale_factor = 1.2,
                            bot_min_bounding_height = 100, bot_min_bounding_width = 40,
                            top_min_bounding_height = 100, top_min_bounding_width = 40,
                            verbose = False):
    
    gFrm = gCache[0]
    H = featCache[0]['H']
    topPlayer = featCache[-1]['topPlayer']
    botPlayer = featCache[-1]['botPlayer']
    
    topROIMask = adaptive_player_ROI(topPlayer, initial_ROI_upper, H, gFrm,
                                     player_tracking_scale_factor = player_tracking_scale_factor,
                                     min_bounding_height = top_min_bounding_height,
                                     min_bounding_width = top_min_bounding_width,
                                     halfcourtLine = halfcourtLine_upper, keep_below_or_above = 'above')
    
    botROIMask = adaptive_player_ROI(botPlayer, initial_ROI_lower, H, gFrm,
                                 player_tracking_scale_factor = player_tracking_scale_factor,
                                 min_bounding_height = bot_min_bounding_height,
                                 min_bounding_width = bot_min_bounding_width,
                                 halfcourtLine = halfcourtLine_lower, keep_below_or_above = 'below')
    
    topPlayer = find_player_contour_with_overlap(gFrm, foregroundMask, topROIMask, topPlayer, 
                                                 shadowMask = shadowMask)
    
    botPlayer = find_player_contour_with_overlap(gFrm, foregroundMask, botROIMask, botPlayer,
                                                 shadowMask = shadowMask)

    ### ROI Masks are returned only for checking...
    return topPlayer, botPlayer

def find_motion_mask(gCache, 
                     intensity_threshold = 10,
                     relevant_indices = [4, 6, 8]):
    centralFrm = gCache[0]
    
    motFunc = lambda othFrm: centralFrm - othFrm > intensity_threshold 
    
    motMask = BoolFrame.true(centralFrm)
    for idx in relevant_indices:
        othFrm_bef, othFrm_aft = gCache[-idx], gCache[idx]
        motMask = motMask & motFunc(othFrm_bef) & motFunc(othFrm_aft)
   
    return motMask

def refine_motion_mask(motMask, topPlayer, botPlayer,
                       dil_kernel_dims = (20, 10),
                       open_kernel_dims = (5, 5)):
    
    ### Dilated player mask
    dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, dil_kernel_dims)
    
    if not topPlayer is None:
        topMask = motMask.within_contour(topPlayer).dilate(dil_kernel)
    else:
        topMask = BoolFrame.false(motMask)
        
    if not botPlayer is None:
        botMask = motMask.within_contour(botPlayer).dilate(dil_kernel)
    else:
        botMask = BoolFrame.false(motMask)
    
    playerMask = topMask | botMask
    
    ### Opened Motion Mask
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, open_kernel_dims)
    motMask = motMask.open(open_kernel)
    
    ### Player and Motion Mask
    intMask = ~playerMask & motMask
    
    return intMask

def find_ball_candidates(intMask, hFrm, gFrm, svmDict = None, max_number_of_contours = 40):
    from ATMANpy.funcs.ball_funcs import make_feature_frame
    
    ### find movement contours
    movCnts, _ = intMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    movCnts = [Contour(c) for c in movCnts if len(c) > 2] ### need at least three points for a contour
    
    ### if there is an excessive amount of contours, we likely have camera panning...
    if len(movCnts) > max_number_of_contours:
        return None
    
    ### if there are no contours, there is nothing we can do...
    if len(movCnts) == 0:
        return None
    
    ### make a feature frame
    cntFrm = make_feature_frame(movCnts, hFrm, gFrm, intMask).dropna()
    
    ### it can be that features can not be calculated for a contour...
    if len(cntFrm) == 0:
        return None
    
    ### Using the SVM estimated, predict whether a contour is a ball.
    clf, features, mean, std = svmDict['svm'], svmDict['features'], svmDict['mean'], svmDict['std']
    cntFrm = (cntFrm - mean) / std
    cntFrm['isBall'] = clf.predict(cntFrm[features].values)
    
    ### get all True predictions
    ballIdx = list(cntFrm[cntFrm.isBall == 1].index)
    ball_candidates = [c for i, c in enumerate(movCnts) if i in ballIdx]

    return ball_candidates
    

