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

from ATMANpy.funcs.sf_apply_funcs import find_courtlines
from ATMANpy.funcs.court_funcs import find_homography, find_homography_225_only, update_homography, find_linesets
from ATMANpy.funcs.scoreboard_funcs import find_scoreboard
from ATMANpy.funcs.player_funcs import find_best_expansion, stop_criterion, find_player_contour_with_overlap

from ATMANpy.types.Frames import Frame, BoolFrame, MaskFrame, GreyFrame, BGRFrame, HSVFrame
from ATMANpy.types.Contours import Contour, Contours

def get_courtlines(gFrm, width_of_line = 8, 
                   int_diff_thresh = 30,
                   white_thresh = 150,
                   rho = 3, theta = np.pi / 180, min_votes = 500,
                   minLineLength = 8 + 1, maxLineGap = 1,
                   merge_threshold = 2 * 8, 
                   horizontal_filter_margin = 40,
                   vertical_filter_margin = None,
                   points_on_line = 100, acceptable_failure_rate = 0.5,
                   schemaLines = schemaLines, only_225 = True, verbose = False):
    
    if only_225:
        H, config = find_homography_225_only(gFrm,
                                             width_of_line = width_of_line,
                                             int_diff_thresh = int_diff_thresh,
                                             white_thresh = white_thresh,
                                             rho = rho, theta = theta, min_votes = min_votes,
                                             minLineLength = minLineLength, maxLineGap = maxLineGap,
                                             merge_threshold = merge_threshold, 
                                             horizontal_filter_margin = horizontal_filter_margin,
                                             vertical_filter_margin = vertical_filter_margin,
                                             verbose = verbose)
    else:
        H, config = find_homography(gFrm,
                                    width_of_line = width_of_line,
                                    int_diff_thresh = int_diff_thresh,
                                    white_thresh = white_thresh,
                                    rho = rho, theta = theta, min_votes = min_votes,
                                    minLineLength = minLineLength, maxLineGap = maxLineGap,
                                    merge_threshold = merge_threshold, 
                                    horizontal_filter_margin = horizontal_filter_margin,
                                    vertical_filter_margin = vertical_filter_margin,
                                    verbose = verbose)
    trackFlag = 'Found'
    
    if not H is None:
        courtLines = find_courtlines(H, schemaLines)
    else:
        trackFlag = 'Not Found'
        courtLines = None
    
    return courtLines, H, config, trackFlag

def track_courtlines(gCache, featCache,
                     width_of_line = 8, 
                     int_diff_thresh = 30,
                     white_thresh = 150,
                     rho = 3, theta = np.pi / 180, min_votes = 500,
                     minLineLength = 8 + 1, maxLineGap = 1,
                     merge_threshold = 2 * 8, 
                     horizontal_filter_margin = 40,
                     vertical_filter_margin = None,
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
                                                 horizontal_filter_margin = horizontal_filter_margin,
                                                 vertical_filter_margin = vertical_filter_margin)
        else:
            H, config = find_homography(curr_gFrm,
                                        width_of_line = width_of_line,
                                        int_diff_thresh = int_diff_thresh,
                                        white_thresh = white_thresh,
                                        rho = rho, theta = theta, min_votes = min_votes,
                                        minLineLength = minLineLength, maxLineGap = maxLineGap,
                                        merge_threshold = merge_threshold, 
                                        horizontal_filter_margin = horizontal_filter_margin,
                                        vertical_filter_margin = vertical_filter_margin)
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

def get_burn_in_mask_simple(hFrm, median_backFrm,
                            ROI = None, h_threshold = 5, v_threshold = 5,
                            gaussian_kernel = (21, 21), closing_kernel = (10, 18),
                            verbose = False):
    diffFrm = hFrm - median_backFrm
    
    diffFrm = diffFrm.gaussian_blur(gaussian_kernel)
    
    if not ROI is None:
        diffFrm = diffFrm.mask(diffFrm.within_region(ROI))
    
    backgroundMask = (diffFrm.H() <= h_threshold) & (diffFrm.V() <= v_threshold)
    foregroundMask = ~backgroundMask
    
    foregroundMask = foregroundMask.close(closing_kernel)
    return foregroundMask
 
def get_burn_in_mask_iterative(hFrm, H, median_backFrm, std_backFrm, 
                               ROI = None,
                               minimum_proportion = 0.01, minimum_step_gain = 0.015,
                               gaussian_kernel = (21, 21), closing_kernel = (10, 18),
                               back_frames_are_warped = True,
                               verbose = False):
    
    """
    medFrm and stdFrm are the standard deviations of the court color, 
    projected onto the court schema during burn in.
    They are basically what is produced and saved by the burn in script
    """
    
    if back_frames_are_warped:
        # H_inv = np.linalg.inv(H)
        wrpFrm = median_backFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
        # if verbose: wrpFrm.show()
        
        stdwrpFrm = std_backFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
        # if verbose: stdwrpFrm.show()
        
        normdiffFrm = (hFrm - wrpFrm) / stdwrpFrm.clip_lo(1)
        
        ### exclude everything_outside wrpFrms scope. this way, it will not bleed into the frame if we gaussian blur.
        normdiffFrm = normdiffFrm.mask((wrpFrm > 0).any_channel())
    else:
        normdiffFrm = (hFrm - median_backFrm) / std_backFrm.clip_lo(1)
        
    ### Adding a blur stage!
    normdiffFrm = normdiffFrm.gaussian_blur(gaussian_kernel)
    
    if ROI is not None:
        normdiffFrm = normdiffFrm.mask(normdiffFrm.within_region(ROI))
    
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
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, closing_kernel)
    foregroundMask = foregroundMask.close(kernel)
    
    if verbose: hFrm.mask(foregroundMask).show()
    return foregroundMask

def get_shadow_mask(hFrm, H, median_backFrm,
                    ROI = None,
                    h_threshold = 5, s_threshold = 10, 
                    v_threshold_lower = 0.75,  ### the brighter the sun, the starker the shadow, the lower this value has to be.
                    v_threshold_upper = 0.97,
                    opening_kernel = (15, 15),
                    closing_kernel = (15, 15),
                    back_frames_are_warped = True,
                    verbose = False):
    
    if back_frames_are_warped:
        like_backFrm = median_backFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
        if verbose: like_backFrm.show()
    else:
        like_backFrm = median_backFrm
    
    diffFrm = hFrm - like_backFrm
    
    if not ROI is None:
        diffFrm = diffFrm.mask(diffFrm.within_region(ROI))
    
    if not h_threshold is None:
        ### H channel -> affected only slightly!
        HMask = diffFrm.H() < h_threshold
    else:
        HMask = MaskFrame.true(hFrm)

    if not s_threshold is None:
        ### S channel -> affected slightly, but usually lowered!
        S_is_lowered = (hFrm.S() < like_backFrm.S())
        SMask = (diffFrm.S() < s_threshold) & S_is_lowered
    else:
        SMask = MaskFrame.true(hFrm)
        
    ### V channel -> lowered by 10-20% typically, according to cucchiara et al 2001
    ratioArr = hFrm.V().array.astype(float) / np.maximum(like_backFrm.V().array.astype(float), 1)
    ratioFrm = Frame(ratioArr)
    VMask = (ratioFrm > v_threshold_lower) & (ratioFrm < v_threshold_upper)
    
    shadowMask = HMask & SMask & VMask 
    
    if verbose: hFrm.mask(~shadowMask).show()
    
    ### open the shadow mask, to remove clutter
    shadowMask = shadowMask.open(opening_kernel)
    
    if verbose: hFrm.mask(~shadowMask).show()
    
    ### close the candidate shadows, since shadows of a player are contiguous.
    shadowMask = shadowMask.close(closing_kernel)
    
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
                     ROI = None,
                     intensity_threshold = 10,
                     relevant_indices = [4, 6, 8],
                     open_kernel = (5, 5),
                     close_kernel = (5, 5)):
    centralFrm = gCache[0]
    
    motFunc = lambda othFrm: centralFrm - othFrm > intensity_threshold 
    
    if ROI is None:
        motMask = MaskFrame.true(centralFrm)
    else:
        motMask = centralFrm.within_region(ROI)
    
    for idx in relevant_indices:
        othFrm_bef, othFrm_aft = gCache[-idx], gCache[idx]
        motMask = motMask & motFunc(othFrm_bef) & motFunc(othFrm_aft)
   
    ### Close and Open, optionally.
    if not close_kernel is None:
        motMask = motMask.close(close_kernel) 
    if not open_kernel is None:
        motMask = motMask.open(open_kernel)
        
    return motMask


def refine_motion_mask(motMask, topPlayer, botPlayer,
                       dil_kernel_dims = (20, 10)):
    
    ### Dilated player mask
    dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, dil_kernel_dims)
    
    if not topPlayer is None:
        topMask = motMask.within_contour(topPlayer).dilate(dil_kernel)
    else:
        topMask = MaskFrame.false(motMask)
        
    if not botPlayer is None:
        botMask = motMask.within_contour(botPlayer).dilate(dil_kernel)
    else:
        botMask = MaskFrame.false(motMask)
    
    playerMask = topMask | botMask
    
    ### Player and Motion Mask
    intMask = ~playerMask & motMask
    
    return intMask

def get_contours_from_mask(intMask, max_number_of_contours = 40):
    ### find movement contours
    cntArr, _ = intMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    movCnts = Contours([Contour(c) for c in cntArr if len(c) > 2]) ### need at least three points for a contour
    
    if 0 < movCnts.length() < max_number_of_contours:
        ### if there are no contours, there is nothing we can do...
        ### if there is an excessive amount of contours, we likely have camera panning...
        return movCnts
    
def get_movement_contours(gCache, featCache, 
                          intensity_threshold = 10, relevant_indices = [4, 6, 8],
                          dil_kernel_dims = (20, 10), open_kernel_dims = (5, 5),
                          max_number_of_contours = 40):
    
    topPlayer = featCache[0]['topPlayer']
    botPlayer = featCache[0]['botPlayer']
    
    motMask = find_motion_mask(gCache, 
                               intensity_threshold = intensity_threshold, 
                               relevant_indices = relevant_indices)
    
    intMask = refine_motion_mask(motMask, topPlayer, botPlayer, 
                                 dil_kernel_dims = dil_kernel_dims,
                                 open_kernel_dims = open_kernel_dims)
    
    movCnts = get_contours_from_mask(intMask, 
                                     max_number_of_contours = max_number_of_contours)
    
    return movCnts

def get_feature_frames(hFrm, gFrm, intMask, movCnts):
    from ATMANpy.funcs.ball_funcs import make_feature_frame
    
    ### make a feature frame
    ballTable = make_feature_frame(movCnts, hFrm, gFrm, intMask).dropna()
    ballTable.index.name = 'idx1'
    
    ### it can be that features can not be calculated for a contour...
    if len(ballTable) == 0:
        return None
    else:
        return ballTable
    
def get_movement_contours_and_frame(gCache, hCache, featCache, 
                                    intensity_threshold = 10, relevant_indices = [4, 6, 8],
                                    dil_kernel_dims = (20, 10), open_kernel_dims = (5, 5),
                                    max_number_of_contours = 40):
    
    topPlayer = featCache[0]['topPlayer']
    botPlayer = featCache[0]['botPlayer']
    gFrm = gCache[0]
    hFrm = hCache[0]
    
    motMask = find_motion_mask(gCache, 
                               intensity_threshold = intensity_threshold, 
                               relevant_indices = relevant_indices)
    
    intMask = refine_motion_mask(motMask, topPlayer, botPlayer, 
                                 dil_kernel_dims = dil_kernel_dims,
                                 open_kernel_dims = open_kernel_dims)
    
    movCnts = get_contours_from_mask(intMask, 
                                     max_number_of_contours = max_number_of_contours)
    
    if movCnts is None:
        ballTable = None
    else:
        ballTable = get_feature_frames(hFrm, gFrm, intMask, movCnts)
    
    return movCnts, ballTable
    
def get_chamfer_mask(gFrm, topPlayer, botPlayer, threshold = 50):
    import ATMANpy.funcs.particle_funcs_vectorized as pf
    
    ########## DECIDE PARTICLE PROPAGATION #########
    if topPlayer is None or botPlayer is None:
        chamferMask = MaskFrame.false(gFrm)
    else:
        chamferMask = pf.chamfer_distance(gFrm, topPlayer, botPlayer,
                                          threshold = threshold) 
        
    return chamferMask

def get_chamfer_contours(gFrm, topPlayer, botPlayer, threshold = 50):
    chamferMask = get_chamfer_mask(gFrm, topPlayer, botPlayer, threshold = threshold)
    chamferCnts, _ = chamferMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return chamferCnts

def find_particles(chamferCache, featCache, 
                   pixel_in_meter = (200, 100), N = 1000, fps = 25,
                   chamfer_threshold = None,
                   normal_propagation = None, close_propagation = None, switch_probability = None,):

    import ATMANpy.funcs.particle_funcs_vectorized as pf
    
    ### Frame ###
    chamferMask = chamferCache[-1]
    
    ### Exogenous Features ###
    # topPlayer = featCache[-1]['topPlayer']
    # botPlayer = featCache[-1]['botPlayer']
    ballContours = featCache[-1]['ballContours']
    
    ### Endogenous Features ###
    particles = featCache[-1]['particles']
    indexHistories = featCache[-1]['indexHistories']
    closeHistories = featCache[-1]['closeHistories']
    modelHistories = featCache[-1]['modelHistories']
    
    
    if particles is None:
        particles = pf.initialize_particles(chamferMask, pixel_in_meter = pixel_in_meter, N = N, fps = fps)
    
    propagation_models, closeBools, modelBools = pf.get_motion_model(particles, chamferMask,
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

    return particles, indexHistories, closeHistories, modelHistories


def get_joint_movement_mask(gCache, featCache,
                            intensity_threshold = 20,
                            relevant_indices = [4, 6, 8],
                            line_thickness = 10,
                            top_player_kernel = (20, 10), bot_player_kernel = (20, 10),
                            ball_kernel = (7, 7), score_board_kernel = (10, 10),
                            open_kernel = (5, 5), close_kernel = (60, 60),
                            verbose = False): 
    ### detect movement contours in sf_postprocess_funcs.py
    
    courtLines = featCache[0]['courtLines']
    topPlayer = featCache[0]['topPlayer']
    botPlayer = featCache[0]['botPlayer']
    ballCnt = featCache[0]['ballCnt']
    scoreBoard = featCache[0]['scoreBoard']
    
    motionMask = find_motion_mask(gCache, 
                                  intensity_threshold = intensity_threshold, 
                                  relevant_indices = relevant_indices)
    
    zeroMask = motionMask.zero()
    
    courtlineMask = zeroMask.put(courtLines, thickness = line_thickness) > 0
            
    if not topPlayer is None:
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, top_player_kernel)
        topMask = zeroMask.within_contour(topPlayer).dilate(dil_kernel)
    else:
        topMask = zeroMask
    
    if not botPlayer is None:
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, bot_player_kernel)
        botMask = zeroMask.within_contour(botPlayer).dilate(dil_kernel)
    else:
        botMask = zeroMask
    
    if not ballCnt is None:
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ball_kernel)
        ballMask = zeroMask.within_contour(ballCnt).dilate(dil_kernel)
    else:
        ballMask = zeroMask
        
    if not scoreBoard is None:
        dil_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, score_board_kernel)
        scoreMask = zeroMask.within_region(scoreBoard).dilate(dil_kernel)
    else:
        scoreMask = zeroMask
    
    jointMask = motionMask & ~courtlineMask & ~topMask & ~botMask & ~ballMask & ~scoreMask
        
    # ### Adding a blur stage!
    # motionMask2 = motionMask2.gaussian_blur((7, 7))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, open_kernel)
    jointMask = jointMask.open(kernel)

    ### Adding a closing stage
    ### large kernel, so that we join blobs belonging to the same ball kid.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, close_kernel)
    jointMask = jointMask.close(kernel)

    return jointMask

def get_large_contours(jointMask, foregroundMask, area_threshold = 600):
    ### foreground_mask -> burn in mask!
    
    finalMask = jointMask & foregroundMask
        
    myCnts, _ = finalMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # largeCnts = Contours([Contour(ct) for ct in cts if len(ct) > 2]) ### need at least three points for a contour
    largeCnts = Contours([myCnt for myCnt in myCnts if myCnt.area() > area_threshold])
    
    return largeCnts

def find_netlines(gFrm, courtLines, net_factor = 0.9, net_points_to_search = 100):

    from ATMANpy.funcs.net_funcs import find_net_points, remove_outlier_net_points
    relPoints = find_net_points(courtLines, gFrm, 
                                factor = net_factor, points_to_search = net_points_to_search)
    relPoints = remove_outlier_net_points(relPoints)
    
    from ATMANpy.funcs.net_funcs import cubic_regress, make_net_cord_lines
    b_0, b_1 = cubic_regress(relPoints)
    netLns = make_net_cord_lines(courtLines, b_0, b_1)

    return netLns

def frame_comparison(cFrm, median_backFrm, std_backFrm, std_multiple = 3):
    normdiffFrm = (cFrm - median_backFrm) / std_backFrm.clip_lo(1)
    match_pixels = (normdiffFrm < std_multiple).sum((0, 1, 2)) / 255
    all_pixels = normdiffFrm.rows * normdiffFrm.cols * normdiffFrm.channels
    match_ratio = match_pixels / all_pixels
    return match_ratio
    
    
