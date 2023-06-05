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
from ATMANpy.funcs.player_funcs import find_best_expansion, stop_criterion

from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, HSVFrame

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
    wrpFrm = hFrm.warp(H_inv, cols = schemaFrm.cols, rows = schemaFrm.rows)
    return wrpFrm


def get_burn_in_mask_iterative(hFrm, H,
                               medFrm = None, stdFrm = None, 
                               minimum_proportion = 0.01, minimum_step_gain = 0.015,
                               verbose = False):
    
    """
    medFrm and stdFrm are the standard deviations of the court color, 
    projected onto the court schema during burn in.
    They are basically what is produced and saved by the burn in script
    """
    
    # H_inv = np.linalg.inv(H)
    wrpFrm = medFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
    # if verbose: wrpFrm.show()
    
    stdwrpFrm = stdFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
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

def get_shadow_mask(hFrm, H,
                    medFrm = None,
                    h_threshold = 5, s_threshold = 10, 
                    v_threshold_lower = 0.75,  ### the brighter the sun, the starker the shadow, the lower this value has to be.
                    v_threshold_upper = 0.97,
                    verbose = False):
    
    wrpFrm = medFrm.warp(H, cols = hFrm.cols, rows = hFrm.rows)
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


# def find_both_players_smart(hFrm, courtLines, H, topPlayer, botPlayer,
#                             medFrm = None, stdFrm = None, 
#                             roiMask_top = None, roiMask_bot = None,
#                             verbose = False):
    
#     ### get the burned in mask for this specific homography
#     burn_foreMask = match_specific_iterative_burn_in_func(hFrm, H, verbose = verbose)
    
#     ### remove shadows
#     burn_shadowMask = match_specific_shadow_func(hFrm, H, verbose = verbose)
    
    
#     topPlayer = find_player_contour_with_overlap(frame, burn_foreMask, roiMask_top, topPlayer,
#                                                   shadowMask = burn_shadowMask)
    
#     botPlayer = find_player_contour_with_overlap(frame, burn_foreMask, roiMask_bot, botPlayer,
#                                                   shadowMask = burn_shadowMask)

#     return topPlayer, botPlayer

