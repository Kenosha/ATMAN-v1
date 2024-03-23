# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:43:58 2024

Extractor functions that act on DecapStreams (and their MultiFrames)

@author: gianl
"""

import numpy as np
import cv2 as cv

from ATMANpy.funcs.util_funcs import shift_array
from ATMANpy.types.MultiFrames import MaskFrames
from ATMANpy.types.Contours import Contours, Contour


def find_motion_mask(gFrms, 
                     ROI = None,
                     intensity_threshold = 10,
                     relevant_indices = [4, 6, 8],
                     open_kernel = (5, 5),
                     close_kernel = (5, 5)):
    
    if ROI is None:
        motMask = MaskFrames.true(gFrms)
    else:
        motMask = gFrms.within_region(ROI)
        
    motFunc = lambda othFrms: gFrms - othFrms > intensity_threshold 
    
    for idx in relevant_indices:
        othFrms_bef, othFrms_aft = gFrms.shift(-idx), gFrms.shift(idx)
        motMask = motMask & motFunc(othFrms_bef) & motFunc(othFrms_aft)
        
    ### Close and Open, optionally.
    if not close_kernel is None:
        motMask = motMask.close(close_kernel) 
    if not open_kernel is None:
        motMask = motMask.open(open_kernel)
        
    return motMask

def get_burn_in_mask_simple(hFrms, median_backFrm,
                            ROI = None, h_threshold = 5, v_threshold = 5,
                            gaussian_kernel = (21, 21), closing_kernel = (10, 18),
                            verbose = False):
    diffFrms = hFrms - median_backFrm
    
    diffFrms = diffFrms.gaussian_blur(gaussian_kernel)
    
    if not ROI is None:
        diffFrms = diffFrms.mask(diffFrms.within_region(ROI))
    
    backgroundMask = (diffFrms.H() <= h_threshold) & (diffFrms.V() <= v_threshold)
    foregroundMask = ~backgroundMask
    
    foregroundMask = foregroundMask.close(closing_kernel)
    
    return foregroundMask

def get_shadow_mask_simple(hFrms, median_backFrm, 
                           ROI = None,
                           h_threshold = 5, s_threshold = 10, 
                           v_threshold_lower = 0.75,  ### the brighter the sun, the starker the shadow, the lower this value has to be.
                           v_threshold_upper = 0.97,
                           opening_kernel = (15, 15),
                           closing_kernel = (15, 15),
                           verbose = False):
    
    diffFrm = hFrms - median_backFrm
    
    if not ROI is None:
        diffFrm = diffFrm.mask(diffFrm.within_region(ROI))
    
    if not h_threshold is None:
        ### H channel -> affected only slightly!
        HMask = diffFrm.H() < h_threshold
    else:
        HMask = MaskFrames.true(hFrms)

    if not s_threshold is None:
        ### S channel -> affected slightly, but usually lowered!
        S_is_lowered = (hFrms.S() < median_backFrm.S())
        SMask = (diffFrm.S() < s_threshold) & S_is_lowered
    else:
        SMask = MaskFrames.true(hFrms)
        
    ### V channel -> lowered by 10-20% typically, according to cucchiara et al 2001
    ratioArr = hFrms.V().array.astype(float) / np.maximum(median_backFrm.V().array.astype(float), 1)
    VArr = (ratioArr > v_threshold_lower) & (ratioArr < v_threshold_upper)
    VMask = MaskFrames(VArr.astype('uint8') * 255)
    
    shadowMask = HMask & SMask & VMask 
    
    if verbose: hFrms.mask(~shadowMask).show()
    
    ### open the shadow mask, to remove clutter
    shadowMask = shadowMask.open(opening_kernel)
    
    if verbose: hFrms.mask(~shadowMask).show()
    
    ### close the candidate shadows, since shadows of a player are contiguous.
    shadowMask = shadowMask.close(closing_kernel)
    
    if verbose: hFrms.mask(~shadowMask).show()
    
    return shadowMask

def find_contours(myMasks, min_size = 25, retr = cv.RETR_EXTERNAL, approx = cv.CHAIN_APPROX_SIMPLE):
    ### use cv.RETR_EXTERNAL to only get external contours! RETR_TREE also gives black contours that are completely within white areas!
    myCnts, _ = myMasks.find_contours(retr, approx)
    myCnts = [[cnt for cnt in subCnts if cnt.area() > min_size] for subCnts in myCnts]
    return myCnts

def find_player_candidate_contours_timeblur(foregroundMasks, shadowMasks,
                                            ROI = None, min_player_size = 50, frame_blur = 1):
    
    jointMasks = MaskFrames.false(foregroundMasks)
    
    for blur_frame in list(range(-frame_blur, frame_blur + 1)):
        shift_fMasks = foregroundMasks.shift(blur_frame)
        shift_sMasks = shadowMasks.shift(blur_frame)
        shift_Masks = shift_fMasks & ~shift_sMasks  
        
        jointMasks = jointMasks | shift_Masks

    if not ROI is None:
        jointMasks = jointMasks & jointMasks.within_region(ROI)
        
    foreCnts = find_contours(jointMasks, min_size = min_player_size)
    return foreCnts


def get_masked_video(cFrms, foreMasks, motiMasks):
    jointMasks = foreMasks | motiMasks
    maskedFrms = cFrms.mask(jointMasks)
    return maskedFrms


