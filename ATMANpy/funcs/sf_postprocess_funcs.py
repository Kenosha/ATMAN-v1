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
    





