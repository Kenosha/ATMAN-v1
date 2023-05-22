# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:00:53 2023

As opposed to sf_postprocess_funcs, these functions work with the types in Features -> StreamFeature and StreamFeatures.
The functions are meant to be applied to SF using .apply -> they take as argument a featDict, plus *args and **kwargs

Function argument names and return names matter, since they are parsed by the .apply method.
Arguments and Values have to be named after the features that they denote.

In principle, these names of features can be arbitrary, but should be consistent!
For example, the courtline homography is named 'H'.

Alternatively, we can also write functions here that take a stream feature and return a stream feature, 
if the feature in question lends itself to vectorization or other speed ups 
that call for treating all features as a whole, across frames 

@author: gianl
"""

import numpy as np

from ATMANpy.types.Features import StreamFeatures
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Region

from ATMANpy.types.Lines import CourtLines
schemaLines = CourtLines.make().add_singles_netline()

def get_court_corner_points(courtLines):
    tl, tr, bl, br = courtLines.get_court_region().to_points()
    return tl, tr, bl, br

def points_to_coords(tl, tr, bl, br):
    tl, tr, bl, br = tl.coords, tr.coords, bl.coords, br.coords
    return tl, tr, bl, br

def median_corner_regions(inSF, how_many_unilateral = 8):
    
    required_features = ['tl', 'tr', 'bl', 'br']
    mySF = inSF[required_features]
    myArr = mySF.apply(points_to_coords).to_array()
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
   
def correct_point(somePt, someReg):
    if somePt.within(someReg):
        
        return somePt
    else:
        return someReg.centroid.round_point()

def calculate_homography(tl, tr, bl, br):
    courtRegion = Region.from_points(tl, tr, bl, br)
    H = schemaLines.get_court_region().get_homography(courtRegion)
    return H


