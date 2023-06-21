# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:00:53 2023

As opposed to sf_postprocess_funcs, these functions work with the types in Features -> StreamFeature and StreamFeatures.
The functions are meant to be applied to SF using .apply -> they take as argument a featDict, plus *args and **kwargs

Function argument names and return names matter, since they are parsed by the .apply method.
Arguments and Values have to be named after the features that they denote.

In principle, these names of features can be arbitrary, but should be consistent!
For example, the courtline homography is named 'H'.
Z
@author: gianl
"""

import numpy as np

from ATMANpy.types.Features import StreamFeatures
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Region

from ATMANpy.types.Lines import CourtLines
schemaLines = CourtLines.make().add_singles_netline()

import ATMANpy.funcs.particle_funcs_vectorized as pf

def get_court_corner_points(courtLines):
    tl, tr, bl, br = courtLines.get_court_region().to_points()
    return tl, tr, bl, br

def points_to_coords(tl, tr, bl, br):
    tl, tr, bl, br = tl.coords, tr.coords, bl.coords, br.coords
    return tl, tr, bl, br

def correct_point(somePt, someReg):
    if somePt.within(someReg):
        return somePt, False
    else:
        return someReg.centroid.round_point(), True

def calculate_homography(tl, tr, bl, br):
    courtRegion = Region.from_points(tl, tr, bl, br)
    H = schemaLines.get_court_region().get_homography(courtRegion)
    return H

def replace_trackflag(trackFlag, replace_value):
    return ''
        
def choose_ball_contour(smooth_particles, candCnts, likelihood_threshold = 10 ** (-15)):
    ballCnt = pf.choose_likely_ball_contour(smooth_particles, candCnts, l_threshold = likelihood_threshold)
    return ballCnt
    

#### Add in boolean features for in/out calling
def distance_from_near_court(courtLines, ballCnt):
    try:
        cPt = ballCnt.center()
    except:
        return None
    else:
        courtReg = courtLines.get_half_region(side = 'Bottom', width = 'Singles')
        
        ### get the distances to the bounding lines, and take the nearest one
        courtDists = [line.distance_to(cPt) for line in courtReg]
        nearest_dist = min(courtDists)
        
        if cPt.within(courtReg):
            return -nearest_dist ### balls within are marked with a negative distance
        else:
            return nearest_dist ### balls without are marked with a positive distance
        
def distance_from_far_court(courtLines, ballCnt):
    try:
        cPt = ballCnt.center()
    except:
        return None
    else:
        courtReg = courtLines.get_half_region(side = 'Top', width = 'Singles')
        
        ### get the distances to the bounding lines, and take the nearest one
        courtDists = [line.distance_to(cPt) for line in courtReg]
        nearest_dist = min(courtDists)
        
        if cPt.within(courtReg):
            return -nearest_dist ### balls within are marked with a negative distance
        else:
            return nearest_dist ### balls without are marked with a positive distance

def certainty_out_near_court(courtLines, ballCnt):
    ### idea: use a sigmoid to express the certainty that a ball is in orr out the resp. region
    # given the distance calculated with the funcs above.
    # the sigmoid is centered at zero, naturally, but the scaling factor K is not set in stone.
    # (is it centered at zero? a ball on the line is pretty much in, at more than 50% certainty...)
    # it has to be reassessed with common sense, using for example a threshold function value of 80%,
    # and checking which distance_from_near_court one would sensibly assign 
    # an 80% chance of being in or out IF the ball is on the ground at this moment.
    x_80 = 6 ### in our case, six pixels. TODO: make this more resilient against changes in aspect ratio!
    dist = distance_from_near_court(courtLines, ballCnt)
    if dist is None:
        return None
    else:
        k = - np.log(0.2) / x_80
        dN = 1 / (1 + np.exp(-k * dist))
        return dN
    
def certainty_out_far_court(courtLines, ballCnt):
    ### idea: use a sigmoid to express the certainty that a ball is in orr out the resp. region
    # given the distance calculated with the funcs above.
    # the sigmoid is centered at zero, naturally, but the scaling factor K is not set in stone.
    # (is it centered at zero? a ball on the line is pretty much in, at more than 50% certainty...)
    # it has to be reassessed with common sense, using for example a threshold function value of 80%,
    # and checking which distance_from_near_court one would sensibly assign 
    # an 80% chance of being in or out IF the ball is on the ground at this moment.
    x_80 = 6 ### in our case, six pixels. TODO: make this more resilient against changes in aspect ratio!
    dist = distance_from_far_court(courtLines, ballCnt)
    if dist is None:
        return None
    else:
        k = - np.log(0.2) / x_80
        dF = 1 / (1 + np.exp(-k * dist))
        return dF
    
