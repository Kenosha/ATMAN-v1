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

def find_courtlines(H, schemaLines = schemaLines):
    if H is None:
        return None
    else:
        courtLines = schemaLines.transform('apply_homography', H).transform('round_points')
    return courtLines

def calibrate_camera(courtLines, netLns, 
                     width = 'Doubles', schemaLines = schemaLines, 
                     add_coords_world = None, add_coords_image = None):
    from ATMANpy.funcs.court_funcs import calibrate_camera
    
    P = calibrate_camera(courtLines, netLns, 
                         width = width, schemaLines = schemaLines,
                         add_coords_world = add_coords_world, 
                         add_coords_image = add_coords_image)
    return P
    
def find_intrinsic_extrinsic_M(P, normalize_K = True):
    from ATMANpy.funcs.HMM.HPost import find_intrinsic_extrinsic_M    
    
    M_int, M_ext = find_intrinsic_extrinsic_M(P, normalize_K = normalize_K)
    return M_int, M_ext

def get_court_corner_points(courtLines):
    tl, tr, bl, br = courtLines.get_court_region().to_points()
    return tl, tr, bl, br

def get_net_corner_points(netLines):
    tl, bl = netLines['Left Post'].high_point(), netLines['Left Post'].low_point()
    tr, br = netLines['Right Post'].high_point(), netLines['Right Post'].low_point()
    tm = netLines['Left Net'].right_point()
    return bl, tl, tm, tr, br

def points_to_coords(tl, tr, bl, br):
    tl, tr, bl, br = tl.coords, tr.coords, bl.coords, br.coords
    return tl, tr, bl, br

def points_to_coords_net(bl, tl, tm, tr, br):
    bl, tl, tm, tr, br = bl.coords, tl.coords, tm.coords, tr.coords, br.coords
    return bl, tl, tm, tr, br

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
    
    
def find_player_positions_3D(P, topPlayer, botPlayer):
    from ATMANpy.funcs.court_funcs import make_P_cond_inv
    from ATMANpy.types.Shapes3D import Point3D
    
    cond_val = 0 ### feet are on the ground!
    cond_dim = 2 ### axis is z dimension -> height!
    P_cond_inv = make_P_cond_inv(P, cond_val, cond_dim)
   
    if topPlayer is None:   
        topPlayer_3D = None
    else:
        top_x, top_y = topPlayer.enclosing_region()['Bottom'].center().coords
        top_3D_homogeneous = P_cond_inv @ np.array([top_x, top_y, 1])
        top_3D = top_3D_homogeneous[:2] / top_3D_homogeneous[2]
        topPlayer_3D = np.insert(top_3D, cond_dim, cond_val)
        topPlayer_3D = Point3D(tuple(topPlayer_3D))
    
    if botPlayer is None:
        botPlayer_3D = None
    else:
        bot_x, bot_y = botPlayer.enclosing_region()['Bottom'].center().coords
        bot_3D_homogeneous = P_cond_inv @ np.array([bot_x, bot_y, 1])
        bot_3D = bot_3D_homogeneous[:2] / bot_3D_homogeneous[2]
        botPlayer_3D = np.insert(bot_3D, cond_dim, cond_val)
        botPlayer_3D = Point3D(tuple(botPlayer_3D))
    
    return topPlayer_3D, botPlayer_3D

    
def impose_known_ball_coordinates(exp_eventFlag, topPlayer_3D, botPlayer_3D, 
                                  schemaLines = schemaLines, bounce_height = 0.2, how_far_in_front = 8):
    
    if exp_eventFlag is None:
        x, y, z = None, None, None
    elif exp_eventFlag[:2] == 'SN': ### IMPOSE SERVE NEAR ###
        x, y, z = None, schemaLines['Bottom Baseline'].center().y, None
    elif exp_eventFlag[:2] == 'SF': ### IMPOSE SERVE FAR ###
        x, y, z = None, schemaLines['Top Baseline'].center().y, None
    elif exp_eventFlag == 'NET': ### IMPOSE NET ###
        x, y, z = None, schemaLines['Net Line'].center().y, None
    elif exp_eventFlag.startswith('B'): ### IMPOSE BOUNCE ###
        x, y, z = None, None, bounce_height
    elif exp_eventFlag == 'HN': ### IMPOSE HIT NEAR ###
        x, y, z = None, botPlayer_3D.get_foot_point().y - how_far_in_front, None
    elif exp_eventFlag == 'HF': ### IMPOSE HIT FAR ###
        x, y, z = None, topPlayer_3D.get_foot_point().y + how_far_in_front, None
    else:
        x, y, z = None, None, None
    
    ball_3D_tuple = (x, y, z)
    return ball_3D_tuple

def find_known_ball_positions_3D(ballCnt, P, exp_eventFlag, topPlayer_3D, botPlayer_3D,
                                 schemaLines = schemaLines, 
                                 bounce_height = 0.2, how_far_in_front = 8):
    from ATMANpy.funcs.court_funcs import make_P_cond_inv
    from ATMANpy.types.Shapes3D import Point3D
    
    x_3D, y_3D, z_3D = impose_known_ball_coordinates(exp_eventFlag, topPlayer_3D, botPlayer_3D,
                                                     schemaLines = schemaLines, 
                                                     bounce_height = bounce_height, 
                                                     how_far_in_front = how_far_in_front)
    
    if ballCnt is None:
        return None
    else:
        x_2D, y_2D = ballCnt.center().coords
    
    ### Figure out if (and if yes, which) coordinate we have
    if x_3D is not None:
        cond_val = x_3D
        cond_dim = 0
    elif y_3D is not None:
        cond_val = y_3D
        cond_dim = 1
    elif z_3D is not None:
        cond_val = z_3D
        cond_dim = 2
    else:
        return None
    
    P_cond_inv = make_P_cond_inv(P, cond_val, cond_dim)

    pt_3D_homogeneous = P_cond_inv @ np.array([x_2D, y_2D, 1])
    pt_3D = pt_3D_homogeneous[:2] / pt_3D_homogeneous[2]

    ball_3D_tuple = np.insert(pt_3D, cond_dim, cond_val)
    
    ball_3D = Point3D(ball_3D_tuple)
    return ball_3D
    