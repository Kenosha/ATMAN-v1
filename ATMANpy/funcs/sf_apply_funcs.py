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

import ATMANpy.auxil.constants as const
from ATMANpy.types.Features import StreamFeatures
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Region
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Line3D

from ATMANpy.types.Lines import Line, CourtLines
schemaLines = CourtLines.make().add_singles_netline()

from ATMANpy.models.ParticleFilter import ParticleFilter

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

def maybe_center(someCnt):
    if someCnt is None:
        somePt = None
    else:
        somePt = someCnt.center()
    return somePt

def maybe_contour(somePt):
    if somePt is None:
        someCnt = None
    else:
        someCnt = Contour.from_point(somePt)
    return someCnt

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
        
# def choose_ball_contour(smooth_particles, candCnts, likelihood_threshold = 10 ** (-15)):
#     ballCnt = pf.choose_likely_ball_contour(smooth_particles, candCnts, l_threshold = likelihood_threshold)
#     return ballCnt
    
def choose_ball_contour(smooth_particles, candCnts, 
                        pFilter = None, standard_deviation_threshold = 2):
    if pFilter is None:
        ballCnt = ParticleFilter().choose_likely_ball_contour(smooth_particles, candCnts, 
                                                              standard_deviation_threshold = standard_deviation_threshold)
    else:
        ballCnt = pFilter.choose_likely_ball_contour(smooth_particles, candCnts, 
                                                     standard_deviation_threshold = standard_deviation_threshold)
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
    
    
def find_player_positions_3D(P, topPlayerCnt, botPlayerCnt):
    from ATMANpy.funcs.court_funcs import make_P_cond_inv
    from ATMANpy.types.Shapes3D import Point3D
    
    cond_val = 0 ### feet are on the ground!
    cond_dim = 2 ### axis is z dimension -> height!
    P_cond_inv = make_P_cond_inv(P, cond_val, cond_dim)
   
    if topPlayerCnt is None:   
        topPlayer_3D = None
    else:
        top_x, top_y = topPlayerCnt.enclosing_region()['Bottom'].center().coords
        top_3D_homogeneous = P_cond_inv @ np.array([top_x, top_y, 1])
        top_3D = top_3D_homogeneous[:2] / top_3D_homogeneous[2]
        topPlayer_3D = np.insert(top_3D, cond_dim, cond_val)
        topPlayer_3D = Point3D(tuple(topPlayer_3D))
    
    if botPlayerCnt is None:
        botPlayer_3D = None
    else:
        bot_x, bot_y = botPlayerCnt.enclosing_region()['Bottom'].center().coords
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

def exceeds_threshold(match_ratio, threshold = 0.5):
    return match_ratio > threshold

def below_threshold(match_ratio, threshold = 0.5):
    return match_ratio < threshold

    
def threshold_false_positives_negatives(fp_rate, fn_rate, fp_threshold = 0.05, fn_threshold = 0.3):
    if fp_threshold is not None:
        fp_bool = fp_rate < fp_threshold
    else:
        fp_bool = True
    
    if fn_threshold is not None:
        fn_bool = fn_rate < fn_threshold
    else:
        fn_bool = True
        
    isCourt = fp_bool & fn_bool
    
    return isCourt

def get_locations_at_serve(topPlayer_srvReg, botPlayer_srvReg, courtLines = None):
    midPt = courtLines['Net Line'].center()
    if botPlayer_srvReg is None:
        servePt = topPlayer_srvReg['Bottom'].center()
        serveLn = Line(servePt, midPt)
        returner_baseLn = courtLines['Bottom Baseline']
        returnPt = serveLn.weak_intersect(returner_baseLn)
        topPt, botPt = servePt, returnPt
    elif topPlayer_srvReg is None:
        servePt = botPlayer_srvReg['Bottom'].center()
        serveLn = Line(servePt, midPt)
        returner_baseLn = courtLines['Top Baseline']
        returnPt = serveLn.weak_intersect(returner_baseLn)
        topPt, botPt = returnPt, servePt
    return topPt, botPt

def get_inferred_hit_points(hitFlag, topPlayerCnt, botPlayerCnt,
                            stroke_px_r_top = 10, stroke_px_r_bot = 10,  ### move to the right, in pixels
                            stroke_px_u_top = -2, stroke_px_u_bot = 10,
                            serve_px_u_top = 20, serve_px_u_bot = 20,    ### move up, in pixels
                            do_top = True, do_bot = True,
                            hand_top = 'right', hand_bot = 'right',
                            resolution = 6):
    
    try:
        side, hit = hitFlag.split('_') ### if we have a serve
        is_serve = True
    except:
        side, stroke, kind = hitFlag.split('_') ### if we have a groundstroke
        is_serve = False
    
    if side == 'top':
        if not do_top:
            return None
        
        myCnt = topPlayerCnt
        
        serve_px_u = serve_px_u_top
        stroke_px_u = stroke_px_u_top
        stroke_px_r = stroke_px_r_top
        
        side_sign = -1
        hand_sign = 1 if hand_top == 'right' else -1
        
    elif side == 'bot':
        if not do_bot:
            return None
        
        myCnt = botPlayerCnt
        
        serve_px_u = serve_px_u_bot
        stroke_px_u = stroke_px_u_bot
        stroke_px_r = stroke_px_r_bot
        
        side_sign = 1
        hand_sign = 1 if hand_top == 'right' else -1
    
    cPt = myCnt.center()
        
    if is_serve:
        hPt = cPt.up(serve_px_u)
    else:
        stroke_sign = 1 if stroke == 'fh' else -1
        
        stroke_px_r_corrected = side_sign * hand_sign * stroke_sign * stroke_px_r
        hPt = cPt.right(stroke_px_r_corrected).up(stroke_px_u)
    
    hitCnt = Contour.from_point(hPt, radius = 1, resolution = resolution)
    
    return hitCnt

def get_3D_hit_point_cloud(P, topPlayerCnt, botPlayerCnt, hitFlag, ball_choicePt,
                           reaches_in_feet = [2.5, 3, 3.5, 4, 4.5, 5],
                           height_in_feet = 5, quantile_threshold = 0.05,
                           resolution_general = 1, resolution_observed = 5):
    ### as opposed to the above function get_inferred_hit_points, 
        # we calculate a cloud of potential hit points, not a single hit point
        # the cloud is calculated in 3D.
    try:
        side, hit = hitFlag.split('_') ### if we have a serve
        is_serve = True
    except:
        side, stroke, kind = hitFlag.split('_') ### if we have a groundstroke
        is_serve = False
        
    topPt3D, botPt3D = find_player_positions_3D(P, topPlayerCnt, botPlayerCnt)
    
    basePt3D = topPt3D if side == 'top' else botPt3D
    height_in_axis_units = height_in_feet * const.SCALE
    hingePt3D = basePt3D.up(height_in_axis_units)
    
    ### TODO: Make depend on HANDEDNESS!
    #### decide angular limits
    if side == 'top':
        if is_serve:
            ### need to go mostly up, and slightly in front
            phi_rad_min, phi_rad_max = 0, 0.3 #0, 0.15
            theta_rad_min, theta_rad_max = 0, 2#0, 1
        else:
            if stroke == 'fh':
                ### need to go to the forehand side, and in front
                phi_rad_min, phi_rad_max = 0, 1
                theta_rad_min, theta_rad_max = 0.5, 1
            elif stroke == 'bh':
                ### need to go to the backhand side, and in front
                phi_rad_min, phi_rad_max = 0, 1
                theta_rad_min, theta_rad_max = 0, 0.5
    elif side == 'bot':
        if is_serve:
            ### need to go mostly up, and slightly in front
            phi_rad_min, phi_rad_max = 0, 0.3 
            theta_rad_min, theta_rad_max = 0, 2 #1, 2
        else:
            if stroke == 'fh':
                ### need to go to the forehand side, and in front
                phi_rad_min, phi_rad_max = 0, 1
                theta_rad_min, theta_rad_max = 1.5, 2
            elif stroke == 'bh':
                ### need to go to the backhand side, and in front
                phi_rad_min, phi_rad_max = 0, 1
                theta_rad_min, theta_rad_max = 1, 1.5
        
    limit_phi = phi_rad_min * np.pi, phi_rad_max * np.pi
    limit_theta = theta_rad_min * np.pi, theta_rad_max * np.pi

    resolution = resolution_general if ball_choicePt is None else resolution_observed    

    pointList = []
    for reach_ft in reaches_in_feet:
        reach_in_axis_units = reach_ft * const.SCALE
        reachable_ball = Ball3D(hingePt3D.coords, radius = reach_in_axis_units, limit_phi = limit_phi, limit_theta = limit_theta)
        reachable_points = reachable_ball.rasterize(resolution = resolution)
        pointList.append(reachable_points)
        
    pointArr = np.concatenate(pointList)
    x, y, z = pointArr.T
    cldPt3D = Point3D((x, y, z)) ### this is vectorized! Point3D can hold many points.
    
    ### filter out points that do not project close to the ball point (if it is observed)
    if ball_choicePt is not None:
        pointArr = filter_hit_candidates_with_ball(cldPt3D, ball_choicePt, P, quantile_threshold = quantile_threshold)
        x, y, z = pointArr.T
        cldPt3D = Point3D((x, y, z)) ### this is vectorized! Point3D can hold many points.
        
    return cldPt3D  
    
def project_hit_point_cloud(cldPt3D, P, unique = True):
    cldPts2D_float = cldPt3D.project(P)
    cldPts2D_int = cldPts2D_float.astype(int)
    if not unique:
        cldPts2D = cldPts2D_int 
    else:
        cldPts2D = np.unique(cldPts2D_int, axis = 0)
    return cldPts2D
        
### Define Criteria that a serving player contour has to fulfill.
# Criterion 1) center of shape is inside of the serve ROI
def keep_contours_in_roi(myCnts, ROI = None):
    inCnts = Contours([cnt for cnt in myCnts if cnt.center().within(ROI)])
    return inCnts
    
# Auxiliary function
def get_distance_to_baseline(myCnt, which = 'Top', courtLines = None):
    assert which in ['Top', 'Bottom']
    
    baseLine = courtLines[f'{which} Baseline']
    botPt = myCnt.enclosing_region()['Bottom'].center()
    linePt = botPt.project_vertically(baseLine)
    
    ### make it so that negative distances are not admissible -> they are in front of the line!
    ### then, the task is simply to choose the smallest positive distance.
    if which == 'Top':
        dist = linePt.y - botPt.y
    elif which == 'Bottom':
        dist = botPt.y - linePt.y
    
    return dist
    
### Criterion 2) bottom of shape is behind the respective baseline, keep the closest to baseline among these.
def keep_closest_contour_behind_baseline(myCnts, which = 'Top', courtLines = None, margin = 5):
    assert which in ['Top', 'Bottom']
   
    bestCnt, best_dist = None, None
    for myCnt in myCnts:
        mrg_dist = get_distance_to_baseline(myCnt, which = which, courtLines = courtLines) + margin # to not exclude players at baseline, add a little margin
        if mrg_dist > 0: ### accept only contours whose bottom is behind the respective line
            if best_dist is None:
                bestCnt = myCnt
                best_dist = mrg_dist
            else:
                if mrg_dist < best_dist:
                    bestCnt = myCnt
                    best_dist = mrg_dist
    
    return bestCnt

def make_camline(ball_choicePt, myCamera = None, length_factor = 1000):
    if ball_choicePt is None:
        camLine = None
    else:
        m_w, b_w = myCamera.get_world_ray_params(ball_choicePt)
        camPt, camVec = Point3D(b_w), Point3D(m_w)
        camLine = Line3D(camPt, camPt - camVec * length_factor, color = const.BGR['strong_pink'])
    return camLine

def merge_contour_into_contours(candCnts, hitCnt):
    if candCnts is None:
        candCnts = Contours([])
        
    if hitCnt is None:
        candplusCnts = candCnts
    else:
        candplusCnts = candCnts.append(hitCnt)
    return candplusCnts

def filter_hit_candidates_with_ball(hitPts3D, ballPt2D, P, quantile_threshold = 0.05):
    ptArr = hitPts3D.rasterize()
    if ballPt2D is None:
        ptArr_filtered = ptArr
    else:
        projArr = project_hit_point_cloud(hitPts3D, P, unique = False)
        dist = ((projArr - ballPt2D.coords) ** 2).sum(axis = 1) ** 0.5
        distance_threshold = np.quantile(dist, quantile_threshold) 
        print(f'Chose the {round(quantile_threshold * 100, 0)}% nearest points, closer than {round(distance_threshold, 1)}px in projection.')
        ptArr_filtered = ptArr[dist <= distance_threshold]
    return ptArr_filtered

def coords_to_ball(x, y, z):
    outBall = Ball3D((x, y, z))
    return outBall
        
def determine_serve_position(topPlayer_3D, botPlayer_3D, hitFlag, schemaLines = schemaLines):
    middle_x = schemaLines.get_half_line().leftPt.x
    
    if hitFlag == 'top_srv':
        groundPt_x = topPlayer_3D.get_ground_point().x
    
        if groundPt_x < middle_x:
            serve_pos = 'STD' ### serve far deuce
        else:
            serve_pos = 'STA' ### serve top ad
    
    elif hitFlag == 'bot_srv':
        groundPt_x = botPlayer_3D.get_ground_point().x
    
        if groundPt_x > middle_x:
            serve_pos = 'SBD' ### serve bot deuce
        else:
            serve_pos = 'SBA' ### serve bot ad
    
    else:
        serve_pos = None
    
    return serve_pos
    
# def determine_in_or_out(bouncePt, hit_type, schemaLines = schemaLines):
    
#     if hit_type == ''
    
    
    
    
    

        
        
    
    
    
