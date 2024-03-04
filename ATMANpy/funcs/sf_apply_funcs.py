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

def point2D_to_coords(pt):
    if pt is None:
        x, y = None, None
    else:
        x, y = pt.coords
    return x, y

def point3D_to_coords(pt):
    if pt is None:
        x, y, z = None, None, None
    else:
        x, y, z = pt.coords
    return x, y, z

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

def get_3D_hit_plane_point(P, topPlayerCnt, botPlayerCnt, hitFlag,
                           max_sideways_reach_in_feet = 5,
                           height_in_feet = 5, resolution = 1):
    ### As opposed to the above function, we calculate a line of points parallel to the baseline.
    ### if there are two such lines (for the hitter and the receiver), 
    ### all connections between any two points on the two lines will PARSIMONIOUSLY describe all relevant flight planes.
    ### Note that for the purposes of trajectory finding, height_in_feet is irrelevant
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
    
    max_reach_in_axis_units = max_sideways_reach_in_feet * const.SCALE
    
    ### TODO: Make depend on HANDEDNESS!
    #### decide angular limits
    if side == 'top':
        if is_serve:
            ### these go up, but generally close to the player sideways
            leftPt3D, rightPt3D = hingePt3D.left(max_reach_in_axis_units / 2), hingePt3D.right(max_reach_in_axis_units / 2)
        else:
            if stroke == 'fh':
                ### a forehand at the top is struck left from the player
                leftPt3D, rightPt3D = hingePt3D.left(max_reach_in_axis_units), hingePt3D
            elif stroke == 'bh':
                ### a backhand at the top is struck right from the player
                leftPt3D, rightPt3D = hingePt3D, hingePt3D.right(max_reach_in_axis_units)
    elif side == 'bot':
        if is_serve:
            ### these go up, but generally close to the player sideways
            leftPt3D, rightPt3D = hingePt3D.left(max_reach_in_axis_units / 2), hingePt3D.right(max_reach_in_axis_units / 2)
        else:
            if stroke == 'fh':
                ### a forehand at the bot is struck right from the player
                leftPt3D, rightPt3D = hingePt3D, hingePt3D.right(max_reach_in_axis_units)
            elif stroke == 'bh':
                ### a backhand at the bot is struck left from the player
                leftPt3D, rightPt3D = hingePt3D.left(max_reach_in_axis_units), hingePt3D
        
    hitLn = Line3D(leftPt3D, rightPt3D)
    pointArr = hitLn.rasterize(resolution)
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

def coords_to_ball3D(x, y, z):
    ball3D = Ball3D((x, y, z))
    return ball3D

def coords_to_ball2D(x, y):
    ball2D = Point2D((x, y))
    return ball2D


def project_ball(ball3D, P):
    ball2D_array = ball3D.project(P, resolution = 0.1)
    ball2D = Point2D(ball2D_array.mean(axis = 0))
    return ball2D
        
def determine_serve_position(topPlayer_3D, botPlayer_3D, hitFlag, schemaLines = schemaLines):
    middle_x = schemaLines.get_half_line().leftPt.x
    
    if hitFlag == 'top_srv':
        groundPt_x = topPlayer_3D.get_ground_point().x
    
        if groundPt_x < middle_x:
            srvFlag = 'STD' ### serve far deuce
        else:
            srvFlag = 'STA' ### serve top ad
    
    elif hitFlag == 'bot_srv':
        groundPt_x = botPlayer_3D.get_ground_point().x
    
        if groundPt_x > middle_x:
            srvFlag = 'SBD' ### serve bot deuce
        else:
            srvFlag = 'SBA' ### serve bot ad
    
    else:
        srvFlag = None
    
    return srvFlag
    
def is_net_degree(net_coordArr, netReg = None):
    ### calculate here to which degree that ball is projected into the net
    
    if net_coordArr is None:
        degree_in_net = None
    else:
        xArr = net_coordArr[:, :, :, 0] ### no need for y-value -> the net-plane is coplanar with the y-plane!
        zArr = net_coordArr[:, :, :, 2]
        
        tl, br = netReg['Top'].leftPt, netReg['Bottom'].rightPt
        z_hi, z_lo = tl.z, br.z
        x_hi, x_lo = br.x, tl.x
        
        ### calculate the distance within the net, in each dimension
        ### a negative value means that the ball is observed outside the net, a positive value means inside
        x_within_net = np.minimum(xArr - x_lo, x_hi - xArr)
        z_within_net = np.minimum(zArr - z_lo, z_hi - zArr)
        
        ### combine the distances
        ### the smaller value between both dimensions always indicates the closest net boundary.
        ### if 0 < z_within_net < x_within_net, the ball is projected within the net, but closer to the top or bottom of the net than to the sides
            ### -> distance within net ought to be z_within_net
        ### if z_within_net < 0 < x_within_net, the ball is outside the net in z dimension, and inside in x
            ### -> distance within net ought to be z_within_net
        ### if z_within_net < x_within_net < 0, the ball is outside the net in both x and y
            ### -> distance within net ought to be -sqrt(z_within_net**2 + x_within_net**2)
        d_within_net = np.where((x_within_net < 0) & (z_within_net < 0), 
                                -(x_within_net ** 2 + z_within_net ** 2) ** 0.5, 
                                np.minimum(x_within_net, z_within_net))    
        
        stretch_factor = 0.25
        degree_in_net = 1 / (1 + np.exp(-stretch_factor * d_within_net))
    return degree_in_net

def single_dim_viterbi_net(degree_in_net_array,
                           p = 0.95, ### probability that a ball in flight stays flying
                           epsilon = 0.01, ### probability that a non-net observation is emitted, given that the ball is in the net
                           q = 0.1 ### probability that a net observation is emitted, given that the ball is in flight
                           ):
    ### degree in_net_array shape: T
    T = len(degree_in_net_array)
   
    A = np.array([[p, 1 - p], [0, 1]]) ### transition matrix
    likelihood_of_degree_given_flight = lambda degree_in_net: degree_in_net * q + (1 - degree_in_net) * (1 - q)
    likelihood_of_degree_given_net = lambda degree_in_net: degree_in_net * (1 - epsilon) + (1 - degree_in_net) * epsilon
    
    trellis = np.full((T, 2), np.nan)
    pointers = np.full((T, 2), np.nan)
    
    ### Forward
    for idx, degree_in_net in enumerate(degree_in_net_array):
        if idx == 0: ### initialize
            trellis[idx] = 1, 0
            pointers[idx] = 0, 0
        else:
            ### probability that observation is emitted, given state
            if np.isnan(degree_in_net):
                p_emit = np.ones(2)
            else:
                p_emit = np.array([likelihood_of_degree_given_flight(degree_in_net), likelihood_of_degree_given_net(degree_in_net)])
            ### Probability that a trellis probability transition between periods i and j = i + 1
            p_trans = np.einsum('i, ij -> ij', trellis[idx - 1], A) ### shape: 
            ### Probability that the new state emits B
            p_all = np.einsum('ij, j -> ij', p_trans, p_emit)
            
            trellis[idx] = np.max(p_all, axis = 0)
            pointers[idx] = np.argmax(p_all, axis = 0)
       
    ### Backward
    states = np.full(T, fill_value = np.nan, dtype = int)
    for t in range(T, 0, -1):
        if t == T:
            s_before = np.argmax(trellis[-1]) ### initialization with highest final trellis probability state
        else:    
            s_before = pointers[t, states[t]]
        states[t-1] = s_before
    
    return states

### TODO: Alternative to Viterbi: Check when the ball is likeliest to have entered the net, if at all 
# 1) with an envelope 
    # -> if the ball is projected into the net up until the end, it's likely there
# 2) with a check of lateral and vertical speed 
    # -> if the ball is in the net, it's likely moving laterally slowly, and generally down (except for short bounces up)
def viterbi_net(net_coordArr, p = 0.95, epsilon = 0.01, q = 0.1):
    from ATMANpy.types.ShapesTennis3D import Net
    myNet = Net.make_singles()
    netReg = myNet.get_rectangular_net_region()
    
    degree_in_net = is_net_degree(net_coordArr, netReg = netReg)
    
    if degree_in_net is None:
        netArr = None
    else:
        netArr = np.apply_along_axis(single_dim_viterbi_net, axis = 2, arr = degree_in_net, 
                                     p = p, epsilon = epsilon, q = q)
        netArr = netArr.astype(bool)
        
    return netArr
        
def calculate_plane_penalties(paramArr, netArr, 
                              frame_cutoff = 5, ### calculate the lambda per time only after frame_cutoff frames -> avoid division by zero, avoid instability
                              verbose = False):
    if paramArr is None:
        penalties = None
    else:
        num_curr_points, num_next_points, T, _ = paramArr.shape
        
        ### parameters for the ball in flight, for evaluation
        if netArr is None:
            flight_paramArr = paramArr
        else:
            flight_paramArr = np.where(~netArr[:, :, :, np.newaxis], 
                                        paramArr, np.nan)
        
        time_idxs = np.arange(T)
        is_nan = np.isnan(paramArr).all(axis = (0, 1, 3))
        non_na_idxs = time_idxs[~is_nan]
        keep_filt = non_na_idxs >= frame_cutoff
        
        ### verify the integrity of the coordinates
        # first, get the non nan arrays 
        flight_paramArr_non_nan = flight_paramArr[:, :, non_na_idxs, :]     
        
        num_trajectories = num_curr_points * num_next_points
        if verbose: print(f'Out of {num_trajectories} trajectories...')
    
        # a) their height can not be negative
        muArr = flight_paramArr_non_nan[:, :, :, 1] ## get only flight path lambdas along ground
        mu_above_ground = muArr >= 0 ### TODO: maybe have a slightly negative allowed value here.
        
        traj_above_ground = mu_above_ground.all(axis = 2)
        num_traj_above_ground = traj_above_ground.sum()    
        if verbose: print(f'   ... {num_traj_above_ground} are always above ground')
        
        # a2) calculate a penalty
        traj_below_ground_penalty = - np.clip(muArr, a_min = None, a_max = 0).mean(axis = 2)
        
        # b) they must have monotonously increasing lambdas along the flight path at ground
        lambdaArr = flight_paramArr_non_nan[:, :, :, 0] ## get only flight path lambdas along ground
        delta_lambdaArr = np.diff(lambdaArr, axis = 2) ## first difference these in time
        
        monotonous_lambdas = (delta_lambdaArr > 0).all(axis = 2)
        num_monotonous_lambdas = monotonous_lambdas.sum()    
        if verbose: print(f'   ... {num_monotonous_lambdas} are fully monotonously increasing')
        
        # b2) calculate a penalty
        traj_nonmonotonous_penalty = - np.clip(delta_lambdaArr, a_min = None, a_max = 0).mean(axis = 2)
        
        # c) the lambdas must have monotonously decreasing first derivatives
        ### -> the ball decelerates in flight!
        # for this, we calculate the lambda divided by the index
        lambda_per_time = lambdaArr[:, :, keep_filt] / non_na_idxs[np.newaxis, np.newaxis, keep_filt]
        
        # this metric (essentially measuring the average ground speed up to each observed point)
        # ought to be decreasing -> at the beginning, speed is very high, then decreases
        delta_lambda_per_time = np.diff(lambda_per_time, axis = 2)
        
        decelerating_lambdas = (delta_lambda_per_time < 0).all(axis = 2)
        num_decelerating_lambdas = decelerating_lambdas.sum()    
        if verbose: print(f'   ... {num_decelerating_lambdas} are fully decelerating')
        
        traj_nondecelerating_penalty = np.clip(delta_lambda_per_time, a_min = 0, a_max = None).mean(axis = 2)
    
        # # d) start and end points must be 'reasonably close' to hypothetical points
        # fs_dev = coordArr[:, :, 0] - pointArr[:, :, 0]
        # fs_dist = (fs_dev ** 2).sum(axis = -1) ** 0.5
        # fs_dist_m = fs_dist / const.SCALE * const.FOOT_TO_METER
        
        # ls_dev = coordArr[:, :, -1] - pointArr[:, :, -1]
        # ls_dist = (ls_dev ** 2).sum(axis = -1) ** 0.5
        # ls_dist_m = ls_dist / const.SCALE * const.FOOT_TO_METER
        
        # join all the conditions
        flawless_trajectory = traj_above_ground & monotonous_lambdas & decelerating_lambdas
        num_flawless_trajectories = flawless_trajectory.sum()
        if verbose: print(f'   ... {num_flawless_trajectories} are flawless!')
        
        # calculate a penalty term
        penalties = traj_below_ground_penalty + traj_nonmonotonous_penalty + traj_nondecelerating_penalty
        tiny_pen_thresh, small_pen_thresh = 0.01, 0.1
        tiny_pen, small_pen = penalties < tiny_pen_thresh, penalties < small_pen_thresh
        num_tiny_pen = tiny_pen.sum()
        num_small_pen = small_pen.sum()
        if verbose: print(f'   ... {num_tiny_pen} are completely acceptable')
        if verbose: print(f'   ... {num_small_pen} are reasonably good')
        
    return penalties
    

def resample(coordArr, paramArr, planeArr, pointArr, netArr, penalties,
             logit_coeff = 1, sampling_rate = None, sampling_number = None):
    # logit_coeff ---> the higher the number, the more we favor low penalty points
    if penalties is None:
        coordArr, paramArr, planeArr, pointArr, netArr, penalties = None, None, None, None, None, None
    else:
        if np.isnan(penalties).all(): ### this happens only if there is not a single observation in the flight path.
            penalties = np.zeros_like(penalties) # in this case, all hit and rec points have the same penalty of 0
        
        rng = np.random.default_rng()
        
        logit = np.exp(-logit_coeff * penalties)
        probs = logit / logit.sum()
        
        hit_n, rec_n = probs.shape
        prob_n = hit_n * rec_n
        
        if sampling_rate is not None:
            out_n = int(sampling_rate * prob_n) + 1
        elif sampling_number is not None:
            out_n = sampling_number
        else:
            raise Exception('Specify either a sampling_number or a sampling_rate')
        
        ### SELECT indices 
        probs_flat = probs.flatten()
        prob_idxs_resampled_flat = rng.choice(prob_n, out_n, p = probs_flat, replace = True)
        hit_idxs_resampled, rec_idxs_resampled = prob_idxs_resampled_flat // rec_n, prob_idxs_resampled_flat % rec_n
        
        ### RESAMPLE using selected indices
        coordArr = coordArr[hit_idxs_resampled, rec_idxs_resampled]
        paramArr = paramArr[hit_idxs_resampled, rec_idxs_resampled]
        planeArr = planeArr[hit_idxs_resampled, rec_idxs_resampled]
        pointArr = pointArr[hit_idxs_resampled, rec_idxs_resampled]
        if netArr is None:
            netArr = None
        else:
            netArr = netArr[hit_idxs_resampled, rec_idxs_resampled]
        penalties = penalties[hit_idxs_resampled, rec_idxs_resampled]
        
    return coordArr, paramArr, planeArr, pointArr, netArr, penalties

def fit_trajectory(planeArr, netArr, groundArr = None, vertArr = None, rpmArr = None, 
                   observation_threshold = 3,
                   sum_of_squares = True, number_of_local_minima = 15,
                   ram_allowed_in_gb = 32):
    
    from ATMANpy.models.BallTrajectory.flight_bounce_model import fit_trajectories_pointless
    
    if planeArr is None:
        trajArr, groundArr_filt, vertArr_filt, rpmArr_filt = None, None, None, None
    else:
        
        #kill plane observations in net
        if netArr is not None:
            planeArr = np.where(~netArr[:, :, np.newaxis], planeArr, np.nan)
          
        coords_are_not_nan = ~np.isnan(planeArr).all(axis = (0, 2))
        valid_observations = coords_are_not_nan.sum()
        if valid_observations < observation_threshold:
            trajArr, groundArr_filt, vertArr_filt, rpmArr_filt = None, None, None, None
        else:
            ### Fit Trajectories
            trajArr, groundArr_filt, vertArr_filt, rpmArr_filt = fit_trajectories_pointless(planeArr,
                                                                                            groundArr, vertArr, rpmArr, 
                                                                                            sum_of_squares = sum_of_squares,
                                                                                            number_of_local_minima = number_of_local_minima,
                                                                                            ram_allowed_in_gb = ram_allowed_in_gb)
            
            if not netArr is None:
                num_frames = groundArr_filt.shape[-1]
                netArr_brdcst = netArr[:, np.newaxis, :num_frames]
                
                groundArr_filt = np.where(~netArr_brdcst, groundArr_filt, np.nan)
                vertArr_filt = np.where(~netArr_brdcst, vertArr_filt, np.nan)
                rpmArr_filt = np.where(~netArr_brdcst, rpmArr_filt, np.nan)
    
    return trajArr, groundArr_filt, vertArr_filt, rpmArr_filt
        
def trajectory_audio_prob(bounceArr, vertArr_filt, netArr):
    ### bounceArr shape: T x 5 (pcond1, pcond2, prior0, prior1, prior2)
    ### vertArr_filt shape: plane x traj x T
    ### netArr shape: None or plane x T
    if bounceArr is None or vertArr_filt is None: ### occurs if final inferred flight point is taken to be hit point 
        bounce_probs = None
    else:
        ### Prep some variables
        bounce_cond1 = bounceArr[:, 0]
        bounce_cond2 = bounceArr[:, 1]
        plane, traj, time = vertArr_filt.shape # plane (between a hit and a rec point), traj (given by initial conditions), time in frames
        vertArr_filt_reshaped = vertArr_filt.reshape((plane * traj, time))
        
        ### Construct Plane Specific Priors
        priorArr_in_flight = bounceArr[0, 2:] ### priors are constant across T
        priorArr_in_net = np.array([0.9, 0.09, 0.01])
        
        if netArr is None:
            priors_by_plane = np.repeat(priorArr_in_flight[np.newaxis, :], repeats = plane, axis = 0)
        else:
            plane_in_net = netArr.argmax(axis = 1).astype(bool)
            priors_by_plane = np.where(plane_in_net[:, np.newaxis], priorArr_in_net[np.newaxis, :], priorArr_in_flight[np.newaxis, :])
        
        ### Gather implied trajectory bounce data ###
        frm_bounces = [np.arange(time)[is_bounce] for is_bounce in vertArr_filt_reshaped == 0]
        num_bounces = (vertArr_filt_reshaped == 0).sum(axis = 1)
        
        ### figure out conditional bounce probabilities
        bounce_cond_probs = []
        for bnc, num in zip(frm_bounces, num_bounces):
            if num == 0:
                joint_bounce_prob = 1
            elif num == 1:
                joint_bounce_prob = bounce_cond1[bnc].prod()
            elif num == 2:
                joint_bounce_prob = bounce_cond2[bnc].prod()
            else: ### since we don't have probabilities for three bounces (which could likely not be heard well anyway)
                joint_bounce_prob = bounce_cond2[bnc].prod()
                
            bounce_cond_probs.append(joint_bounce_prob)
        bounce_cond_probs = np.array(bounce_cond_probs).reshape(plane, traj)
        
        ### figure out priors
        num_bounces = num_bounces.reshape(plane, traj)
        
        choose_prior = lambda plane_bounces, plane_priors: np.where(plane_bounces == 0, plane_priors[0], 
                                                                    np.where(plane_bounces == 1, plane_priors[1], 
                                                                             plane_priors[2]))
        
        bounce_prior_probs = np.array([choose_prior(plane_bounces, plane_priors) for plane_bounces, plane_priors in zip(num_bounces, priors_by_plane)])
    
        ### Combine priors and conditional probabilities
        bounce_probs = bounce_cond_probs * bounce_prior_probs
        return bounce_probs
    
def trajectory_deviation_prob(trajArr):
    if trajArr is None:
        dev_probs = None
    else:
        dev_probs = 1 / (1 + trajArr[:, :, -1])
    return dev_probs
    
    
def compare_trajectories_with_audio(vertArr_filt, bounceFrm, prior_override = None):
    
    ######## Integrate bounce audio ##########
    bounce_frames = bounceFrm.index.tolist()
    hit_frame = bounce_frames[0] - 1 
    flight_frames = [hit_frame, *bounce_frames]
    
    if prior_override is None:
        p0, p1, p2 = bounceFrm['prior0'].unique()[0], bounceFrm['prior1'].unique()[0], bounceFrm['prior2'].unique()[0]
    else:
        p0, p1, p2 = prior_override
    
    plane, traj, time = vertArr_filt.shape # plane (between a hit and a rec point), traj (given by initial conditions), time in frames
    vertArr_filt_reshaped = vertArr_filt.reshape((plane * traj, time))
    
    ### Gather implied trajectory bounce data ###
    frm_bounces = [np.array(flight_frames)[is_bounce] for is_bounce in vertArr_filt_reshaped == 0]
    num_bounces = (vertArr_filt_reshaped == 0).sum(axis = 1)
    
    ### Calculate conditional bounce probabilities and priors
    bounce_cond_probs = []
    for bnc, num in zip(frm_bounces, num_bounces):
        if hit_frame in bnc:
            joint_bounce_prob = 0 ### there can be no bounce in the current hit frame, after the hit
        else:
            if num == 0:
                joint_bounce_prob = 1
            elif num < 3:
                joint_bounce_prob = bounceFrm.loc[bnc, f'bounce_prob_cond{num}'].prod()
            else: ### since we don't have probabilities for three bounces (which could likely not be heard well anyway)
                joint_bounce_prob = bounceFrm.loc[bnc, 'bounce_prob_cond2'].prod()
                
        bounce_cond_probs.append(joint_bounce_prob)
    bounce_cond_probs = np.array(bounce_cond_probs)
    
    bounce_priors = np.where(num_bounces == 0, p0, 
                              np.where(num_bounces == 1, p1, p2))
    
    ### Join Conditional Probabilities and Priors
    bounce_probs = bounce_cond_probs * bounce_priors
    bounce_probs = bounce_probs.reshape(plane, traj) ### shape back to input shape
    return bounce_probs

def calculate_trajectory_coordinates(groundArr_filt, vertArr_filt, pointArr):
    # groundArr_filt.shape, vertArr_filt.shape ### Plane x Traj x T
    # pointArr.shape ### Plane x 2 (start, end) x 3 (x, y, z)
    if groundArr_filt is None:
        coords = None
    else:
        ### rescale ground and vert arrays
        groundArr_resc = groundArr_filt * const.SCALE / const.FOOT_TO_METER
        vertArr_resc = vertArr_filt * const.SCALE / const.FOOT_TO_METER
        
        ### calculate the flight line of the ball for each path, and a length of the plane line
        flatArr = pointArr[:, :, [0, 1]] ### points projected on the ground
        fsArr, lsArr = flatArr[:, 0], flatArr[:, 1]
        vecArr = lsArr - fsArr
        distArr = (vecArr ** 2).sum(axis = 1) ** 0.5 ### shape: Plane
    
        ### for each plane, divide the distance travelled along the plane by the length of the plane line 
        lmbd = groundArr_resc / distArr[:, np.newaxis, np.newaxis] ### shape: Plane x Traj x T
        
        ### calculate the ground coordinates
        ground_coords = fsArr[:, np.newaxis, np.newaxis] + lmbd[:, :, :, np.newaxis] * vecArr[:, np.newaxis, np.newaxis] ### shape: Plane x Traj x T x 2 (x, y)
        vert_coords = vertArr_resc[:, :, :, np.newaxis] ### shape: Plane x Traj x T x 1 (z)
        
        traj_coordArr = np.concatenate([ground_coords, vert_coords], axis = -1)
        return traj_coordArr
        

def make_trajectory_arrays(best_plane, best_traj, 
                           traj_coordArr, coordArr, planeArr, netArr,
                           trajArr, groundArr_filt, vertArr_filt, rpmArr_filt,
                           v_init_ground_space = None, v_init_vert_space = None, spin_rate_space = None, height_space = None):
    
    ### traj is best is a boolean array with N entries, one for each trajectory
    ### all other arrays are N (x T) x ... arrays
    if trajArr is None:
        outArr, columns = None, None
    else:
        T = traj_coordArr.shape[2]
        ### 1) Gather optimal coordinates ... 
        # ... for the observed points ...
        coordArr_opt = coordArr[best_plane]
        # ... and the fitted trajectories
        traj_coordArr_opt = traj_coordArr[best_plane, best_traj]
    
        ### 2) Gather optimal planar coordinates ...
        # ... for the observed points ...
        planeArr_opt = planeArr[best_plane] 
        # ... and the fitted trajectories
        groundArr_opt = groundArr_filt[best_plane, best_traj] * const.SCALE / const.FOOT_TO_METER
        vertArr_opt = vertArr_filt[best_plane, best_traj] * const.SCALE / const.FOOT_TO_METER
        rpmArr_opt = rpmArr_filt[best_plane, best_traj]     
        traj_planeArr_opt = np.stack([groundArr_opt, vertArr_opt, rpmArr_opt], axis = -1)
        
        ### 3) Gather optimal initial conditions ...
        traj_paramArr_opt = trajArr[best_plane, best_traj, :4]
        vx0_idx, vy0_idx, rpm0_idx, height_idx = traj_paramArr_opt.astype(int)
        vx0, vz0, rpm0, height0 = v_init_ground_space[vx0_idx], v_init_vert_space[vy0_idx], spin_rate_space[rpm0_idx], height_space[height_idx]
        condArr_opt = np.ones((T, 4)) * np.array([[vx0, vz0, rpm0, height0]])
        
        ### 4) Gather the net data -> when is the ball in the net?
        if netArr is None:
            netArr_opt = np.zeros((T, 1), dtype = 'bool')
        else:
            netArr_opt = netArr[best_plane][:, np.newaxis]
        
        ### 5) Join Arrays
        outArr = np.concatenate([coordArr_opt[:T],
                                 traj_coordArr_opt[:T],
                                 planeArr_opt[:T],
                                 traj_planeArr_opt[:T],
                                 condArr_opt[:T],
                                 netArr_opt[:T]],
                                axis = 1)
        columns =  [*['x_o', 'y_o', 'z_o'], 
                    *['x_t', 'y_t', 'z_t'], 
                    *['px_o', 'pz_o'],
                    *['px_t', 'pz_t', 'rpm_t'],
                    *['vx0', 'vz0', 'rpm0', 'height0'],
                    *['is_net']]
                        
        assert outArr.shape[1] == len(columns)
        
    return outArr, columns


    
    

        
        
    
    
    
