# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:24:00 2024

@author: gianl
"""

import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import get_width_height

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points


from ATMANpy.types.Visual3D import Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D

# def get_ball_cloud_around(centerPt3D, hitFlag):
    
def load(vid, sto):
    HPCF = ConstFeatures.from_store(vid, sto, featnameList = ['H', 'P', 'M_int', 'M_ext'])
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
    ballSF = StreamFeature.from_store(vid, sto, featname = 'ball_choicePt')
    hitSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)
    return HPCF, segBF, ballSF, hitSF, plySF, audFrm

def save(vid, sto):
    pass


s_x, s_y = 1920, 1080
sensor_dims = s_x, s_y
cam_height = 4000
o_x = s_x / 2
o_y = s_y / 2

def check(*objs, resolution = 10):
    ### Visualise
    myScene = Scene.make_court()
    
    for obj in objs:
        myScene = myScene.put(obj)
    
    ctPt = CourtLines.make().get_center_point()
    x, y = ctPt.coords
    ctBall3D = Ball3D((x, y, 0), radius = 10, color = const.BGR['intense_red'])
    testScene = Scene.make_empty().put(ctBall3D)
    
    myCamera = Camera.calibrated_AO_behind()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    myCamera = Camera.low_angled_behind()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    myCamera = Camera.mid_lateral()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    s_x, s_y = 1920, 1080
    sensor_dims = s_x, s_y
    cam_height = 4000
    o_x = s_x / 2
    o_y = s_y / 2
    
    myCamera = Camera.birds_eye(cam_height = cam_height, o_x = o_x, o_y = o_y, sensor_dims = sensor_dims)
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    # testView = View.make(myCamera, testScene, resolution = resolution)
    # testView.snap().show()

def view3D(objects3D, resolution = 10):
    ### Visualise
    myScn = Scene.make_court()
    
    for obj in objects3D:
        myScn = myScn.put(obj)
    
    myCamera = Camera.calibrated_AO_behind()
    myView = View.make(myCamera, myScn, resolution = resolution)
    myView.snap().show()

sample_frame = 255
resolution = 5

distance_threshold = 10

##### TRAJECTORIES ####
from ATMANpy.models.BallTrajectory.flight_bounce_model import load_trajectories, initialize
groundArr, vertArr, rpmArr, trajDict = load_trajectories('FullTrajectories_tight', dtype = 'float32')

# groundArr_halfstep, vertArr_halfstep, trajDict_halfstep = load_trajectories('TrajectoriesRPM_tight_halfstep', dtype = 'float32')

### initialise search spaces with proper arguments
import inspect
kwargList = inspect.getargspec(initialize).args
kwargs = {k: v for k, v in trajDict.items() if k in kwargList}
v_init_ground_space, v_init_vert_space, spin_rate_space, height_space = initialize(dtype = 'float32', 
                                                                                   squeeze = True,  
                                                                                   **kwargs)

def visualize_penalties(points, pens, resolution = 0.5):
    bad_color = const.BGR['intense_red']
    good_color = const.BGR['grass_green']
    
    def mix_colors(c1, c2, lmbd):
        return tuple(np.round(np.array(c1) * lmbd + np.array(c2) * (1 - lmbd), 0).astype(int))
        
    fs, ls = pens.shape
    
    points_re = points.reshape((fs * ls, 2, 3))
    pens_re = pens.reshape((fs * ls))
    
    fs_re = points_re[:, 0]
    ls_re = points_re[:, 1]
    
    ### figure out penalty quantiles
    pcts = np.linspace(0, 1, 11)
    quantiles = np.quantile(pens_re, pcts)
    
    ### Visualise
    myScn = Scene.make_court()
    
    for fs_coords, ls_coords, pen in zip(fs_re, ls_re, pens_re):
        ### figure out color
        q = pcts[np.argmin(np.abs(quantiles - pen))]
        c = mix_colors(bad_color, good_color, q)
        
        fsPt, lsPt = Point3D(fs_coords), Point3D(ls_coords)
        myLn = Line3D(fsPt, lsPt, color = c)
        
        myScn = myScn.put(myLn)
    
    myCamera = Camera.calibrated_AO_behind()
    myView = View.make(myCamera, myScn, resolution = resolution)
    myView.snap().show()
    
    # myCamera = Camera.mid_lateral()
    # myView = View.make(myCamera, myScn, resolution = resolution)
    # myView.snap().show()
    
    myCamera = Camera.birds_eye(cam_height = cam_height, o_x = o_x, o_y = o_y, sensor_dims = sensor_dims)
    myView = View.make(myCamera, myScn, resolution = resolution)
    myView.snap().show()
        
def resample(coordArr, paramArr, planeArr, pointArr, penalties,
             logit_coeff = 1, sampling_rate = None, sampling_number = None):
    # logit_coeff ---> the higher the number, the more we favor low penalty points
    if penalties is None:
        return None, None, None, None, None
    
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
    
    probs_flat = probs.flatten()
    prob_idxs_resampled_flat = rng.choice(prob_n, out_n, p = probs_flat, replace = True)
    
    hit_idxs_resampled, rec_idxs_resampled = prob_idxs_resampled_flat // rec_n, prob_idxs_resampled_flat % rec_n
    
    coordArr_resampled = coordArr[hit_idxs_resampled, rec_idxs_resampled]
    paramArr_resampled = paramArr[hit_idxs_resampled, rec_idxs_resampled]
    planeArr_resampled = planeArr[hit_idxs_resampled, rec_idxs_resampled]
    pointArr_resampled = pointArr[hit_idxs_resampled, rec_idxs_resampled]
    penalties_resampled = penalties[hit_idxs_resampled, rec_idxs_resampled]
    
    return coordArr_resampled, paramArr_resampled, planeArr_resampled, pointArr_resampled, penalties_resampled

def resampled_point_cloud(trajDict, logit_coeff = 10, std_in_vx = 3):
    rng = np.random.default_rng()
    
    resampled_point_pen_dict = {}
    for hit_frame in trajDict:
        coordArr, paramArr, planeArr, pointArr, penalties = trajDict[hit_frame]
        if pointArr is None and penalties is None: ### fails on the last frame -> not a hit frame, strictly, but a hypothetical flight frame
            resampled_point_pen_dict[hit_frame] = Point3D(rec_pts_resampled.T)
        else:
            ### Resample
            hit_n = coordArr.shape[0]
            _, _, _, pointArr_resampled, _ = resample(coordArr, paramArr, planeArr, pointArr, penalties,
                                                      logit_coeff = logit_coeff, sampling_number = hit_n)
            ### Add Uncertainty
            pointArr_resampled = rng.normal(pointArr_resampled, std_in_vx)
            
            ### Write to Dict
            hit_pts_resampled, rec_pts_resampled = pointArr_resampled[:, 0], pointArr_resampled[:, 1]
            resampled_point_pen_dict[hit_frame] = Point3D(hit_pts_resampled.T)
    
        ### TODO: use the previous rec points to regularize the following hit points (-> is same!)
        ### Do this here or after trajectories are actually done?
        
    resSF = StreamFeature(resampled_point_pen_dict, name = 'cldPt3D')
    return resSF

########################

def main(conf, params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    width, height = get_width_height(cap)
    
    HPCF, segBF, ballSF, hitSF, plySF, audFrm = load(vid, sto)
    
    courtLines = CourtLines.make()
    courtDims = courtLines.see_dimensions()
    
    audSF = StreamFeatures.from_dataframe(audFrm)
    
    P = HPCF['P'].feature
    H = HPCF['H'].feature
    M_int = HPCF['M_int'].feature
    M_ext = HPCF['M_ext'].feature
    matchCamera = Camera(M_int, M_ext, (width, height))
    
    ### find 3D player positions
    mySF = plySF.merge(HPCF['P'])
    ply3DSF = mySF.apply(sa.find_player_positions_3D)
    
    ### make point cloud
    hit_player_ball_SF = hitSF.merge(plySF).merge(HPCF['P']).merge(ballSF)
    params = {'reaches_in_feet': [2.5, 3, 3.5, 4, 4.5, 5],
              'resolution_general': 0.002, 'resolution_observed': 0.2,  ### originally: resolutions = 0.005, 0.5
              'quantile_threshold': 0.01}
    cld3DSF_hits = hit_player_ball_SF.apply(sa.get_3D_hit_point_cloud, **params)
    
    ### add in a points cloud after the last shot, to simulate the ball flying off after last hit.
    hit_player_ball_SF = hitSF.merge(plySF).merge(HPCF['P']).merge(ballSF, how = 'outer')
    hit_player_ball_SFs = hit_player_ball_SF.conform(segBF).filter(segBF).split()
    
    params = {'height_in_feet': 5, 'resolution': 0.01, 'radius_in_feet': 15} ### originally: resolution = 0.02
    cld3DSFs_last = hit_player_ball_SFs.apply_whole(sf.add_in_hit_point, **params)
    cld3DSF_last = cld3DSFs_last.concatenate()
    
    cld3DSF = cld3DSF_hits.extend(cld3DSF_last['cldPt3D']).sort()
    hitSF_ext = hitSF.extend(cld3DSF_last['hitFlag']).sort()
    
    # # ### check point cloud
    # for frame in cld3DSF_last.get_frames()[3:6]:
    #     topplayer3D = ply3DSF[frame]['topPlayer_3D']
    #     botplayer3D = ply3DSF[frame]['botPlayer_3D']
    #     topplayer3D = Player.make(topplayer3D.coords)
    #     botplayer3D = Player.make(botplayer3D.coords)
    #     check(topplayer3D, botplayer3D, cld3DSF_last[frame])
    #     BGRFrame.from_video(cap, frame).show()
    
    ### from player positions at hit and bounce audio, calculate the probability of bounce
    mySFs = hitSF_ext.merge(ply3DSF, how = 'outer').merge(audSF, how = 'outer').conform(segBF).filter(segBF).split()
    bounceSF = mySFs.apply_whole(sf.combine_bounce_priors_and_probs).concatenate()
    
    ### Calculate Trajectories
    mySFs = ballSF.merge(cld3DSF, how = 'outer').merge(bounceSF, how = 'outer').conform(segBF).filter(segBF).split()
    mySF = mySFs[0]
    
    trajDicts_raw = mySFs[:5].apply_whole(sf.test_trajectories, matchCamera = matchCamera, P = P)
    
    params = {'logit_coeff': 10, 'std_in_vx': 3}
    cld3DSFs_resampled = SFList([resampled_point_cloud(trajDict, **params) for trajDict in trajDicts_raw])
    cld3DSF_resampled = cld3DSFs_resampled.concatenate()
    
    ### Resampling like this works well when the low penalty hit points at two successive hits are close to each other
    ### Note that when we re-launch the trajectory test in the second instance, 
    ### the receiving points for any iteration except the last are not used 
    ### -> instead, we use the next hit points. however, low penalties apply to hit and receive points in the previous shot.
    ### TODO: Therefore, introduce another penalty that somehow penalises rec points which are far away from successive hit points
    
    ### Idea for this penalty -> look at implied paths somehow 
        # -> get the points, and connect at previous rec point and next hit point
        # by calculating distance. probability of a path decreases with increasing distance (viterbi?)
    
    mySFs = ballSF.merge(cld3DSF_resampled, how = 'outer').merge(bounceSF, how = 'outer').conform(segBF).filter(segBF).split()
    mySF = mySFs[0]
    
    trajDicts_resampled = mySFs[:5].apply_whole(sf.test_trajectories, matchCamera = matchCamera, P = P)
        
    idx = 2
    trajDict = trajDicts_resampled[idx]
    mySF = mySFs[idx]
    
    params = {'logit_coeff': 10, 'sampling_rate': 0.05}
    trajDict_downsampled = {frame: resample(*arrays, **params) for frame, arrays in trajDict.items()}
    
    observation_threshold = 3
    trajFrms = []
    event_frames = list(trajDict.keys())
    for hit_frame in event_frames[:-1]:
        rec_frame = event_frames[event_frames.index(hit_frame) + 1]
        flight_frames = list(range(hit_frame, rec_frame + 1))
    
        coordArr, paramArr, planeArr, pointArr, penalties = trajDict_downsampled[hit_frame]
        
        example_coords_are_nan = np.isnan(coordArr[0])
        valid_observations = (~example_coords_are_nan).all(axis = 1).sum()
        if valid_observations >= observation_threshold:
            ### Fit Trajectories
            trajArr, groundArr_filt, vertArr_filt, rpmArr_filt = sf.fit_trajectories(coordArr, paramArr, planeArr, pointArr, 
                                                                                     groundArr, vertArr, rpmArr)
            
            ### Compare Trajectories with audio
            bncSF = mySF[['bounce_prob_cond1', 'bounce_prob_cond2', 'prior0', 'prior1', 'prior2']]
            bounceFrm = bncSF[flight_frames].to_dataframe().iloc[1:] ### first frame can not have bounce (after hit)
            bounce_probs = sf.compare_trajectories_with_audio(vertArr_filt, bounceFrm)
            
            ### Compare Trajectories with observed path
            traj_probs = 1 / (trajArr[:, 3] + 1)
            
            ### Join Probs
            probs = bounce_probs * traj_probs
            traj_is_best = probs == probs.max()
            
            trajFrm = sf.make_trajectory_frame(traj_is_best, 
                                               coordArr, paramArr, planeArr, pointArr, 
                                               groundArr_filt, vertArr_filt, rpmArr_filt,
                                               v_init_ground_space, v_init_vert_space, spin_rate_space,
                                               trajArr,
                                               flight_frames = flight_frames)
                
            ### Visualize
            x, y = trajFrm['px_o'].values, trajFrm['pz_o'].values
            t_x, t_y = trajFrm['px_t'].values, trajFrm['pz_t'].values
            plt.scatter(x, y)
            plt.scatter(t_x, t_y)
            plt.show()
        else:
            trajFrm = pd.DataFrame()
            
        trajFrms.append(trajFrm)
    
    
    trajFrm = trajFrms[15]
    
    ### Visualize on the actual Stream
    trajFrm = pd.concat([trajFrm.iloc[:-1] for trajFrm in trajFrms])
    trajFrm = trajFrm[['x_t', 'y_t', 'z_t']]
    trajFrm.columns = ['x', 'y', 'z']
    ballSF = StreamFeatures.from_dataframe(trajFrm).apply(sa.coords_to_ball).rename('ball3D')
    ballSF = ballSF.merge(HPCF['P'])
    ballSF = ballSF.apply(sa.project_ball)
    
    Stream.from_SF(ballSF, cap).show()
    
    ### Attempt to find 'best' bounce points for the function.
    # idea: add a bounce point at different lambdas (with mu = 0, since height = 0)
    # and then check the fit of a constructed spline representation
    # the quality of the fit is a measure of the support for the bounce point hypothesis.
    from ATMANpy.funcs.spin_funcs import calculate_planar_ball_flight, get_optimal_trajectory_indices
    
    def get_planar_vector(trajFrm):
        """ return vector of length 1m in planar direction """
        fsPt, lsPt = trajFrm.iloc[0][['x', 'y']], trajFrm.iloc[-1][['x', 'y']]
        vecPt = lsPt - fsPt
        len_vecPt = (vecPt ** 2).sum() ** 0.5
        norm_vecPt = (vecPt / len_vecPt) * const.SCALE / const.FOOT_TO_METER
        return norm_vecPt
    
    planFrm = calculate_planar_ball_flight(trajFrm[['x', 'y', 'z']])
    curr_hit_frame, next_hit_frame = int(planFrm.index[0]), int(planFrm.index[-1])
    planVec = get_planar_vector(trajFrm)
    planOrig = trajFrm.iloc[0][['x', 'y']]
    hit_height = trajFrm.loc[curr_hit_frame, 'z']
    
    idx_vals = planFrm.index.values
    idx_vals_halfstep = idx_vals + 0.5
    idx_halfstep = np.sort(np.concatenate([idx_vals, idx_vals_halfstep]))
    
    planFrm_halfstep = planFrm.reindex(idx_halfstep)
    
    ### use a ground speed interpolation to approximate the ground position at all times
    ground_itpl = planFrm_halfstep['ground_m'].interpolate(method = 'quadratic')
    zero_height = - hit_height / const.SCALE * const.FOOT_TO_METER
    
    initial_conditions = {}
    fitted_trajs = {}
    mean_deviations = {}
    bounce_deviations = {}
    itpl_bounce_points = {}
    for hypothetical_bounce_frame in idx_vals_halfstep[:-1]: ### bounce can not be after the second hit frame
        curr_idx = np.argmax(idx_halfstep == hypothetical_bounce_frame)
        
        bounce_point_planar = (ground_itpl[hypothetical_bounce_frame], zero_height)
        bounce_point_3D = ground_itpl[hypothetical_bounce_frame] * planVec + planOrig
        bounce_point_3D.at['z'] = 0
        
        truncFrm = planFrm_halfstep.loc[:hypothetical_bounce_frame].copy()
        truncFrm.at[hypothetical_bounce_frame] = bounce_point_planar
        
        
        pos_ground_measured_flight, pos_vert_measured_flight = truncFrm.values.T
        v_ground_idx, v_vert_idx, rpm_idx, dev = get_optimal_trajectory_indices(groundArr, vertArr,
                                                                                pos_ground_measured_flight, pos_vert_measured_flight, 
                                                                                return_dev = True, sum_of_squares = False)
        
        ### find fitted trajectories 
        fitted_ground = groundArr[v_ground_idx, v_vert_idx, rpm_idx]
        fitted_vert = vertArr[v_ground_idx, v_vert_idx, rpm_idx]
        fitted_traj = np.vstack([fitted_ground, fitted_vert]).T
        
        ### find theoretical bounce point of fitted trajectory
        # -> in the best case, this is consistent with the bounce point hypothesis 
        fitted_bounce_idx = np.argmax(fitted_traj[:, 1] <= zero_height)
        fitted_bounce_frame = curr_hit_frame + fitted_bounce_idx // 2
        bounce_dev = np.abs(hypothetical_bounce_frame - fitted_bounce_frame)
        
        ### calculate mean deviation, corrected for 3 degrees of freedom -> v_vert, v_hori, spin
        corr_denominator = (~np.isnan(dev)).sum() - 3
        mean_dev = np.nansum(dev) / corr_denominator 
        
        fitted_trajs[hypothetical_bounce_frame] = fitted_traj
        initial_conditions[hypothetical_bounce_frame] = (v_ground_idx, v_vert_idx, rpm_idx)
        mean_deviations[hypothetical_bounce_frame] = mean_dev
        bounce_deviations[hypothetical_bounce_frame] = bounce_dev
        itpl_bounce_points[hypothetical_bounce_frame] = bounce_point_3D

    devFrm = pd.concat([pd.Series(mean_deviations, name = 'mean_dev'), pd.Series(bounce_deviations, name = 'bounce_dev')], axis = 1)
    exp_devFrm = np.exp(-devFrm)
    prob_devFrm = exp_devFrm / exp_devFrm.sum()
    # prob_devFrm = prob_devFrm.reindex(idx_halfstep).interpolate(limit_area = 'inside').reindex(idx_vals).dropna()
    
    ### add in the sound information
    bounceSer = audFrm.loc[curr_hit_frame + 1: next_hit_frame - 1, 'bounce']
    # bring to half frame index
    bounceSer = bounceSer.reindex(idx_halfstep).interpolate(limit_area = 'inside').reindex(idx_vals_halfstep).dropna()
    
    probFrm = pd.concat([prob_devFrm, bounceSer], axis = 1)
    probSer = probFrm.prod(axis = 1) ** (1 / 3)

    bounce_frame = probSer.index[probSer.argmax()]
    bounce_frames = [int(bounce_frame), int(bounce_frame) + 1]
    bounce_point = itpl_bounce_points[bounce_frame]
    bouncePt = Point3D((bounce_point.x, bounce_point.y, bounce_point.z))
    
    #### Locate players 
        # (especially for serve, we need to know from which side the serve happens)
        # (maybe we could do this in the hit label script...)
    srvSF = ply3DSF.merge(hitSF, how = 'inner').apply(sa.determine_serve_position)
    
    
    #### check if bounce occurs inside or outside of court
    bounce_point
    
    #############
    ### Create Fitted Trajectory Object ###
    from ATMANpy.types.Abstract import Trajectory, TrajectoryFitted, PlayerMovement, Shot
    
    # a) flight trajectory
    obsFrm = trajFrm.loc[(trajFrm.source == 'observed'), ['x', 'y', 'z']]
    synFrm = trajFrm.loc[(trajFrm.source == 'hit'), ['x', 'y', 'z']]
    
    evtDict = {}
    
    evtDict[curr_hit_frame] = hitSF[curr_hit_frame]
    
    for frame in bounce_frames:
        synFrm.at[frame] = bounce_point
        evtDict[frame] = 'bounce'
    
    evtDict[next_hit_frame] = hitSF[next_hit_frame]
        
    synSF = StreamFeatures.from_dataframe(synFrm).apply(sa.coords_to_ball).rename('synBall')
    obsSF = StreamFeatures.from_dataframe(obsFrm).apply(sa.coords_to_ball).rename('obsBall')
    evtSF = StreamFeature(evtDict, name = 'event')
    
    trajSF = obsSF.merge(synSF, how = 'outer').merge(evtSF, how = 'outer')
    
    v_ground_idx, v_vert_idx, rpm_idx = initial_conditions[bounce_frame]
    v_init_ground = v_init_ground_space[v_ground_idx]
    v_init_vert = v_init_vert_space[v_vert_idx]
    spin_rpm = spin_rate_space[rpm_idx]
    
    pre_bounce_frames = list(range(curr_hit_frame, bounce_frames[-1] + 1))
    post_bounce_frames = list(range(bounce_frames[0], next_hit_frame))
    shot_frames = list(range(curr_hit_frame, next_hit_frame + 1))
    
    preSF = trajSF.conform(pre_bounce_frames)
    postSF = trajSF.conform(post_bounce_frames)
    
    shotTraj = TrajectoryFitted(preSF, v_init_ground, v_init_vert, spin_rpm)
    bounceTraj = Trajectory(postSF)
    
    #b) bounce trajectory
    evtDict = {}
    
    for frame in bounce_frames:
        synFrm.at[frame] = bounce_point
        evtDict[frame] = 'bounce'
        
    #c) player movements
    topMvmt = PlayerMovement(ply3DSF['topPlayer_3D'].conform(shot_frames))
    botMvmt = PlayerMovement(ply3DSF['botPlayer_3D'].conform(shot_frames))
    
    myShot = Shot(shotTraj, [bounceTraj], topMvmt, botMvmt)
    
    #d) visualize
    topSF = myShot.feat_top_player()
    botSF = myShot.feat_bot_player()
    ballSF = myShot.feat_ball().recolor(const.BGR['ball_yellow'])
    
    racquetSF = myShot.feat_top_racquet().recolor(const.BGR['minty_green'])
    racquetSF2 = myShot.feat_bot_racquet().recolor(const.BGR['smokey_grey']).rename('Racquet2')
    
    showSFs = topSF.merge(botSF).merge(ballSF).merge(racquetSF).merge(racquetSF2)
    
    for frame in showSFs:
        frntScn = Scene.make_empty()
        for feat, obj in showSFs[frame].items():
            frntScn = frntScn.put(obj)
            
        frontView = View.make(myCam, frntScn, resolution = FOREGROUND_RES)
        
        showFrm = backView.combine(frontView).snap()
        
        show_continuous(showFrm.array, ms_between_frames = MS_BETWEEN_FRAMES)
        
    cv.destroyAllWindows()
    
    
    check_frame = 265.5
    traj = fitted_trajs[check_frame]
        
    plt.scatter(traj[:, 0], traj[:, 1])
    plt.scatter(planFrm.loc[:check_frame].ground_m, planFrm.loc[:check_frame].vert_m)
    
    v_ground_idx, v_vert_idx, rpm_idx = initial_conditions[274]
    
    v_init_ground = v_init_ground_space[v_ground_idx]
    v_init_vert = v_init_vert_space[v_vert_idx]
    spin_rpm = spin_rate_space[rpm_idx]
    
    Stream.from_SF(mySF, cap).show(ms_between_frames = 500, frame_counter = True)
    
    from ATMANpy.types.Abstract import Trajectory
    # instead of splines, use trajectories directly
    StreamFeatures.from_dataframe()
    
    trajSF = Trajectory.from_features()
    
    xyFrm = trajFrm[['l', 'm']]
    
    BGRFrame.from_video(cap, 255).show()
    
    
    
    # recall that splines have to start and end at the bounce point
    
    

    from scipy.interpolate import splrep, BSpline
    
    
    # xyFrm = trajFrm[['l', 'm']].sort_values('l') ### for spline construction, lambdas (our xs) have to be sorted. 
    tck, fp, ier, msg = splrep(xyFrm['l'], xyFrm['m'], full_output = True, s = 100)
    
    plt.plot(trajFrm['l'], BSpline(*tck)(trajFrm['l']), '-', label='s=0')
    
    ### use splines to simplify the functions
    
    
    audFrm.loc[255: 275]
    
    anchor_frames
    
    Stream.from_SF(mySF, cap).show(ms_between_frames = 250, frame_counter = True)
    
    ### Read in the sound, and check for a correspondence on bounce.
    audFrm.loc[frm_currhit: frm_nexthit, 'bounce']
    
    
    # def plot_trajectory(para)
    
    
    
    
    delta_lambda_per_time[0, 0]
    np.interpo
    
    paramArr[0, 0, :, 0]
    
    np.isnan(coordArr[0, 0]) | (coordArr[0, 0] > 0)
    
    
    
    
    chkArr = (coordArr > 0).all(axis = (2, 3))       
    chkArr = (coordArr > 0).mean(axis = (2, 3))
    
    
    ### visualize clouds
    mySF = cld3DSF.merge(HPCF['P'])
    cld2DSF = mySF.apply(sa.project_hit_point_cloud)
    
    frm = frm_currhit
    ptList = [Point2D(pt) for pt in cld2DSF[frm]]
    BGRFrame.from_video(cap, frm).show()
    BGRFrame.from_video(cap, frm).put(ptList).show()

    num_points, dims = ptArr_next.shape
    ### Turn coordinates homogenous
    homArr = np.concatenate([ptArr_next, np.ones((num_points, 1))], axis = 1)
    
    ### 
    homArr_2D = np.einsum('ij, nj -> ni', P, homArr)
    pixArr = homArr_2D[:, :2] / homArr_2D[:, 2][:, np.newaxis]
    
    ptList = [Point2D(pt) for pt in pixArr]
    BGRFrame.from_video(cap, frm).show()
    BGRFrame.from_video(cap, frm).put(ptList).show()

    ### visualize camlines
    courtScn = Scene.make_court()
    emptyScn = Scene.make_empty()
    
    s_x, s_y = 1920, 1080
    sensor_dims = s_x, s_y
    cam_height = 4000
    o_x = s_x / 2
    o_y = s_y / 2
    showCam = Camera.birds_eye(cam_height = cam_height, o_x = o_x, o_y = o_y, sensor_dims = sensor_dims)
    
    showCam = Camera.calibrated_AO_behind()
    
    for frame in flight_frames:
        camLn = camlnSF[frame]
        
        if not camLn is None:
            longLn = Line3D(camLn.leftPt, camLn.leftPt + camLn.to_vector() * 1000, color = const.BGR['intense_red'])
            
            myScn = courtScn.put(longLn).put(flightReg)
            myView = View.make(showCam, myScn)
            
            myView.snap().show()
    
        
    
    coordArr[0, 0]
    
    
            
    
    np.array(coordList)
    np.array(paramList)
    
    s_x, s_y = 1920, 1080
    sensor_dims = s_x, s_y
    cam_height = 12000
    o_x = s_x / 2
    o_y = s_y / 2
    
    myCamera = Camera.birds_eye(cam_height = cam_height, o_x = o_x, o_y = o_y, sensor_dims = sensor_dims)
    
    from ATMANpy.funcs.HMM.HPost import calculate_interpolation_lambda
    for frame in range(frm_currhit, frm_nexthit):
        try:
            ballPt = shapeSF[frame]
        except KeyError:
            pass
        else:
            if ballPt is not None:
                
                m_w, b_w = matchCamera.get_world_ray_params(ballPt)
                camPt, camVec = Point3D(b_w), Point3D(m_w)
                camLine = Line3D(camPt, camPt - camVec * 10000, color = const.BGR['strong_pink'])
        
                intPt = camLine.intersect(myReg)
        
                myScene = Scene.make_court().put(camLine).put(myReg)
                
                View.make(myCamera, myScene).snap().show()
        
                calculate_interpolation_lambda(m_w, b_w, pt1, pt2)
              
    
    Stream.from_SF(shapeSF, cap).show(ms_between_frames = 500)
              
    BGRFrame.from_video(cap, frame).put(ballPt).show()
                
    
    view3D(myReg)
    
    
    
    
    
    mySF = cld3DSF.merge(HPCF['P'])
    cld2DSF = mySF.apply(sa.project_hit_point_cloud)
    
    
    coords = cld2DSF[sample_frame]
    myPts = [Point2D(coord) for coord in coords]
    BGRFrame.from_video(cap, sample_frame).put(myPts).show()
    
    outArr = sa.project_hit_point_cloud(cldPt3D, P)
    
    HPCF['M_int'].feature
    
    cldPt3D = cld3DSF[sample_frame]
    ptArr = cldPt3D.rasterize()
    
    cldPt3D.project(P)
    
    myCamera
    
    
    myFrm = BGRFrame.from_video(cap, sample_frame).put([Point2D(tuple(pt)) for pt in outArr])
    myFrm.show()
    
    courtScn = Scene.make_court()
    myScene = courtScn.put(cldPt3D)
    
    myCamera = Camera.calibrated_brest_behind()
    myView = View.make(myCamera, myScene, resolution = resolution)
    myView.snap().show()
    
    np.sort(myView.pxArr)
    
    myCamera.project_points(ptArr)
    
    mySF 
    
    cldPt3D, P = mySF[sample_frame].values()
    
    myFrm = BGRFrame.from_video(cap, sample_frame).put([Point2D(tuple(pt)) for pt in cldPts2D_float])
    myFrm.show()
    
    

    

    

    
    
if __name__ == '__main__':
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    params = {}

