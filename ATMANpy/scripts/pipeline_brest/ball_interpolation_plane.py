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
traj_name = 'FullTrajectories_g60_v41_s31_h20'
groundArr, vertArr, rpmArr, trajDict = load_trajectories(traj_name, dtype = 'float32')

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
        
    if len(pens.shape) == 2:
        fs, ls = pens.shape
        
        points_re = points.reshape((fs * ls, 2, 3))
        pens_re = pens.reshape((fs * ls))
    else:
        points_re = points
        pens_re = pens
    
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

def resampled_point_cloud(trajSF, logit_coeff = 10, std_in_vx = 3):
    rng = np.random.default_rng()
    
    resampled_point_pen_dict = {}
    for hit_frame in trajSF.get_frames():
        coordArr, paramArr, planeArr, pointArr, netArr, penalties = trajSF[hit_frame].values()
        if pointArr is None and penalties is None: ### fails on the last frame -> not a hit frame, strictly, but a hypothetical flight frame
            resampled_point_pen_dict[hit_frame] = Point3D(rec_pts_resampled.T)
        else:
            ### Resample
            hit_n = coordArr.shape[0]
            arrays_resampled = resample(coordArr, paramArr, planeArr, pointArr, netArr, penalties,
                               logit_coeff = logit_coeff, sampling_number = hit_n)
            pointArr_resampled = arrays_resampled[3]
            
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

how_many_rallyes = 2
res_multiplier = 1

def main(conf, params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    width, height = get_width_height(cap)
    
    HPCF, segBF, ballSF, hitSF, plySF, audFrm = load(vid, sto)
    
    courtLines = CourtLines.make()
    courtLines = courtLines.add_singles_netline()
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
    
    #### For serve, we need to know from which side the serve happens)
    srvSF = ply3DSF.merge(hitSF, how = 'inner').apply(sa.determine_serve_position)
    
    ### make point cloud
    hit_player_ball_SF = hitSF.merge(plySF).merge(HPCF['P']).merge(ballSF)
    params = {'max_sideways_reach_in_feet': 5, 'resolution': 1}
    cld3DSF_hits = hit_player_ball_SF.apply(sa.get_3D_hit_plane_point, **params)
    
    ### add in a points cloud after the last shot, to simulate the ball flying off after last hit.
    hit_player_ball_SF = hitSF.merge(plySF).merge(HPCF['P']).merge(ballSF, how = 'outer')
    hit_player_ball_SFs = hit_player_ball_SF.conform(segBF).filter(segBF).split()
    
    params = {'height_in_feet': 5, 'resolution': 0.5, 'radius_in_feet': 15} ### originally: resolution = 0.02
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
    
    trajSFs_raw = mySFs[:how_many_rallyes].apply_whole(sf.test_trajectories, matchCamera = matchCamera, P = P)
    
    # idx = 0
    # trajSF = trajSFs_raw[idx]
    # for frame in trajSF.get_frames()[:-1]:
    #     featDict = trajSF[frame]
    #     points, penalties = featDict['pointArr'], featDict['penalties']
    #     visualize_penalties(points, penalties)
        
    # params = {'logit_coeff': 10, 'std_in_vx': 3}
    # cld3DSFs_resampled = trajSFs_raw.apply_whole(resampled_point_cloud, **params)
    # cld3DSF_resampled = cld3DSFs_resampled.concatenate()
    
    # ### Resampling like this works well when the low penalty hit points at two successive hits are close to each other
    # ### Note that when we re-launch the trajectory test in the second instance, 
    # ### the receiving points for any iteration except the last are not used 
    # ### -> instead, we use the next hit points. however, low penalties apply to hit and receive points in the previous shot.
    # ### TODO: Therefore, introduce another penalty that somehow penalises rec points which are far away from successive hit points
    
    # ### Idea for this penalty -> look at implied paths somehow 
    #     # -> get the points, and connect at previous rec point and next hit point
    #     # by calculating distance. probability of a path decreases with increasing distance (viterbi?)
    
    # mySFs = ballSF.merge(cld3DSF_resampled, how = 'outer').merge(bounceSF, how = 'outer').conform(segBF).filter(segBF).split()
    # trajSFs_resampled = mySFs[:how_many_rallyes].apply_whole(sf.test_trajectories, matchCamera = matchCamera, P = P)
        
    params = {'logit_coeff': 10, 'sampling_number': 25}
    trajSFs_downsampled = trajSFs_raw.apply(resample, **params)
    
    idx = 0
    trajSF = trajSFs_downsampled[idx]
    for frame in trajSF.get_frames()[:-1]:
        featDict = trajSF[frame]
        points, penalties = featDict['pointArr'], featDict['penalties']
        visualize_penalties(points, penalties)
        
    all_trajectory_frames = []
    for idx in range(trajSFs_downsampled.length()):
        
        trajSF = trajSFs_downsampled[idx]
        mySF = mySFs[idx]
        
        ### TODO: Factorize this function into an sf_apply thing once trajectory path penalties are introduced
        observation_threshold = 3
        trajFrms = []
        event_frames = trajSF.get_frames()
        for hit_frame in event_frames[:-1]:
            rec_frame = event_frames[event_frames.index(hit_frame) + 1]
            flight_frames = list(range(hit_frame, rec_frame + 1))
        
            coordArr, paramArr, planeArr, pointArr, netArr, penalties = trajSF[hit_frame].values()
            
            example_coords_are_nan = np.isnan(coordArr[0])
            valid_observations = (~example_coords_are_nan).all(axis = 1).sum()
            if valid_observations >= observation_threshold:
                
                ### Fit Trajectories
                # a) kill plane observations in net
                if netArr is not None:
                    planeArr = np.where(~netArr[:, :, np.newaxis], planeArr, np.nan)
                    
                # b) run fit
                trajArr, groundArr_filt, vertArr_filt, rpmArr_filt = sf.fit_trajectories_pointless(planeArr,
                                                                                                   groundArr, vertArr, rpmArr)
                
                if not netArr is None:
                    groundArr_filt = np.where(~netArr[:, np.newaxis, :], groundArr_filt, np.nan)
                    vertArr_filt = np.where(~netArr[:, np.newaxis, :], vertArr_filt, np.nan)
                    rpmArr_filt = np.where(~netArr[:, np.newaxis, :], rpmArr_filt, np.nan)
                
                ### Compare Trajectories with audio
                # a) gather bounce frame   
                bncSF = mySF[['bounce_prob_cond1', 'bounce_prob_cond2', 'prior0', 'prior1', 'prior2']]
                bounceFrm = bncSF[flight_frames].to_dataframe().iloc[1:] ### first frame can not have bounce (after hit)
                
                # b) overide priors if ball is likely in net
                if not netArr is None: ### improve priors if the ball is likely to be in the net
                    p0_fly, p1_fly, p2_fly = bounceFrm['prior0'].unique()[0], bounceFrm['prior1'].unique()[0], bounceFrm['prior2'].unique()[0]
                    
                    planes_in_net = np.max(netArr, axis = 1)
                    prob_in_net = planes_in_net.mean()
                    p0_in_net, p1_in_net, p2_in_net = 0.9, 0.09, 0.01
                    p0 = (1 - prob_in_net) * p0_fly + prob_in_net * p0_in_net
                    p1 = (1 - prob_in_net) * p1_fly + prob_in_net * p1_in_net
                    p2 = (1 - prob_in_net) * p2_fly + prob_in_net * p2_in_net
                    
                    prior_override = p0, p1, p2
                else:
                    prior_override = None
                    
                bounce_probs = sf.compare_trajectories_with_audio(vertArr_filt, bounceFrm, prior_override = prior_override)
                
                ### Compare Trajectories with observed path
                plane_traj_probs = 1 / (trajArr[:, :, 4] + 1)
                
                ### Join Probs
                probs = bounce_probs * plane_traj_probs
                plane_traj_is_best = probs == probs.max()
                best_plane, best_traj = ut.argmax_multidim(plane_traj_is_best)
                
                ### Make Trajectory Frame
                trajFrm = sf.make_trajectory_frame(plane_traj_is_best, 
                                                   coordArr, paramArr, planeArr, pointArr,
                                                   groundArr_filt, vertArr_filt, rpmArr_filt,
                                                   v_init_ground_space, v_init_vert_space, spin_rate_space,
                                                   trajArr,
                                                   flight_frames = flight_frames)
                
                ### Add Events
                hit_event = hitSF.maybe_get(hit_frame)
                srv_event = srvSF.maybe_get(hit_frame)
                side = hit_event.split('_')[0]
                
                if srv_event is not None:
                    srv_side = srv_event[2]
                else:
                    srv_side = None
                
                trajFrm['is_net'] = False if netArr is None else netArr[best_plane]
                trajFrm['is_bounce'] = np.where(trajFrm.z_t == 0, True, False)
                
                if srv_side is None:
                    inReg = courtLines.get_half_region('Bottom' if side == 'top' else 'Top', width = 'Singles')
                else:
                    inReg = courtLines.get_service_box('Bottom' if side == 'top' else 'Top', which = 'Deuce' if srv_side == 'D' else 'Ad')
                    
                bounce_coords = trajFrm.loc[trajFrm['is_bounce'], ['x_t', 'y_t']]
                
                if not bounce_coords.empty:
                    def coords_in_region(coords):
                        return Point2D(coords).within(inReg)
                
                    trajFrm['is_bounce_in'] = bounce_coords.apply(coords_in_region, axis = 1, raw = True)
                else:
                    trajFrm['is_bounce_in'] = np.nan
                
                trajFrm['event'] = np.where(trajFrm.is_bounce, 
                                            np.where(trajFrm.is_bounce_in, 'BI', 'BO'),
                                            np.where(trajFrm.is_net, 'N', 'F'))
                
                if srv_event is not None:
                    trajFrm.loc[hit_frame, 'event'] = srv_event
                else:
                    trajFrm.loc[hit_frame, 'event'] = 'H' + hit_event[0].capitalize()
                
                trajFrm = trajFrm.drop(columns = ['is_net', 'is_bounce', 'is_bounce_in'])
                
                ### Visualize
                x, y = trajFrm['px_o'].values, trajFrm['pz_o'].values
                t_x, t_y = trajFrm['px_t'].values, trajFrm['pz_t'].values
                plt.scatter(x, y)
                plt.scatter(t_x, t_y)
                plt.show()
            else:
                trajFrm = pd.DataFrame()
                
            trajFrms.append(trajFrm)
    
    
        # trajFrm = trajFrms[-5]
        # ### Visualize
        # x, y = trajFrm['px_o'].values, trajFrm['pz_o'].values
        # t_x, t_y = trajFrm['px_t'].values, trajFrm['pz_t'].values
        # plt.scatter(x, y)
        # plt.scatter(t_x, t_y)
        # plt.show()
        
        
        
    
        # trajFrm = pd.concat([trajFrm.iloc[:-1] for trajFrm in trajFrms])
        # all_trajectory_frames.append(trajFrm)
    
        
        ### Visualize on the actual Stream
        showFrm = trajFrm[['x_t', 'y_t', 'z_t']]
        showFrm['x_t'] = showFrm['x_t'].fillna(trajFrm['x_o']).interpolate() ### filling and interpolating only needed for net observations
        showFrm['y_t'] = showFrm['y_t'].fillna(trajFrm['y_o']).interpolate()
        showFrm['z_t'] = showFrm['z_t'].fillna(trajFrm['z_o']).interpolate()
        
        showFrm.columns = ['x', 'y', 'z']
        ball3DSF = StreamFeatures.from_dataframe(showFrm).apply(sa.coords_to_ball).rename('ball3D')
        ball3DSF = ball3DSF.merge(HPCF['P'])
        ball3DSF = ball3DSF.apply(sa.project_ball)
        
        Stream.from_SF(ball3DSF, cap).show(ms_between_frames = 100, frame_counter = True)
        
        Stream(cap, 250, 280).show(ms_between_frames = 500, frame_counter = True)
        
        
        check(Point3D(pointArr[:, 1, :].T))
    
    
    
    
    
    
    
    
    for frm in all_trajectory_frames:
        ff.save_feature_frame(frm, 'trajTable', vid, sto)
    

    
    
if __name__ == '__main__':
    
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    params = {}

