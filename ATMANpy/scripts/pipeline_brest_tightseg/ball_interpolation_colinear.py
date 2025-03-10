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
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Visual3D import Scene, Camera, View
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D

from ATMANpy.scripts.pipeline_brest_tightseg.ball_interpolation_plane import check_planes, check_trajectories

def load(vid, sto):
    HPCF = ConstFeatures.from_store(vid, sto, featnameList = ['H', 'P', 'M_int', 'M_ext'])
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
    ballSF = StreamFeature.from_store(vid, sto, featname = 'ball_choicePt')
    hitSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)
    infoSF = StreamFeatures.from_store(vid, sto, featnameList = ['top_name_Flag', 'top_hand_Flag', 'bot_name_Flag', 'bot_hand_Flag'])
    infoSF_tb = StreamFeatures.from_store(vid, sto, featnameList = ['top_name_tb_Flag', 'top_hand_tb_Flag', 'bot_name_tb_Flag', 'bot_hand_tb_Flag'])
    return HPCF, segBF, ballSF, hitSF, plySF, audFrm, infoSF, infoSF_tb

def save(trajDFs, vid, sto):
    trajFrm = pd.concat(trajDFs)
    ff.save_feature_frame(trajFrm, 'trajTable', vid, sto)

def main(conf, 
         trajectory_params,
         plane_point_params, hit_point_params, 
         downsample_params, viterbi_net_params, penalty_params, 
         viterbi_traj_params,
         name = None):
    
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    width, height = get_width_height(cap)
        
    VISUALIZE = True
    timestamp = ut.make_timestamp() 

    #### READ IN ####
    HPCF, segBF, ballSF, hitSF, plySF, audFrm, infoSF, infoSF_tb = load(vid, sto)
    hitSF = hitSF.thin()

    ### FIX NAMES AND HANDEDNESS IN TIEBREAKS    
    if infoSF_tb.is_empty():
        infoSF_fix = infoSF
    else:
        replSF = infoSF_tb
        for feature_name in replSF.get_features():
            replSF = replSF.rename_feature(feature_name, feature_name.replace('_tb', ''))
        infoSF_fix = infoSF.replace(replSF)

    courtLines = CourtLines.make()
    courtLines = courtLines.add_singles_netline()
    
    audSF = StreamFeatures.from_dataframe(audFrm)
    
    P = HPCF['P'].feature
    H = HPCF['H'].feature
    M_int = HPCF['M_int'].feature
    M_ext = HPCF['M_ext'].feature
    matchCamera = Camera(M_int, M_ext, (width, height))
    
    ### MAKE A HAS_PLAYER FILTER -> we need players throughout the entire exchange for this
    ### TODO: turn this into a separate step, earlier in the pipeline (right after player tracking / in player tracking)
    plySFs = plySF.conform(segBF).filter(segBF).split()
    
    players_are_there_dict = {f: False for f in segBF.get_frames()}
    for pSF in plySFs:
        top_all_there = pSF['topPlayerCnt'].dropna().length() == pSF.length()
        bot_all_there = pSF['botPlayerCnt'].dropna().length() == pSF.length()
        if top_all_there and bot_all_there:
            for f in pSF.get_frames():
                players_are_there_dict[f] = True

    plyBF = BoolFeature(players_are_there_dict, name = 'hasPlayerAlways')
    
    plySF = plySF.conform(plyBF).filter(plyBF)

    ##### TRAJECTORIES ####
    from ATMANpy.models.BallTrajectory.flight_bounce_model import load_trajectories, initialize
    groundArr, vertArr, rpmArr, trajDict = load_trajectories(trajectory_params['traj_name'], dtype = 'float32')
    
    ### initialise search spaces with proper arguments
    import inspect
    kwargList = inspect.getfullargspec(initialize).args
    kwargs = {k: v for k, v in trajDict.items() if k in kwargList}
    v_init_ground_space, v_init_vert_space, spin_rate_space, height_space = initialize(dtype = 'float32', 
                                                                                       squeeze = True,  
                                                                                       **kwargs)
    
    traj_fit_params = {p: v for p, v in trajectory_params.items() if p != 'traj_name'}
    traj_fit_params['groundArr'] = groundArr
    traj_fit_params['vertArr'] = vertArr
    traj_fit_params['rpmArr'] = rpmArr
    traj_fit_params['height_space'] = height_space ### this is needed to segment the trajectory fitting into serves and overheads (hit high) and groundstrokes (hit low)

    ### FIND SHOT POSITIONS ###
    # a) find 3D player positions
    mySF = plySF.merge(HPCF['P'])
    ply3DSF = mySF.apply(sa.find_player_positions_3D)
    
    # # b) For serve, we need to know from which side the serve happens)
    srvSF = ply3DSF.merge(hitSF, how = 'inner').apply(sa.determine_serve_position)
    srvSFs = srvSF.conform(plyBF).filter(plyBF).split()
    
    ### CONSTRUCT ANCHOR POINTS FOR THE HYPOTHETICAL TRAJECTORY PLANES ###
    # a) make hit plane points
    hit_player_ball_SF = plySF.merge(hitSF, how = 'inner').merge(HPCF['P']).merge(ballSF).merge(infoSF_fix)
    cld3DSF_hits = hit_player_ball_SF.apply(sa.get_3D_hit_plane_point, **plane_point_params)
    
    # b) add in points after the last shot, to simulate the ball flying off after last hit.
    hit_player_ball_SF = hitSF.merge(plySF).merge(HPCF['P']).merge(ballSF, how = 'outer')
    hit_player_ball_SFs = hit_player_ball_SF.conform(plyBF).filter(plyBF).split()
    
    cld3DSFs_last = hit_player_ball_SFs.apply_whole(sf.add_in_hit_point, **hit_point_params)
    cld3DSF_last = cld3DSFs_last.concatenate()
    
    cld3DSF = cld3DSF_hits.extend(cld3DSF_last['cldPt3D']).sort()
    hitSF_ext = hitSF.extend(cld3DSF_last['hitFlag']).sort()

    ### CALCULATE BOUNCE PROBS FROM POSITIONS AND BOUNCE AUDIO ###
    mySFs = hitSF_ext.merge(ply3DSF, how = 'outer').merge(audSF, how = 'outer').conform(plyBF).filter(plyBF).split()
    bounceSFs = mySFs.apply_whole(sf.combine_bounce_priors_and_probs)
    



    ### check the equation for the colinear lines
    # (Red Notebook, 09.03.2025)
    d = 500
    h = 50
    num_angles = 1001
    fs_pi, ls_pi = -np.pi / 4, np.pi / 4
    max_slope = 10 ** 6

    angles = np.linspace(fs_pi, ls_pi, num = num_angles)
    
    cx, cy, cz = matchCamera.get_cw()
    M_proj = matchCamera.get_P()
    

    d_sin_angles, d_cos_angles = d * np.sin(angles), d* np.cos(angles)
    
    ang_vecs = np.array([d_sin_angles, -d_cos_angles, np.zeros(num_angles), np.zeros(num_angles)]).T # shape: angles x 4

    ### get the A, B, C --> these are dependent on theta, but not on the height
    A0, B0, C0 = np.einsum('ij, nj -> in', M_proj, ang_vecs)
    A1, B1, C1 = A0 + M_proj[0, 2], B0 + M_proj[1, 2], C0 + M_proj[2, 2]

    ### dehomogenize
    ux, uy = A0 / C0, B0 / C0
    vx, vy = A1 / C1, B1 / C1
    u, v = np.array([ux, uy]).T, np.array([vx, vy]).T

    ### get the slope of the colinear projected line
    # -> points flying colinearly with the camera will be projected close to one of these lines.
    m = (vy - uy) / (vx - ux) ### these are needed for minimising distance of observed points
                                # if vx - ux = 0, then there is an easier formula for minimal distance
    # correct extreme slopes (vx - ux almost 0) to avoid numerical issues
    m = np.where(np.abs(m) > max_slope, np.nan, m)

    ### calculate normal vector coordinates
    diff_vecs = v - u ### shape: angles x 2
    dx, dy = diff_vecs[:, 0], diff_vecs[:, 1]
    nx, ny = dy, -dx
    len_n = (nx ** 2 + ny ** 2) ** 0.5
    nx_unit, ny_unit = nx / len_n, ny / len_n

    ### some broadcasting for later. shape: angles, obs
    nx_unit_brdc = nx_unit[:, np.newaxis] 
    ny_unit_brdc = ny_unit[:, np.newaxis]
    m_brdc = m[:, np.newaxis]
    ux_brdc = ux[:, np.newaxis]
    uy_brdc = uy[:, np.newaxis]

    ### minimize distance to observed balls
    mySFs = ballSF.merge(cld3DSF, how = 'outer').conform(plyBF).filter(plyBF).split()
    
    mySF = mySFs[110]
    anchorSF = mySF['cldPt3D']
    shapeSF = mySF['ball_choicePt']

    anchor_frames = anchorSF.dropna().get_frames()

    idx = 2
    for idx in range(len(anchor_frames) - 1):
        hit_idx_is_last = idx == len(anchor_frames) - 2
    
        frm_currhit, frm_nexthit = anchor_frames[idx], anchor_frames[idx + 1]
        anchorPts_curr, anchorPts_next = anchorSF[frm_currhit], anchorSF[frm_nexthit]

        flight_frames = list(range(frm_currhit, frm_nexthit + 1))
        observed_points = list(shapeSF[flight_frames].dropna().values())
        obsArr = np.array([obsPt.coords for obsPt in observed_points]) ### shape obs, 2
        px, py = obsArr[:, 0], obsArr[:, 1]
        
        px_brdc = px[np.newaxis, :] # shape: angles, obs
        py_brdc = py[np.newaxis, :] # shape: angles, obs

        ### optimal gamma -> usual formula where ux != vx
        opt_gamma_numer = m_brdc * (px_brdc - ux_brdc) - (py_brdc - uy_brdc)
        opt_gamma_denom = ny_unit_brdc - m_brdc * nx_unit_brdc
        opt_gamma_usual = opt_gamma_numer / opt_gamma_denom

        ### optimal gamma -> fallback formula where ux = vx (and therefore m is dangerously large in absolute terms)
        opt_gamma_fallback = (px_brdc - ux_brdc) / nx_unit_brdc

        ### join the two formulae
        opt_gamma = np.where(np.isnan(opt_gamma_usual), opt_gamma_fallback, opt_gamma_usual)
        
        ### the distance of every observation to every line is the absolute value of the gamma (the parameter along the normalized normal vector)
        opt_dist = np.abs(opt_gamma)

        ### for every set of angles, we calculate the mean and variance of distances of the projected points to a theoretical colinear projected line
        opt_dist_avg = opt_dist.mean(axis = 1)
        opt_dist_std = opt_dist.std(axis = 1)

        ### find the best mean distance, best variance of distances, and the corresponding best angle
        best_angle_idx = np.argmin(opt_dist_avg)
        best_dist_avg = opt_dist_avg[best_angle_idx]
        best_dist_std = opt_dist_std[best_angle_idx]
        
        ### TODO: use also the hitting points (cldpt3DSF, projected) as indicators of co-linearity!
        # jointly with the best co-linear plane, establish which (if any) start and end points are actually sensible
            # -> they must also be projected onto the same co-linear projected line

        ### from the best angle we immediately get a flight plane!
        # this is the flight plane given that the ball flies co-linear with the camera
        # in this flight plane, we can not locate the ball by intersecting plane and camera line through observation (-> they're identical)
        # instead, we use the height of the ball on the line (which can easily be calculated with the other optimization parameter, lambda)
        # and then find a trajectory consistent with these projected heights.
        # also consistent with bounce info, etc., of course.

        angles[best_angle_idx]

    ang = angles[4]
    c_w = np.array([cx, cy, cz, 1])
    x_w = np.array([d * np.sin(ang), -d * np.cos(ang), -cz, 0])
    d_w = np.array([0, 0, h, 0])

    u_hom = M_proj @ x_w
    v_hom = u_hom + M_proj @ d_w
    
    u_hom = M_proj @ (c_w + x_w)
    v_hom = u_hom + M_proj @ (c_w + d_w)
    
    u, v = u_hom[:2] / u_hom[2], v_hom[:2] / v_hom[2]

    from ATMANpy.types.Frames import BGRFrame
    from ATMANpy.types.Points import Point2D
    from ATMANpy.types.Lines import Line

    uPt, vPt = Point2D(tuple(u)), Point2D(tuple(v))
    coLn = Line(uPt, vPt)

    blackFrm = BGRFrame.make_black(640, 360)
    blackFrm.put(coLn).show()


    ######### TESTING #########
    pnt = 110
    frm = 82514
    
    ball = ballSF[frm]
    cldPt_vec = cld3DSF[frm]

    cldPt_vec.coords
    cldPt = cldPt_vec.devectorize()

    sa.make_camline(cldPt[0], myCamera = matchCamera)

    hit_player_ball_SFs[pnt]['ball_choicePt'].streamDict


    ### find cam line from observed balls
    mySFs = ballSF.merge(cld3DSF, how = 'outer').conform(plyBF).filter(plyBF).split()
    
    mySF = mySFs[pnt]

    



    ### INTERSECT OBSERVED BALLS AND PLANES ###
    mySFs = ballSF.merge(cld3DSF, how = 'outer').conform(plyBF).filter(plyBF).split()
    
    planeSFs = mySFs.apply_whole_mp(sf.intersect_ball_observations_and_planes, nwo, matchCamera = matchCamera, P = P)
    
    ### CHECK FOR NET POINTS ###
    netSFs = planeSFs['net_coordArr'].apply_mp(sa.viterbi_net, nwo, **viterbi_net_params)

    ### FIGURE OUT EXACT NET IMPACT TIMING FOR EACH PLANE, AND NULL EVERYTHING POST NET ###
    mySFs = planeSFs.merge(netSFs)
    planeSFs_netted = mySFs.apply(sa.null_post_net_values)
    planeSFs_netted = planeSFs_netted.merge(planeSFs['pointArr'])

    ### TODO: 
        # -> netArr can be turned into a single boolean, because we have a new, better filter (last pre net index)
        # -> add last pre net index to sf_apply functions
        # -> apply the function to coordArr, paramArr, planeArr, pointArr by culling all indices after the last pre net index
            # (essentially, this is kind of like resample in form)
        # -> plane penalties do not need netArr anymore, but now, in the net case we have planes of heterogenous length in penalty calculation: problem?
        # -> netArr should not be needed in fit trajectory -> what is that condition at the end?
        # -> trajectory audio prob: netArr simply becomes a Boolean.
        # -> trajectory net prob: netArr simply becomes a Boolean.
        # -> for trajectory array, we might want to save the exact index (i.e., frame) at which the ball is in net, also!!!

        # -> PENALISE NON-NET HITTING TRAJECTORIES (NEVER PASS NET) IF NET_BOOL = TRUE.

    ### since it is unclear when the net is hit ...
    ### (it can be projected onto the net while still in flight, esp. when coming from below!)
    ### we simply limit every trajectory to flight points that are before the net line.
    ### we do this by extrapolating the flight path and keeping elements we know are before the net
    ### (this exploits us knowing that the ball starts out in flight).
    ### once the extrapolated ball crosses the net, we cull everything else.

    ### PICK LOW PENALTY PLANES ###
    remSFs = planeSFs_netted.apply_whole(sf.count_remaining_shots)
    mySFs = planeSFs_netted.merge(netSFs).merge(remSFs)
    penSFs = mySFs.apply(sa.calculate_plane_penalties_unvectorized, **penalty_params)
    
    planeSFs_raw = planeSFs_netted.merge(penSFs)
    
    # if VISUALIZE:
    #     for idx, planeSF_raw in enumerate(planeSFs_raw):
    #         check_planes(planeSF_raw, vid, sto, 
    #                      folder_name = 'planes_pre_resampling', graph_name = f'rally_{idx}', 
    #                      dim = 5, timestamp = timestamp)
    
    planeSFs_downsampled = planeSFs_raw.apply(sa.resample, **downsample_params)
    
    ### MERGE IN HIT FLAGS
    hitSFs = hitSF.conform(plyBF).filter(plyBF).split().dropna()
    planeSFs_downsampled = planeSFs_downsampled.merge(hitSFs)

    ###### FIT TRAJECTORIES ###   
    nwo_fit = 4
    total_ram = 12
    ram_allowed_in_gb = total_ram / nwo_fit
    trajSFs_fitted = planeSFs_downsampled.apply_mp(sa.fit_trajectory, nwo_fit, ram_allowed_in_gb = ram_allowed_in_gb, **traj_fit_params)

    ###### GET TRAJECTORY COORDINATES ###
    ### use the pointArr to calculate a flight line, with which the fitted trajectory coordinates can be reconstructed
    mySFs = trajSFs_fitted.merge(netSFs).merge(planeSFs_downsampled['pointArr'])
    traj_coordSFs = mySFs.apply(sa.calculate_trajectory_coordinates)

    ##### GET TRAJECTORY PROBABILITIES ###
    # a) bounce
    mySFs = traj_coordSFs.merge(bounceSFs).merge(netSFs)
    audio_probSFs = mySFs.apply(sa.trajectory_audio_prob)

    # b) deviation
    dev_probSFs = trajSFs_fitted.apply(sa.trajectory_deviation_prob)

    # c) is net / is not in net
    mySFs = traj_coordSFs.merge(netSFs)
    net_probSFs = mySFs.apply(sa.trajectory_net_prob, stretch_factor = viterbi_net_params['stretch_factor'])
    
    # d) prior probability over spin, given the visually detected type of shot
    mySFs = trajSFs_fitted.merge(hitSFs)
    spin_probSFs = mySFs.apply(sa.trajectory_spin_prob)

    # e) prior probability over ball having to be in court if the rally continues after the shot in question
    # -> first, need an indication shot by shot how long the rally will still be going on after.
    remSFs = traj_coordSFs.apply_whole(sf.count_remaining_shots)
    mySFs = traj_coordSFs.merge(remSFs).merge(hitSFs).merge(srvSFs)
    fair_probSFs = mySFs.apply(sa.trajectory_fair_prob)
    
    ####### FIND BEST VITERBI PATH ###
    probSFs = audio_probSFs.merge(dev_probSFs).merge(net_probSFs).merge(spin_probSFs).merge(fair_probSFs)
    prob_coordSFs = probSFs.merge(traj_coordSFs)

    vitSFs = prob_coordSFs.apply_whole(sf.viterbi_trajectories, **viterbi_traj_params)
    
    ###### CONSTRUCT TRAJECTORY FRAMES ####
    # a) make arrays
    combinedSFs = trajSFs_fitted.merge(vitSFs).merge(planeSFs_downsampled).merge(traj_coordSFs)
    trajSFs = combinedSFs.apply(sa.make_trajectory_arrays, 
                                v_init_ground_space = v_init_ground_space,
                                v_init_vert_space = v_init_vert_space,
                                spin_rate_space = spin_rate_space,
                                height_space = height_space)
    
    # b) add in hit and serve info to construct events
    evtSFs = srvSF.merge(hitSF, how = 'outer').merge(infoSF_fix, how = 'left').conform(plyBF).filter(plyBF).split()
    
    # c) collect frames 
    traj_evtSFs = trajSFs.merge(evtSFs)
    trajDFs = traj_evtSFs.apply_whole(sf.make_trajectory_frame, courtLines = courtLines)
    
    if VISUALIZE:
        ##### SAVE PLOTS #######
        for rally_idx, trajDF in enumerate(trajDFs):
            if trajDF is None:
                continue
            else:
                try:
                    check_trajectories(trajDF, vid, sto, 
                    folder_name = 'fitted', graph_name = f'rally_{rally_idx}', 
                    timestamp = timestamp)
                except:
                    print(f'Could not visualize {rally_idx}')

    # ### Show on Stream!
    # idx = 5
    # myDF = trajDFs[idx]
    # coordDF = myDF[['x_t', 'y_t', 'z_t']]
    # coordDF.columns = ['x', 'y', 'z']

    # myBF = plyBF.filter(plyBF).split()[idx]

    # b2DSF = StreamFeatures.from_dataframe(coordDF).apply(sa.coords_to_ball3D).merge(HPCF).apply(sa.project_ball)
    
    # Stream.from_SF(ballSF.conform(myBF), cap).show(ms_between_frames = 100)

    # Stream.from_SF(b2DSF.conform(myBF), cap).show(ms_between_frames = 100)

    ### SAVE TRAJECTORY TABLES ###
    save(trajDFs, vid, sto)
    
if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[0])

    trajectory_params = prm_mod.ball_interpolation_plane.trajectory_params
    plane_point_params = prm_mod.ball_interpolation_plane.plane_point_params    
    hit_point_params = prm_mod.ball_interpolation_plane.hit_point_params
    downsample_params = prm_mod.ball_interpolation_plane.downsample_params
    viterbi_net_params = prm_mod.ball_interpolation_plane.viterbi_net_params
    penalty_params = prm_mod.ball_interpolation_plane.penalty_params
    viterbi_traj_params = prm_mod.ball_interpolation_plane.viterbi_traj_params

    name = ppl._name

    main(conf, trajectory_params, 
         plane_point_params, hit_point_params, 
         downsample_params, viterbi_net_params, penalty_params, 
         viterbi_traj_params, name = name)

