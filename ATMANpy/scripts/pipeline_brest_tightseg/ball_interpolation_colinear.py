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

from ATMANpy.types.Points import Point2D

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
    ff.save_feature_frame(trajFrm, 'colinearTable', vid, sto)

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
    matchCamera = Camera(M_int, M_ext, (int(width), int(height)))
    
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
    plane_point_params['serve_height_modifier'] = 2

    hit_player_ball_SF = plySF.merge(hitSF, how = 'inner').merge(HPCF['P']).merge(ballSF).merge(infoSF_fix)
    cld3DSF_hits = hit_player_ball_SF.apply(sa.get_3D_hit_plane_point, **plane_point_params)
    
    # b) add in points after the last shot, to simulate the ball flying off after last hit.
    hit_player_ball_SF = hitSF.merge(plySF).merge(HPCF['P']).merge(ballSF, how = 'outer')
    hit_player_ball_SFs = hit_player_ball_SF.conform(plyBF).filter(plyBF).split()
    hitSFs = hit_player_ball_SFs['hitFlag']

    cld3DSFs_last = hit_player_ball_SFs.apply_whole(sf.add_in_hit_point, **hit_point_params)
    cld3DSF_last = cld3DSFs_last.concatenate()
    
    cld3DSF = cld3DSF_hits.extend(cld3DSF_last['cldPt3D']).sort()
    hitSF_ext = hitSF.extend(cld3DSF_last['hitFlag']).sort()

    ### CALCULATE BOUNCE PROBS FROM POSITIONS AND BOUNCE AUDIO ###
    mySFs = hitSF_ext.merge(ply3DSF, how = 'outer').merge(audSF, how = 'outer').conform(plyBF).filter(plyBF).split()
    bounceSFs = mySFs.apply_whole(sf.combine_bounce_priors_and_probs)

    ### CHECK FOR NET PLANES
    mySFs = ballSF.merge(cld3DSF, how = 'outer').conform(plyBF).filter(plyBF).split()
    net_planeSFs = mySFs.apply_whole_mp(sf.intersect_ball_observations_and_net_plane, nwo, matchCamera = matchCamera)
    netSFs = net_planeSFs['net_coordArr'].apply_mp(sa.viterbi_net, nwo, **viterbi_net_params)

    ###### TESTING ######
    from ATMANpy.types.Frames import BGRFrame
    from ATMANpy.types.Points import Point2D
    from ATMANpy.types.Lines import Line
    
    blackFrm = BGRFrame.make_black(640, 360)

    mySFs = ballSF.merge(cld3DSF, how = 'outer').conform(plyBF).filter(plyBF).split()    
    colinear_params = {'num_angles': 1001, 'lim_pi': np.pi / 4, 'max_slope': 10 ** 6, 'w_obs': 0.8, 'w_cld': 0.2}
    coplSFs = mySFs.apply_whole_mp(sf.best_colinear_planes, nwo, matchCamera = matchCamera, **colinear_params)

    mySFs = coplSFs.merge(netSFs).merge(hitSFs).merge(bounceSFs)
    traj_params = {'observation_threshold': 3, 'number_of_local_minima': 100}
        
    nwo_fit = 4
    trajSFs = mySFs.apply_mp(sa.fit_trajectory_colinear, nwo_fit, matchCamera = matchCamera, 
                             groundArr = groundArr, vertArr = vertArr, rpmArr = rpmArr, height_space = height_space,
                             **traj_params)

    mySFs = trajSFs.merge(netSFs['netBool'])
    traj_coordSFs = mySFs.apply_mp(sa.calculate_trajectory_coordinates, nwo)

    ##### GET TRAJECTORY PROBABILITIES ###
    # a) bounce
    #### -> done in fit_trajectory_colinear

    # b) deviation
    #### -> done in fit_trajectory_colinear

    # c) is net / is not in net
    mySFs = traj_coordSFs.merge(netSFs)
    net_probSFs = mySFs.apply(sa.trajectory_net_prob, stretch_factor = viterbi_net_params['stretch_factor'])
    
    # d) prior probability over spin, given the visually detected type of shot
    ### -> done in fit_trajectory_colinear

    # e) prior probability over ball having to be in court if the rally continues after the shot in question
    # -> first, need an indication shot by shot how long the rally will still be going on after.
    remSFs = traj_coordSFs.apply_whole(sf.count_remaining_shots)
    mySFs = traj_coordSFs.merge(remSFs).merge(hitSFs).merge(srvSFs)
    fair_probSFs = mySFs.apply(sa.trajectory_fair_prob)
    
    ### CHOOSE BEST TRAJECTORY ###
    mySFs = trajSFs['trajArr'].merge(net_probSFs).merge(fair_probSFs)
    maxSFs = mySFs.apply(sa.get_highest_prob)

    emSFs = traj_coordSFs.apply(sa.make_empty_coord_and_plane_array)

    mySFs = maxSFs.merge(emSFs).merge(traj_coordSFs).merge(trajSFs)
    
    traj_arrSFs = mySFs.apply(sa.make_trajectory_arrays, 
                                v_init_ground_space = v_init_ground_space, v_init_vert_space = v_init_vert_space, 
                                spin_rate_space = spin_rate_space, height_space = height_space)

    
    # b) add in hit and serve info to construct events
    evtSFs = srvSF.merge(hitSF, how = 'outer').merge(infoSF_fix, how = 'left').conform(plyBF).filter(plyBF).split()
    
    # c) collect frames 
    traj_evtSFs = traj_arrSFs.merge(evtSFs)
    trajDFs = traj_evtSFs.apply_whole(sf.make_trajectory_frame, courtLines = courtLines)
    
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

