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

def load(vid, sto):
    HPCF = ConstFeatures.from_store(vid, sto, featnameList = ['H', 'P', 'M_int', 'M_ext'])
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally')
    ballSF = StreamFeature.from_store(vid, sto, featname = 'ball_choicePt')
    hitSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')
    plySF = StreamFeatures.from_store(vid, sto, featnameList = ['topPlayerCnt', 'botPlayerCnt'])
    audFrm = ff.read_feature_frame('sound_hitTable', vid, sto)
    return HPCF, segBF, ballSF, hitSF, plySF, audFrm

def save(trajDFs, vid, sto):
    trajFrm = pd.concat(trajDFs)
    ff.save_feature_frame(trajFrm, 'trajTable', vid, sto)

def check_planes(planeSF_raw, vid, sto, folder_name = 'planes', graph_name = 'rally', timestamp = None, dim = 10):
    save_folder = ut.get_folder_struct('graphs')['trajectories']
    sto_path = ut.maybe_make_folder(save_folder, sto)
    save_path = ut.maybe_make_folder(sto_path, vid)
    sub_save_path = ut.maybe_make_folder(save_path, folder_name)
    if timestamp is None:
        timestamp = 'no_run_specified'
    run_path = ut.maybe_make_folder(sub_save_path, timestamp)
    
    import matplotlib as mpl
    penalty_map = mpl.colormaps['Reds']

    ### establish how many random backgrounds we want to show
    grid_length = dim
    num_backgrounds = grid_length ** 2

    aspect_ratio = 4/3
    fig_height = grid_length * 5
    fig_width = fig_height * aspect_ratio
    
    ### tighten the plot
    bleed = 0.1
    gridspec_kw = {'wspace': 0.1, 'hspace': 0.1}
    
    rng = np.random.default_rng()
    
    for frame in planeSF_raw.dropna().get_frames():
        featDict = planeSF_raw[frame]
        planeArr, netArr, penArr = featDict['planeArr'], featDict['netArr'], featDict['penalties']

        len_i, len_j, len_t = planeArr.shape[:3]

        chosen_is, chosen_js = rng.choice(len_i, size = num_backgrounds), rng.choice(len_j, size = num_backgrounds)
            
        fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

        for idx, i_j in enumerate(zip(chosen_is, chosen_js)):
            i, j = i_j
            plotArr = planeArr[i, j]
            if netArr is None:
                plot_netArr = np.zeros(len_t, dtype = bool)
            else:
                plot_netArr = netArr
            penalty = penArr[i, j]

            ax = axes[idx // grid_length, idx % grid_length]
            ax.set_facecolor(penalty_map(penalty))

            flightArr_nonnan = plotArr[~plot_netArr & ~np.isnan(plotArr).all(axis = 1)]                
            netArr_nonnan = plotArr[plot_netArr & ~np.isnan(plotArr).all(axis = 1)]
            x_flight, y_flight = flightArr_nonnan[:, 0], flightArr_nonnan[:, 1]
            x_net, y_net = netArr_nonnan[:, 0], netArr_nonnan[:, 1]

            ax.scatter(x_flight, y_flight, color = 'forestgreen')
            ax.scatter(x_net, y_net, color = 'goldenrod')
    
        fig.savefig(os.path.join(run_path, f'{graph_name}_{frame}.png'))
        plt.close()

def check_trajectories(trajDF, vid, sto, folder_name = 'fitted', graph_name = 'planes', timestamp = None):
    num_shots = trajDF.shot_idx.max() + 1
    dim = int(num_shots ** 0.5) + 1

    save_folder = ut.get_folder_struct('graphs')['trajectories']
    sto_path = ut.maybe_make_folder(save_folder, sto)
    save_path = ut.maybe_make_folder(sto_path, vid)
    sub_save_path = ut.maybe_make_folder(save_path, folder_name)
    if timestamp is None:
        timestamp = 'no_run_specified'
    run_path = ut.maybe_make_folder(sub_save_path, timestamp)
    
    
    import matplotlib as mpl
    penalty_map = mpl.colormaps['Reds']

    ### establish how many random backgrounds we want to show
    aspect_ratio = 4/3
    fig_height = dim * 5
    fig_width = fig_height * aspect_ratio
    
    ### tighten the plot
    bleed = 0.1
    gridspec_kw = {'wspace': 0.1, 'hspace': 0.1}
    
    rng = np.random.default_rng()

    fig, axes = plt.subplots(dim, dim, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    # stream_traj_frame(trajDF, HPCF, cap)
    for shot_idx, shotFrm in trajDF.groupby('shot_idx'):
        ax = axes[shot_idx // dim, shot_idx % dim]
            
        flightFrm = shotFrm[shotFrm.event != 'N']
        dev = (flightFrm[['x_o', 'y_o', 'z_o']].values - flightFrm[['x_t', 'y_t', 'z_t']].values)
        mse = ((dev[~np.isnan(dev).any(axis = 1)] ** 2).sum(axis = 1) ** 0.5).mean()
        ax.set_facecolor(penalty_map(int(mse)))

        x, y = flightFrm['px_o'].values, flightFrm['pz_o'].values
        t_x, t_y = flightFrm['px_t'].values, flightFrm['pz_t'].values
        ax.scatter(x, y)
        ax.scatter(t_x, t_y)

    fig.savefig(os.path.join(run_path, f'{graph_name}.png'))
    plt.close()


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
    HPCF, segBF, ballSF, hitSF, plySF, audFrm = load(vid, sto)
    hitSF = hitSF.thin()
    
    courtLines = CourtLines.make()
    courtLines = courtLines.add_singles_netline()
    
    audSF = StreamFeatures.from_dataframe(audFrm)
    
    P = HPCF['P'].feature
    H = HPCF['H'].feature
    M_int = HPCF['M_int'].feature
    M_ext = HPCF['M_ext'].feature
    matchCamera = Camera(M_int, M_ext, (width, height))
    
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

    ### FIND SHOT POSITIONS ###
    # a) find 3D player positions
    mySF = plySF.merge(HPCF['P'])
    ply3DSF = mySF.apply(sa.find_player_positions_3D)
    
    # # b) For serve, we need to know from which side the serve happens)
    srvSF = ply3DSF.merge(hitSF, how = 'inner').apply(sa.determine_serve_position)
    
    ### CONSTRUCT ANCHOR POINTS FOR THE HYPOTHETICAL TRAJECTORY PLANES ###
    # a) make hit plane points
    hit_player_ball_SF = plySF.merge(hitSF, how = 'inner').merge(HPCF['P']).merge(ballSF)
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
    
    ### INTERSECT OBSERVED BALLS AND PLANES ###
    mySFs = ballSF.merge(cld3DSF, how = 'outer').conform(plyBF).filter(plyBF).split()
    
    planeSFs = mySFs.apply_whole_mp(sf.intersect_ball_observations_and_planes, nwo, matchCamera = matchCamera, P = P)
    
    ### CHECK FOR NET POINTS ###
    netSFs = planeSFs['net_coordArr'].apply_mp(sa.viterbi_net, nwo, **viterbi_net_params)
    
    ### PICK LOW PENALTY PLANES ###
    mySFs = planeSFs['paramArr'].merge(netSFs)
    penSFs = mySFs.apply(sa.calculate_plane_penalties, **penalty_params)
    
    planeSFs_raw = planeSFs.merge(penSFs).merge(netSFs)
    
    if VISUALIZE:
        for idx, planeSF_raw in enumerate(planeSFs_raw):
            check_planes(planeSF_raw, vid, sto, 
                         folder_name = 'planes_pre_resampling', graph_name = f'rally_{idx}', 
                         dim = 5, timestamp = timestamp)
    
    planeSFs_downsampled = planeSFs_raw.apply(sa.resample, **downsample_params)
    
    ###### FIT TRAJECTORIES ###   
    nwo_fit = 6
    total_ram = 12
    ram_allowed_in_gb = total_ram / nwo_fit
    trajSFs_fitted = planeSFs_downsampled.apply_mp(sa.fit_trajectory, nwo_fit, ram_allowed_in_gb = ram_allowed_in_gb, **traj_fit_params)
    
    ##### GET TRAJECTORY PROBABILITIES ###
    # a) bounce
    mySFs = trajSFs_fitted.merge(bounceSFs).merge(planeSFs_downsampled['netArr'])
    audio_probSFs = mySFs.apply(sa.trajectory_audio_prob)

    # b) deviation
    dev_probSFs = trajSFs_fitted.apply(sa.trajectory_deviation_prob)
    
    ###### GET TRAJECTORY ###
    ### use the pointArr to calculate a flight line, with which the fitted trajectory coordinates can be reconstructed
    traj_coordSFs = trajSFs_fitted.merge(planeSFs_downsampled['pointArr']).apply(sa.calculate_trajectory_coordinates)
    
    ####### FIND BEST VITERBI PATH ###
    probSFs = audio_probSFs.merge(dev_probSFs)
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
    evtSFs = srvSF.merge(hitSF, how = 'outer').conform(segBF).filter(segBF).split()
    
    # c) collect frames 
    traj_evtSFs = trajSFs.merge(evtSFs)
    trajDFs = traj_evtSFs.apply_whole(sf.make_trajectory_frame, courtLines = courtLines)
    
    if VISUALIZE:
        ##### SAVE PLOTS #######
        for rally_idx, trajDF in enumerate(trajDFs):
            if trajDF is None:
                continue
            else:
                check_trajectories(trajDF, vid, sto, 
                folder_name = 'fitted', graph_name = f'rally_{rally_idx}', 
                timestamp = timestamp)

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

