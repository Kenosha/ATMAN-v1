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


#%% VISUALIZATION FUNCTIONS

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
    
    s_x, s_y = 1920, 1080
    sensor_dims = s_x, s_y
    cam_height = 4000
    o_x = s_x / 2
    o_y = s_y / 2
    
    myCamera = Camera.birds_eye(cam_height = cam_height, o_x = o_x, o_y = o_y, sensor_dims = sensor_dims)
    myView = View.make(myCamera, myScn, resolution = resolution)
    myView.snap().show()
        
#### visualize planes
def visualize_planes(planeArr, dim = 10, name = 'planes', timestamp = None):
    save_folder = ut.get_folder_struct('graphs')['trajectories']
    save_path = ut.maybe_make_folder(save_folder, 'planes_flat')
    if timestamp is None:
        timestamp = 'no_run_specified'
    run_path = ut.maybe_make_folder(save_path, timestamp)
     
     
    num_curr_points, num_next_points = planeArr.shape[:2]
    
    len_dim1 = num_curr_points // dim
    len_dim2 = num_next_points // dim
     
    for g1 in range(len_dim1):
        g1_fs, g1_ls = g1 * dim, (g1 + 1) * dim
        for g2 in range(len_dim2):
            g2_fs, g2_ls = g2 * dim, (g2 + 1) * dim
           
            plotArr = planeArr[g1_fs:g1_ls, g2_fs:g2_ls]
           
            fig, ax = plt.subplots(10, 10, figsize = (50, 50))
            for idx_1, curr_plotArr in enumerate(plotArr):
                for idx_2, curr_next_plotArr in enumerate(curr_plotArr):
                    x, z = curr_next_plotArr[:, 0], curr_next_plotArr[:, 1]
                    ax[idx_1, idx_2].scatter(x, z)
                   
            fig.savefig(os.path.join(run_path, f'{name}_x{g1}_y{g2}.png'))
            plt.close()
 
#### visualize planes
def visualize_planes_flat(planeArr, dim = 10, name = 'planes', timestamp = None):
    save_folder = ut.get_folder_struct('graphs')['trajectories']
    save_path = ut.maybe_make_folder(save_folder, 'planes_flat')
    if timestamp is None:
        timestamp = 'no_run_specified'
    run_path = ut.maybe_make_folder(save_path, timestamp)
     
    num_traj = planeArr.shape[0]
    
    fig, ax = plt.subplots(dim, dim, figsize = (50, 50))
    plotArr = planeArr
    for idx, curr_plotArr in enumerate(plotArr):
        x_idx, y_idx = idx // dim, idx % dim
        x, z = curr_plotArr[:, 0], curr_plotArr[:, 1]
        ax[x_idx, y_idx].scatter(x, z)
   
    fig.savefig(os.path.join(run_path, f'{name}.png'))
    plt.close()

def plot_traj_frame(trajFrm, timestamp = None, subfolder_name = None):
    save_folder = ut.get_folder_struct('graphs')['trajectories']
    save_path = ut.maybe_make_folder(save_folder, 'traj_frames')
    if timestamp is None:
        timestamp = 'no_run_specified'
    run_path = ut.maybe_make_folder(save_path, timestamp)
    if subfolder_name is None:
        fld_path = run_path
    else:
        fld_path = ut.maybe_make_folder(run_path, subfolder_name)
    
    fs, ls = trajFrm.index[0], trajFrm.index[-1]
    name = f'hit_{fs}'
    
    ### Visualize
    fig, ax = plt.subplots(1, 1, figsize = (7, 7))
    
    flightFrm = trajFrm[trajFrm.event != 'N']
    
    x, y = flightFrm['px_o'].values, flightFrm['pz_o'].values
    t_x, t_y = flightFrm['px_t'].values, flightFrm['pz_t'].values
    ax.scatter(x, y)
    ax.scatter(t_x, t_y)
    fig.savefig(os.path.join(fld_path, f'{name}.png'))
    plt.close()
    
def stream_traj_frame(trajFrm, HPCF, cap, ms_between_frames = 100, frame_counter = True):

    ### Visualize on the actual Stream
    showFrm = trajFrm[['x_t', 'y_t', 'z_t']]
    showFrm['x_t'] = showFrm['x_t'].fillna(trajFrm['x_o']).interpolate() ### filling and interpolating only needed for net observations
    showFrm['y_t'] = showFrm['y_t'].fillna(trajFrm['y_o']).interpolate()
    showFrm['z_t'] = showFrm['z_t'].fillna(trajFrm['z_o']).interpolate()
    
    showFrm.columns = ['x', 'y', 'z']
    ball3DSF = StreamFeatures.from_dataframe(showFrm).apply(sa.coords_to_ball).rename('ball3D')
    ball3DSF = ball3DSF.merge(HPCF['P'])
    ball3DSF = ball3DSF.apply(sa.project_ball)
    
    Stream.from_SF(ball3DSF, cap).show(ms_between_frames = ms_between_frames, frame_counter = frame_counter)
    
#%% MAIN

VISUALIZE = True
timestamp = ut.make_timestamp() 

def main(conf, 
         plane_point_params, hit_point_params, 
         downsample_params, viterbi_net_params, penalty_params, 
         traj_fit_params, viterbi_traj_params):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = get_video(vid)
    width, height = get_width_height(cap)
    
    #### READ IN ####
    HPCF, segBF, ballSF, hitSF, plySF, audFrm = load(vid, sto)
    
    courtLines = CourtLines.make()
    courtLines = courtLines.add_singles_netline()
    
    audSF = StreamFeatures.from_dataframe(audFrm)
    
    P = HPCF['P'].feature
    H = HPCF['H'].feature
    M_int = HPCF['M_int'].feature
    M_ext = HPCF['M_ext'].feature
    matchCamera = Camera(M_int, M_ext, (width, height))
    
    ### FIND SHOT POSITIONS ###
    # a) find 3D player positions
    mySF = plySF.merge(HPCF['P'])
    ply3DSF = mySF.apply(sa.find_player_positions_3D)
    
    # # b) For serve, we need to know from which side the serve happens)
    srvSF = ply3DSF.merge(hitSF, how = 'inner').apply(sa.determine_serve_position)
    
    ### CONSTRUCT ANCHOR POINTS FOR THE HYPOTHETICAL TRAJECTORY PLANES ###
    # a) make hit plane points
    hit_player_ball_SF = hitSF.merge(plySF).merge(HPCF['P']).merge(ballSF)
    cld3DSF_hits = hit_player_ball_SF.apply(sa.get_3D_hit_plane_point, **plane_point_params)
    
    # b) add in points after the last shot, to simulate the ball flying off after last hit.
    hit_player_ball_SF = hitSF.merge(plySF).merge(HPCF['P']).merge(ballSF, how = 'outer')
    hit_player_ball_SFs = hit_player_ball_SF.conform(segBF).filter(segBF).split()
    
    cld3DSFs_last = hit_player_ball_SFs.apply_whole(sf.add_in_hit_point, **hit_point_params)
    cld3DSF_last = cld3DSFs_last.concatenate()
    
    cld3DSF = cld3DSF_hits.extend(cld3DSF_last['cldPt3D']).sort()
    hitSF_ext = hitSF.extend(cld3DSF_last['hitFlag']).sort()

    ### CALCULATE BOUNCE PROBS FROM POSITIONS AND BOUNCE AUDIO ###
    mySFs = hitSF_ext.merge(ply3DSF, how = 'outer').merge(audSF, how = 'outer').conform(segBF).filter(segBF).split()
    bounceSFs = mySFs.apply_whole(sf.combine_bounce_priors_and_probs)
    
    ### INTERSECT OBSERVED BALLS AND PLANES ###
    mySFs = ballSF.merge(cld3DSF, how = 'outer').conform(segBF).filter(segBF).split()
    
    planeSFs = mySFs.apply_whole_mp(sf.intersect_ball_observations_and_planes, nwo, matchCamera = matchCamera, P = P)
    
    ### CHECK FOR NET POINTS ###
    netSFs = planeSFs['net_coordArr'].apply_mp(sa.viterbi_net, nwo, **viterbi_net_params)
    
    ### PICK LOW PENALTY PLANES ###
    mySFs = planeSFs['paramArr'].merge(netSFs)
    penSFs = mySFs.apply(sa.calculate_plane_penalties, **penalty_params)
    
    planeSFs_raw = planeSFs.merge(penSFs).merge(netSFs)
    planeSFs_downsampled = planeSFs_raw.apply(sa.resample, **downsample_params)
    
    ###### FIT TRAJECTORIES ###   
    nwo_fit = 12
    total_ram = 18
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
    
    ##### SAVE PLOTS #######
    for rally_idx, trajDF in enumerate(trajDFs):
        if trajDF is None:
            continue
        # stream_traj_frame(trajDF, HPCF, cap)
        for shot_idx, shotFrm in trajDF.groupby('shot_idx'):
            plot_traj_frame(shotFrm, timestamp = timestamp, subfolder_name = f'rally{rally_idx}')
            
    ### SAVE TRAJECTORY TABLES ###
    save(trajDFs, vid, sto)
    
if __name__ == '__main__':
        
    ##### TRAJECTORIES ####
    from ATMANpy.models.BallTrajectory.flight_bounce_model import load_trajectories, initialize
    traj_name = 'FullTrajectories_g60_v41_s31_h20'
    groundArr, vertArr, rpmArr, trajDict = load_trajectories(traj_name, dtype = 'float32')
    
    ### initialise search spaces with proper arguments
    import inspect
    kwargList = inspect.getargspec(initialize).args
    kwargs = {k: v for k, v in trajDict.items() if k in kwargList}
    v_init_ground_space, v_init_vert_space, spin_rate_space, height_space = initialize(dtype = 'float32', 
                                                                                       squeeze = True,  
                                                                                       **kwargs)
        
    conf = ut.read_config('brest', config_name = '_config_pipe2')
    
    plane_point_params = {'max_sideways_reach_in_feet': 5, 'resolution': 1}
    hit_point_params = {'height_in_feet': 5, 'resolution': 0.5, 'radius_in_feet': 15}
    
    downsample_params = {'logit_coeff': 10, 'sampling_number': 25}
    viterbi_net_params = {'p': 0.95, 'epsilon': 0.01, 'q': 0.1}
    penalty_params = {'frame_cutoff': 5}
    
    traj_fit_params = {'groundArr': groundArr, 'vertArr': vertArr, 'rpmArr': rpmArr, 
                       'observation_threshold': 3, 'sum_of_squares': True, 'number_of_local_minima': 30}
    viterbi_traj_params = {'dst_factor': 0.2, 'dev_factor': 0.4, 'bnc_factor': 0.4}    

    main(conf, 
         plane_point_params, hit_point_params, 
         downsample_params, viterbi_net_params, penalty_params, 
         traj_fit_params, viterbi_traj_params)

