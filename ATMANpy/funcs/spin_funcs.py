# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:49:29 2023

@author: gianl
"""


import os
import pandas as pd
import numpy as np
from ATMANpy.funcs.util_funcs import get_folder_struct, shift_array, add_in_nan_rows, argmin_multidim

### SEARCH SPACE PARAMETERS ###
GRANULARITY_INITIAL_GROUND = 75 #100 # 325 ### how many different initial conditions in x
GRANULARITY_INITIAL_VERT = 41 #50 # 200 ### how many different initial conditions in z
GRANULARITY_SPIN = 51 #301 ### how many different spin rates
MIN_SPEED_GROUND = 1 ## lowest possible ground speed of ball, in m/s
MAX_SPEED_GROUND = 75 ## highest possible ground speed of ball, in m/s
MIN_SPEED_VERT = -20
MAX_SPEED_VERT =  20
MIN_SPIN_RATE = -5000 ### highest topspin RPM (negative, because topspin causes negative lift)  
MAX_SPIN_RATE = 5000  ### higgest slice RPM (positive, because slice causes positive lift)
DURATION_OF_TRAJECTORY = 4 ### in seconds
PATHOLOGICAL_RANGE = 2 # 10 -> twice the min/max spin rate is added to the thing. 
PATHOLOGICAL_TAIL_GRANULARITY = 0 ## zero for no pathological cases

### VIDEO PARAMETERS ####
# FPS = 25
STEP_DURATION = 1 / 50

### PHYSICAL PARAMETERS ####
p = 1.21 ### air density, in kg / m ^ 3
R = 33 / 10**3 ### radius of tennis ball in meters (diameter = 66mm = 66m * 10**-3) 
m = 58 / 10**3 ### mass of tennis ball in kg (58 g = 58 kg * 10**-3)
g = 9.78 ### gravity of earth, in m/s**2

C_D = 0.55 ### drag coefficient
C_L_COEFFICIENT = 0.6

k = 0.5 * p * np.pi * R**2 / m ### unit 1 / m
###################

########
NAME = f'TrajectoriesRPM_tight_halfstep'
########

def lift_coeff_to_RPM(lift_coeff, ball_speed, 
                      C_L_coefficient = C_L_COEFFICIENT, R = R):
    S = lift_coeff / C_L_coefficient ### dimensionless ratio
    R_w = S * ball_speed ### meters per second (or whatever the unit of ball_speed is!)
    RPM = R_w / (2 * np.pi * R) * 60 ### revolutions per minute!
    
    return RPM

def RPM_to_lift_coeff(RPM, ball_speed,
                      C_L_coefficient = C_L_COEFFICIENT, R = R):
    R_w = RPM * (2 * np.pi * R) / 60
    S = R_w / ball_speed
    lift_coeff = S * C_L_coefficient
    return lift_coeff

def initialize(dtype = 'float64', squeeze = False,
               granularity_initial_ground = GRANULARITY_INITIAL_GROUND, 
               granularity_initial_vert = GRANULARITY_INITIAL_VERT, 
               granularity_spin = GRANULARITY_SPIN,
               min_speed_ground = MIN_SPEED_GROUND, max_speed_ground = MAX_SPEED_GROUND,
               min_speed_vert = MIN_SPEED_VERT, max_speed_vert = MAX_SPEED_VERT,
               min_spin_rate = MIN_SPIN_RATE, 
               max_spin_rate = MAX_SPIN_RATE,
               pathological_range = PATHOLOGICAL_RANGE,
               pathological_tail_granularity = PATHOLOGICAL_TAIL_GRANULARITY):
        
    initial_v_x = np.linspace(min_speed_ground, max_speed_ground, granularity_initial_ground, dtype = dtype) # initial condition: horizontal speed
    initial_v_z = np.linspace(min_speed_vert, max_speed_vert, granularity_initial_vert, dtype = dtype) # initial condition: vertical speed
    spin_rate_space = np.linspace(min_spin_rate, max_spin_rate, granularity_spin, dtype = dtype)
    
    ### add pathological tails to spin rate space
    avg_spin_rate = (min_spin_rate + max_spin_rate) / 2
    pathological_lo_spin_rate = (min_spin_rate - avg_spin_rate) * pathological_range
    pathological_hi_spin_rate = (max_spin_rate - avg_spin_rate) * pathological_range
    one_tail_granularity = int(pathological_tail_granularity * granularity_spin) ### 5% extra space on each tail -> 10% extra total
    lo_spin_rate_space = np.linspace(pathological_lo_spin_rate, min_spin_rate, one_tail_granularity)[:-1] 
    hi_spin_rate_space = np.linspace(max_spin_rate, pathological_hi_spin_rate, one_tail_granularity)[1:]
    
    spin_rate_space = np.concatenate([lo_spin_rate_space, spin_rate_space, hi_spin_rate_space])
    
    if squeeze:
        return initial_v_x, initial_v_z, spin_rate_space
    else:
        v_x, v_z = initial_v_x[:, np.newaxis, np.newaxis], initial_v_z[np.newaxis, :, np.newaxis]
        spin_rates = spin_rate_space[np.newaxis, np.newaxis, :]
        return v_x, v_z, spin_rates

def calculate_speed_step(v_x, v_z, spin_rates,
                         step_duration = STEP_DURATION,
                         C_D = C_D, C_L_coefficient = C_L_COEFFICIENT,
                         R = R, k = k, g = g):
    ### using the trajectory differential equations, calculate the speed in the next second
    ### we are working with matrices of the shape v_x, v_y, C_L
    v = (v_x ** 2 + v_z ** 2) ** 0.5
    kv = k * v
    
    # calculate lift coefficients
    lift_coeffs = RPM_to_lift_coeff(spin_rates, v, C_L_coefficient = C_L_coefficient, R = R)
    
    a_x = - kv * (C_D * v_x - lift_coeffs * v_z) 
    a_z = - kv * (lift_coeffs * v_x + C_D * v_z) - g
    
    return v_x + a_x * step_duration, v_z + a_z * step_duration

def save_trajectories(name, xArr, zArr, **kwargs):
    folDict = get_folder_struct('trajectories_stored')
    
    if name in folDict:
        print(f'Trajectories {name} exist. Will not overwrite.')
        return None
    else:
        trajDir = os.path.join(folDict['main'], name)
        os.mkdir(trajDir)
    
        np.save(os.path.join(trajDir, 'xArr.npy'), xArr)
        np.save(os.path.join(trajDir, 'zArr.npy'), zArr)
        
        lines_to_write = [f'{key}: {value}\n' for key, value in kwargs.items()]
        with open(os.path.join(trajDir, 'info.txt'), 'w') as f:
            f.writelines(lines_to_write)
        
def load_trajectories(name, dtype = 'float32'):
    folDict = get_folder_struct('trajectories_stored')
    
    if name not in folDict:
        print(f'Trajectories {name} do not exist')
        return None
    else:
        trajDir = os.path.join(folDict['main'], name)
        
        ### read info
        def read_info(myStr):
            try:
                return int(myStr)
            except:
                pass
            
            try:
                return float(myStr)
            except:
                pass
            
            return myStr
        
        with open(os.path.join(trajDir, 'info.txt'), 'r') as f:
            lines = f.readlines()
            lines = [l.rstrip('\n').split(': ') for l in lines]
            infoDict = {x: read_info(y) for x, y in lines}
        
        ### read arrays
        xArr = np.load(os.path.join(trajDir, 'xArr.npy'))
        zArr = np.load(os.path.join(trajDir, 'zArr.npy'))
        
        if not xArr.dtype == dtype:
            xArr = xArr.astype(dtype)
        if not zArr.dtype == dtype:
            zArr = zArr.astype(dtype)
        
        return xArr, zArr, infoDict
    
def create_trajectories(name, dtype_calc = 'float64', dtype_save = 'float64',
                        granularity_initial_ground = GRANULARITY_INITIAL_GROUND, 
                        granularity_initial_vert = GRANULARITY_INITIAL_VERT, 
                        granularity_spin = GRANULARITY_SPIN,
                        min_speed_ground = MIN_SPEED_GROUND, max_speed_ground = MAX_SPEED_GROUND,
                        min_speed_vert = MIN_SPEED_VERT, max_speed_vert = MAX_SPEED_VERT,
                        min_spin_rate = MIN_SPIN_RATE, max_spin_rate = MAX_SPIN_RATE,
                        duration_of_trajectory = DURATION_OF_TRAJECTORY,
                        step_duration = STEP_DURATION,
                        p = p, R = R, m = m, g = g, C_D = C_D):  

    import time
    
    ### check if folder already exists!
    folDict = get_folder_struct('trajectories_stored')
    
    t0 = time.time()

    k = 0.5 * p * np.pi * R**2 / m ### unit 1 / m
    
    v_x, v_z, spin_rates = initialize(dtype = dtype_calc, 
                                      granularity_initial_ground = granularity_initial_ground, 
                                      granularity_initial_vert = granularity_initial_vert, 
                                      granularity_spin = granularity_spin,
                                      min_speed_ground = min_speed_ground, max_speed_ground = max_speed_ground,
                                      min_speed_vert = min_speed_vert, max_speed_vert = max_speed_vert,
                                      min_spin_rate = min_spin_rate, 
                                      max_spin_rate = max_spin_rate)
    dimensions = v_x.shape[0], v_z.shape[1], spin_rates.shape[2]  
    
    xList = [np.zeros(dimensions, dtype = dtype_calc)] ### this thing will be (v_x, v_y, RPM, t)
    zList = [np.zeros(dimensions, dtype = dtype_calc)] ### this thing will be (v_x, v_y, RPM, t)
    for frames in range(0, int(duration_of_trajectory / step_duration)):
        
        x = xList[-1] + v_x * step_duration
        z = zList[-1] + v_z * step_duration
        
        xList.append(x)
        zList.append(z)
        
        v_x, v_z = calculate_speed_step(v_x, v_z, spin_rates,
                                        step_duration = step_duration,
                                        k = k, C_D = C_D, g = g)
        
    xArr = np.stack(xList, axis = -1).astype(dtype_save)
    zArr = np.stack(zList, axis = -1).astype(dtype_save)

    t1 = time.time()
    seconds_elapsed = f'{round(t1 - t0)} seconds'
    print(f'Done! Took {seconds_elapsed}')
    
    if name in folDict:
        print(f'Trajectories {name} exist. Will not overwrite.')
    else:
        save_trajectories(name, xArr, zArr, 
                          granularity_initial_ground = granularity_initial_ground, 
                          granularity_initial_vert = granularity_initial_vert, 
                          granularity_spin = granularity_spin,
                          min_speed_ground = min_speed_ground, max_speed_ground = max_speed_ground,
                          min_speed_vert = min_speed_vert, max_speed_vert = max_speed_vert,
                          min_spin_rate = min_spin_rate, 
                          max_spin_rate = max_spin_rate,
                          duration_of_trajectory = duration_of_trajectory,
                          step_duration = step_duration,
                          p = p, R = R, m = m, g = g, C_D = C_D,
                          time_taken = seconds_elapsed)

    return xArr, zArr

#%%############ SPIN ESTIMATION  ############

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.HMM.HPost as HPost

def calculate_initial_conditions(trajFrm, frames_to_use = 1):
    ### frames to use: how many of the first frames do we use to calculate the initial speeds?    
    xDim, yDim, zDim = trajFrm.columns
    
    ground_speeds = []
    vert_speeds = []
    for f in range(frames_to_use): 
        offset = f + 1 ### if we only use 1 frame, offset will be 1, as in the default of calc_ball_dist
        
        ground_dist = HPost.calc_ball_dist(trajFrm[[xDim, yDim]], norm = 2, offset = offset) ### in points
        ground_speed = HPost.calc_ball_speed(ground_dist, frames = offset, unit = 'mps')
    
        vert_dist = HPost.calc_ball_dist(trajFrm[[zDim]], norm = 1, offset = offset)
        vert_speed = HPost.calc_ball_speed(vert_dist, frames = offset, unit = 'mps') # transform to m/s
    
        ground_speeds.append(ground_speed.iloc[0])
        vert_speeds.append(vert_speed.iloc[0])
        
    avg_ground_speed = np.mean(ground_speeds)
    avg_vert_speed = np.mean(vert_speeds)
    
    return avg_ground_speed, avg_vert_speed
    
def calculate_planar_ball_flight(trajFrm):
    orig_index = trajFrm.index
    trajFrm_nonna = trajFrm.dropna()
    observed_index = trajFrm.index
    
    xDim, yDim, zDim = trajFrm_nonna.columns
    ground_dist = HPost.calc_ball_dist(trajFrm_nonna[[xDim, yDim]], norm = 2) ### in points
    ground_dist_meters = ground_dist / const.SCALE * const.FOOT_TO_METER ### in meters
    ground_dist_meters = ground_dist_meters.cumsum().shift(1).fillna(0) ### normalised, so that at hit, m = 0

    vert_dist = HPost.calc_ball_dist(trajFrm_nonna[[zDim]], norm = 1)
    vert_dist_meters = vert_dist / const.SCALE * const.FOOT_TO_METER ### in meters
    vert_dist_meters = vert_dist_meters.cumsum().shift(1).fillna(0) ### normalised, so that at hit, m = 0

    ground_dist_meters = pd.Series(ground_dist_meters, name = 'ground_m').reindex(orig_index)
    vert_dist_meters = pd.Series(vert_dist_meters, name = 'vert_m').reindex(orig_index)

    return pd.concat([ground_dist_meters, vert_dist_meters], axis = 1)

def calculate_planar_ball_flight_vectorised(coordArr):
    """
    calculates planar flight values in meters
    coordArr: shape: dim1, dim2, ..., time, space (=3)
    """
    ### TODO: drop nan times in the beginning (otherwise dist calculation does not work with shift...), and then add them back in at the end
    transArr = coordArr.T ## shape: space, time, ..., dim2, dim1
    
    nan_times = np.isnan(transArr).all(axis = tuple([ax for ax in range(len(transArr.shape)) if ax != 1]))
    transArr_nonnan = transArr[:, ~nan_times]
    
    shiftArr = shift_array(transArr_nonnan, 1, axis = 1) ### this axis is the time axis!
    
    squared_dists = (transArr_nonnan - shiftArr) ** 2
    squared_dists[:, 0] = 0 ### distance in first time (at hit point) is zero!
    ground_dist = squared_dists[[0, 1]].sum(axis = 0) ** 0.5
    vert_dist = squared_dists[2] ** 0.5

    ground_dist_meters = ground_dist / const.SCALE * const.FOOT_TO_METER
    ground_cum_dist_meters = ground_dist_meters.cumsum(axis = 0) ### add up all distances between two values
    # ground_cum_dist_meters = shift_array(ground_cum_dist_meters, 1, axis = 0, fill_value = 0) ### move first differences to the right, and add in zeros
    
    vert_dist_meters = vert_dist / const.SCALE * const.FOOT_TO_METER
    vert_cum_dist_meters = vert_dist_meters.cumsum(axis = 0) ### add up all distances between two values
    # vert_cum_dist_meters = shift_array(vert_cum_dist_meters, 1, axis = 0, fill_value = 0) ### move first differences to the right, and add in zeros
    
    planar_dists = np.stack([ground_cum_dist_meters, vert_cum_dist_meters], axis = 0)
    ### -> SHAPE: 2 x T x ... x dim2 x dim1

    ### add nan_times back in!
    planar_dists = add_in_nan_rows(planar_dists, nan_times, axis = 1)
    
    return planar_dists.T ### -> SHAPE: dim1 x dim2 x ... x T x 2 
    
    
def get_optimal_trajectory_indices(groundArr, vertArr, 
                                   pos_ground_measured, pos_vert_measured,
                                   sum_of_squares = True, return_dev = False):
    ### Search over all initial conditions, minimise distances
    ### sum_of_squares: if true, minimise sum of squares; if false, minimise absolute deviation.
    assert len(pos_ground_measured) == len(pos_vert_measured)
    num_frames = len(pos_ground_measured)
        
    pos_ground_theoretical = groundArr[:, :, :, :num_frames]
    pos_vert_theoretical = vertArr[:, :, :, :num_frames]
        
    ground_dev = pos_ground_measured[np.newaxis, np.newaxis, np.newaxis, :] - pos_ground_theoretical
    vert_dev = pos_vert_measured[np.newaxis, np.newaxis, np.newaxis, :] - pos_vert_theoretical
    
    if sum_of_squares:
        my_dev = (ground_dev ** 2 + vert_dev ** 2) ** 0.5
    else:
        my_dev = np.abs(ground_dev) + np.abs(vert_dev)
        
    sum_dev = np.nansum(my_dev, axis = 3)
    best_idx_flat = sum_dev.argmin()
    
    v_ground_idx, v_vert_idx, rpm_idx = np.unravel_index(best_idx_flat, sum_dev.shape)
    
    if return_dev:
        best_dev = my_dev[v_ground_idx, v_vert_idx, rpm_idx]
        return v_ground_idx, v_vert_idx, rpm_idx, best_dev
    else:
        return v_ground_idx, v_vert_idx, rpm_idx
    
def get_optimal_trajectory_indices_vectorised(groundArr, vertArr, 
                                              pos_ground_measured, pos_vert_measured,
                                              sum_of_squares = True):
    ### shape groundArr, vertArr: v_ground x v_vert x spin_rpm x time
    ### shape pos_ground_measured, pos_vert_measured: dim1 x dim2 x ... x t
    measured_shape = pos_ground_measured.shape
    extra_dims, num_frames = measured_shape[:-1], measured_shape[-1]
    
    pos_ground_theoretical = groundArr[:, :, :, :num_frames]
    pos_vert_theoretical = vertArr[:, :, :, :num_frames]
    
    for _ in extra_dims:
        pos_ground_theoretical = np.expand_dims(pos_ground_theoretical, axis = -2)
        pos_vert_theoretical = np.expand_dims(pos_vert_theoretical, axis = -2)
    
    for _ in range(3): ### initial condition space
        pos_ground_measured = np.expand_dims(pos_ground_measured, axis = 0)
        pos_vert_measured = np.expand_dims(pos_vert_measured, axis = 0)
    
    ground_dev = pos_ground_measured - pos_ground_theoretical
    vert_dev = pos_vert_measured - pos_vert_theoretical
    
    if sum_of_squares:
        my_dev = (ground_dev ** 2 + vert_dev ** 2) ** 0.5
    else:
        my_dev = np.abs(ground_dev) + np.abs(vert_dev)
        
    sum_dev = np.nansum(my_dev, axis = -1) ### shape: v_ground, v_vert, rpm, traj
    
    traj_conds = []
    for traj_idx in range(sum_dev.shape[-1]): # this only works for 1 extra dimension!!!
        traj_dev = sum_dev[:, :, :, traj_idx]
        traj_ground_idx, traj_vert_idx, traj_rpm_idx = argmin_multidim(traj_dev)
        traj_best_dev = traj_dev[traj_ground_idx, traj_vert_idx, traj_rpm_idx]
        traj_conds.append((traj_vert_idx, traj_ground_idx, traj_rpm_idx, traj_best_dev))
    
    return np.array(traj_conds)
    
import matplotlib.pyplot as plt
def calculate_distribution_at_time(xArr, zArr, quant_step = 0.05):
    qs = np.linspace(0, 1, num = int(1 / quant_step) + 1, endpoint = True)

    del1_xArr = xArr - shift_array(xArr, 1, axis = 3)
    del2_xArr = del1_xArr - shift_array(del1_xArr, 1, axis = 3)
    
    del1_zArr = zArr - shift_array(zArr, 1, axis = 3)
    del2_zArr = del1_zArr - shift_array(del1_zArr, 1, axis = 3)
    
    x_quants = np.quantile(xArr, qs, axis = (0, 1, 2))
    dx_quants = np.quantile(del1_xArr, qs, axis = (0, 1, 2))
    d2x_quants = np.quantile(del2_xArr, qs, axis = (0, 1, 2))
    
    z_quants = np.quantile(zArr, qs, axis = (0, 1, 2))
    dz_quants = np.quantile(del1_zArr, qs, axis = (0, 1, 2))
    d2z_quants = np.quantile(del2_zArr, qs, axis = (0, 1, 2))
    
    x_out = np.stack([x_quants, dx_quants, d2x_quants], axis = -1).T ### shape -> 3 x T x Qs
    z_out = np.stack([z_quants, dz_quants, d2z_quants], axis = -1).T ### shape -> 3 x T x Qs

    return x_out, z_out

def calculate_trajectory_derivatives(planeArr):
    """
    planeArr shape: dim1, dim2, ..., time, space (=2; ground and height)
    """
    dims = len(planeArr.shape)
    time_dim, space_dim = dims - 2, dims - 1
    num_flight_frames = planeArr.shape[time_dim]
    
    non_time_axes = tuple([i for i in range(dims) if i != time_dim])
    is_nan_idxs = np.isnan(planeArr).all(axis = non_time_axes)
    
    ### modify the planeArr to include time as a dimension 
        ### -> this way, we can drop nans and approximate derivatives 
        ### without dealing with nans.
    timeArr = np.arange(0, num_flight_frames)
    _, timeArr = np.broadcast_arrays(planeArr[:, :, :, 0], timeArr[np.newaxis, np.newaxis, :])
    timeArr = timeArr[:, :, :, np.newaxis]
    planeArr_mod = np.concatenate([planeArr, timeArr], axis = 3)
    planeArr_mod = planeArr_mod[:, :, ~is_nan_idxs]
    
    ### calculate derivatives
    diffArr = planeArr_mod - shift_array(planeArr_mod, 1, axis = time_dim)
    der1Arr = diffArr.take([0, 1], axis = space_dim) / diffArr.take([2], axis = space_dim)
    der2Arr = der1Arr - shift_array(der1Arr, 1, axis = time_dim)
    
    ### prepare to join them
    diffArr = diffArr.take([0, 1], axis = space_dim)
    derivArr = np.stack([diffArr, der1Arr, der2Arr], axis = 0)
    time_dim = time_dim + 1 ### now that we have added the derivative dimension
    derivArr = add_in_nan_rows(derivArr, is_nan_idxs, axis = time_dim)
    derivArr = np.moveaxis(derivArr, time_dim, 1)
    x_derivs, z_derivs = derivArr.take(0, axis = -1), derivArr.take(1, axis = -1)
    return x_derivs, z_derivs
    
# def compare_derivatives_with_quantiles(x_derivs, z_derivs, x_quants, z_quants, 
#                                        trajectory_cutoff = 10, min_q = 0.05, max_q = 0.95):
#     #x_quants, z_quants shape: deriv_number, time, quantile
#     #x_derivs, z_derivs shape: deriv_number, time, dim1, dim2, ...
    
#     ### calculate the quantile index corresponding to the min_q and max_q chosen
#     num_qs = x_quants.shape[-1]
#     qs = np.linspace(0, 1, num = num_qs, endpoint = True)
#     min_idx, max_idx = np.argmin(np.abs(qs - min_q)), np.argmin(np.abs(qs - max_q))
   
#     ### limit to trajectory cutoff observations
#     x_derivs_trunc, z_derivs_trunc = x_derivs[:, :trajectory_cutoff], z_derivs[:, :trajectory_cutoff]
#     x_quants_trunc, z_quants_trunc = x_quants[:]
    
#     np.linspace(0, 1, endpoint = True, num = num_qs)
#     np.arange(0, 100, 100 // num_qs)
    
#     x_derivs, y_derivs = derivArr.take(0, axis = -1), derivArr.take(1, axis = -1)
    
    


#%% MAIN ##
if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import argmin_multidim
    import matplotlib.pyplot as plt
    
    # for dtype in ['float64', 'float32', 'float16']:
    #     t = time.time()
    #     xArr, zArr = create_trajectories(f'TestTrajectories_{dtype}', 
    #                                      dtype_calc = dtype, dtype_save = dtype)
    #     print(f'Done with {dtype}! Took {time.time() - t} seconds')
        
    # #### NOTE HOW FLOAT16 TAKES LONGEST (APPARENTLY NO HARDWARE SUPPORT...)
    
    xArr, zArr = create_trajectories(NAME, 
                                     dtype_calc = 'float32', dtype_save = 'float16',
                                     step_duration = STEP_DURATION)
    
    ########## TESTING ##########
    initial_v_x, initial_v_z, spin_rate_space = initialize(squeeze = True)
    def get_match_idx(arr, val):
        return np.argmin(np.abs(arr - val))
    def limit_to_threshold(my_xArr, my_zArr, threshold = -1):
        filt = my_zArr >= threshold
        return my_xArr[filt], my_zArr[filt]
    
    
    ##### Replicate Fig. 6 in Rod Cross Paper
    spd0, ang0, rpm0 = 30, 7.4, 0
    spd1, ang1, rpm1 = 30, 15.2, 4000
    
    vx0, vy0 = spd0 * np.cos(ang0 * np.pi / 180), spd0 * np.sin(ang0 * np.pi / 180) 
    vx1, vy1 = spd1 * np.cos(ang1 * np.pi / 180), spd1 * np.sin(ang1 * np.pi / 180) 
    
    vx0_idx, vy0_idx = get_match_idx(initial_v_x, vx0), get_match_idx(initial_v_z, vy0)
    vx1_idx, vy1_idx = get_match_idx(initial_v_x, vx1), get_match_idx(initial_v_z, vy1)
    
    rpm0_idx, rpm1_idx = get_match_idx(spin_rate_space, rpm0), get_match_idx(spin_rate_space, rpm1)
    
    fig, ax = plt.subplots(1, 1)
    
    xArr0, zArr0 = xArr[vx0_idx, vy0_idx, rpm0_idx], zArr[vx0_idx, vy0_idx, rpm0_idx]
    xArr1, zArr1 = xArr[vx1_idx, vy1_idx, rpm1_idx], zArr[vx1_idx, vy1_idx, rpm1_idx]
    
    xArr0, zArr0 = limit_to_threshold(xArr0, zArr0)
    xArr1, zArr1 = limit_to_threshold(xArr1, zArr1)
    
    ax.plot(xArr0, zArr0)
    ax.plot(xArr1, zArr1)
    
    
    
    #### Visualize Spin
    ground_speed = 30
    vert_speed = 2
    spins = np.array([-4000, 0, 4000])
    
    initial_v_x_idx = np.argmin(np.abs(initial_v_x - ground_speed))
    initial_v_z_idx = np.argmin(np.abs(initial_v_z - vert_speed))
    
    xArr_chk, zArr_chk = xArr[initial_v_x_idx, initial_v_z_idx], zArr[initial_v_x_idx, initial_v_z_idx]
    
    spin_idxs = []
    for spin_devs in np.abs(spin_rate_space[np.newaxis, :] - spins[:, np.newaxis]):
        spin_idxs.append(np.argmin(spin_devs))
    
    y_threshold = -1
    fig, ax = plt.subplots(1, 1)
    # ax.set_ylim(bottom = y_threshold, top = None)
    
    for spin_idx in spin_idxs:
        xArr_plot, zArr_plot = xArr_chk[spin_idx], zArr_chk[spin_idx]
        xArr_plot, zArr_plot = xArr_plot[zArr_plot > y_threshold], zArr_plot[zArr_plot > y_threshold]
    
        ax.plot(xArr_plot, zArr_plot)
        


    
    
    
    
        