# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:49:29 2023

@author: gianl
"""


import os
import pandas as pd
import numpy as np
from ATMANpy.funcs.util_funcs import get_folder_struct, shift_array, add_in_nan_rows, argmin_multidim, detect_local_minima

### SEARCH SPACE PARAMETERS ###
GRANULARITY_INITIAL_GROUND = 60 #100 # 325 ### how many different initial conditions in x
GRANULARITY_INITIAL_VERT = 41 #50 # 200 ### how many different initial conditions in z
GRANULARITY_SPIN = 31 #301 ### how many different spin rates
GRANULARITY_HEIGHT = 20 ### how many different initial heights
MIN_SPEED_GROUND = 1 ## lowest possible ground speed of ball, in m/s
MAX_SPEED_GROUND = 60 ## highest possible ground speed of ball, in m/s
MIN_SPEED_VERT = -20
MAX_SPEED_VERT =  20
MIN_SPIN_RATE = -5000 ### highest topspin RPM (negative, because topspin causes negative lift)  
MAX_SPIN_RATE = 5000  ### higgest slice RPM (positive, because slice causes positive lift)
MIN_HIT_HEIGHT = 0 ### min height of ball, at hit
MAX_HIT_HEIGHT = 3 ### max height of ball, at hit
DURATION_OF_TRAJECTORY = 4 ### in seconds
PATHOLOGICAL_RANGE = 2 # 10 -> twice the min/max spin rate is added to the thing. 
PATHOLOGICAL_TAIL_GRANULARITY = 0 ## zero for no pathological cases

### VIDEO PARAMETERS ####
# FPS = 25
STEP_DURATION = 1 / 25

### PHYSICAL PARAMETERS ####
p = 1.21 ### air density, in kg / m ^ 3
R = 33 / 10**3 ### radius of tennis ball in meters (diameter = 66mm = 66m * 10**-3) 
m = 58 / 10**3 ### mass of tennis ball in kg (58 g = 58 kg * 10**-3)
g = 9.78 ### gravity of earth, in m/s**2

C_D = 0.55 ### drag coefficient
C_L_COEFFICIENT = 0.6

###################

### BOUNCE PARAMETERS ###

alpha = 0.55 ### inertia parameter -> depends on ball wall thickness
mu = 0.73 ### coefficient of sliding friction -> depends on court speed
e_x = 0.1
# e_y = 0.75 ### normal coefficient of restitution  -> depends on court surface and temperature
### -> e_y calculated depending on the ball bounce angle

### DERIVED PARAMETERS ###
k = 0.5 * p * np.pi * R**2 / m ### unit 1 / m
I_cm = alpha * m * R**2 ### moment of inertia


########
NAME = f'FullTrajectories_g{GRANULARITY_INITIAL_GROUND}_v{GRANULARITY_INITIAL_VERT}_s{GRANULARITY_SPIN}_h{GRANULARITY_HEIGHT}'
########

def RPM_to_omega(RPM):
    return RPM / 60 * (2 * np.pi)

def omega_to_RPM(omega): 
    return omega * 60 / (2 * np.pi)

def lift_coeff_to_RPM(lift_coeff, ball_speed, 
                      C_L_coefficient = C_L_COEFFICIENT, R = R):
    S = lift_coeff / C_L_coefficient ### dimensionless ratio
    R_w = S * ball_speed ### meters per second (or whatever the unit of ball_speed is!)
    RPM = omega_to_RPM(R_w / R) 
    return RPM

def RPM_to_lift_coeff(RPM, ball_speed,
                      C_L_coefficient = C_L_COEFFICIENT, R = R):
    R_w = R * RPM_to_omega(RPM)
    S = R_w / ball_speed
    lift_coeff = S * C_L_coefficient
    return lift_coeff

def initialize(dtype = 'float64', squeeze = False,
               granularity_initial_ground = GRANULARITY_INITIAL_GROUND, 
               granularity_initial_vert = GRANULARITY_INITIAL_VERT, 
               granularity_spin = GRANULARITY_SPIN,
               granularity_height = GRANULARITY_HEIGHT,
               min_speed_ground = MIN_SPEED_GROUND, max_speed_ground = MAX_SPEED_GROUND,
               min_speed_vert = MIN_SPEED_VERT, max_speed_vert = MAX_SPEED_VERT,
               min_spin_rate = MIN_SPIN_RATE, 
               max_spin_rate = MAX_SPIN_RATE,
               min_hit_height = MIN_HIT_HEIGHT,
               max_hit_height = MAX_HIT_HEIGHT):
        
    initial_v_x = np.linspace(min_speed_ground, max_speed_ground, granularity_initial_ground, dtype = dtype) # initial condition: horizontal speed
    initial_v_y = np.linspace(min_speed_vert, max_speed_vert, granularity_initial_vert, dtype = dtype) # initial condition: vertical speed
    initial_spins = np.linspace(min_spin_rate, max_spin_rate, granularity_spin, dtype = dtype)
    initial_heights = np.linspace(min_hit_height, max_hit_height, granularity_height, dtype = dtype)
    
    if squeeze:
        return initial_v_x, initial_v_y, initial_spins, initial_heights
    else:
        initial_v_x = initial_v_x[:, np.newaxis, np.newaxis, np.newaxis]
        initial_v_y = initial_v_y[np.newaxis, :, np.newaxis, np.newaxis]
        initial_spins = initial_spins[np.newaxis, np.newaxis, :, np.newaxis]
        initial_heights = initial_heights[np.newaxis, np.newaxis, np.newaxis, :]
        return initial_v_x, initial_v_y, initial_spins, initial_heights

def calculate_speed_step(v_x, v_y, spin_rates,
                         step_duration = STEP_DURATION,
                         C_D = C_D, C_L_coefficient = C_L_COEFFICIENT,
                         R = R, k = k, g = g):
    ### using the trajectory differential equations, calculate the speed in the next second
    ### we are working with matrices of the shape v_x, v_y, C_L
    v = (v_x ** 2 + v_y ** 2) ** 0.5
    kv = k * v
    
    # calculate lift coefficients
    lift_coeffs = RPM_to_lift_coeff(spin_rates, v, C_L_coefficient = C_L_coefficient, R = R)
    
    a_x = - kv * (C_D * v_x - lift_coeffs * v_y) 
    a_y = - kv * (lift_coeffs * v_x + C_D * v_y) - g
    
    return v_x + a_x * step_duration, v_y + a_y * step_duration, spin_rates

def calculate_bounce_parameters(v_x, v_y, spin_rates):
    v = (v_x ** 2 + v_y ** 2) ** 0.5
    
    tan_theta = -v_y / v_x ## -> the vertical speed vector in the bounce model is reversed
    theta = np.arctan(tan_theta)
    
    e_y = 0.95 - 0.005 * theta * 180 / np.pi ### TODO: apparently, vertical e_y = 0.75, but with this formulation, we'd have vertical e_y = 0.45
    
    D = 0.0003 * v ### how far does the ball slide before beginning to bounce up
    
    return e_y, tan_theta, D
    
def calculate_sliding_bounce_step(v_x, v_y, spin_rates,
                                  e_y, tan_theta, D,
                                  mu = mu, R = R, alpha = alpha):
    ### sliding solutions
    ground_speed_ratio = 1 - mu * (1 + e_y) * tan_theta
    
    v_x2 = ground_speed_ratio * v_x
    v_y2 = e_y * -v_y
    
    omega_incoming = RPM_to_omega(spin_rates)
    omega_outgoing = omega_incoming + (1 + e_y) * (mu - D / R) * (-v_y/ (alpha * R))

    spin_rates2 = omega_to_RPM(omega_outgoing)
    
    return v_x2, v_y2, spin_rates2

def calculate_gripping_bounce_step(v_x, v_y, spin_rates,
                                   e_y, tan_theta, D,
                                   e_x = e_x, R = R, alpha = alpha):
    omega_incoming = RPM_to_omega(spin_rates)
    
    S1 = R * omega_incoming / v_x
    
    comp1 = (1 - alpha * e_x) / (1 + alpha)
    comp2 = (alpha * (1 + e_x) * S1) / (1 + alpha)
    comp3 = ((1 + e_y) * D * tan_theta) / ((1 + alpha) * R)
    
    ground_speed_ratio = comp1 + comp2 - comp3
    
    v_x2 = ground_speed_ratio * v_x
    v_y2 = e_y * -v_y
    
    comp1 = omega_incoming
    comp2 = (v_x - v_x2) / (alpha * R)
    comp3 = ((1 + e_y) * D * -v_y) / (alpha * R ** 2)
    
    omega_outgoing = comp1 + comp2 - comp3
    spin_rates2 = omega_to_RPM(omega_outgoing)
    
    return v_x2, v_y2, spin_rates2

def save_trajectories(name, xArr, zArr, rpmArr, **kwargs):
    folDict = get_folder_struct('trajectories_stored')
    
    if name in folDict:
        print(f'Trajectories {name} exist. Will not overwrite.')
        return None
    else:
        trajDir = os.path.join(folDict['main'], name)
        os.mkdir(trajDir)
    
        np.save(os.path.join(trajDir, 'xArr.npy'), xArr)
        np.save(os.path.join(trajDir, 'zArr.npy'), zArr)
        np.save(os.path.join(trajDir, 'rpmArr.npy'), rpmArr)
        
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
        rpmArr = np.load(os.path.join(trajDir, 'rpmArr.npy'))
        
        if not xArr.dtype == dtype:
            xArr = xArr.astype(dtype)
        if not zArr.dtype == dtype:
            zArr = zArr.astype(dtype)
        if not rpmArr.dtype == dtype:
            rpmArr = rpmArr.astype(dtype)
        
        return xArr, zArr, rpmArr, infoDict
    
granularity_initial_ground = GRANULARITY_INITIAL_GROUND
granularity_initial_vert = GRANULARITY_INITIAL_VERT
granularity_spin = GRANULARITY_SPIN
min_speed_ground = MIN_SPEED_GROUND
max_speed_ground = MAX_SPEED_GROUND
min_speed_vert = MIN_SPEED_VERT
max_speed_vert = MAX_SPEED_VERT
min_spin_rate = MIN_SPIN_RATE
max_spin_rate = MAX_SPIN_RATE
duration_of_trajectory = DURATION_OF_TRAJECTORY
step_duration = STEP_DURATION
dtype_calc = 'float64'
dtype_save = 'float64'
    
def create_trajectories(name, dtype_calc = 'float64', dtype_save = 'float64',
                        granularity_initial_ground = GRANULARITY_INITIAL_GROUND, 
                        granularity_initial_vert = GRANULARITY_INITIAL_VERT, 
                        granularity_spin = GRANULARITY_SPIN,
                        min_speed_ground = MIN_SPEED_GROUND, max_speed_ground = MAX_SPEED_GROUND,
                        min_speed_vert = MIN_SPEED_VERT, max_speed_vert = MAX_SPEED_VERT,
                        min_spin_rate = MIN_SPIN_RATE, max_spin_rate = MAX_SPIN_RATE,
                        duration_of_trajectory = DURATION_OF_TRAJECTORY,
                        step_duration = STEP_DURATION,
                        p = p, R = R, m = m, g = g, C_D = C_D,
                        k = k, I_cm = I_cm):  

    import time
    
    ### check if folder already exists!
    folDict = get_folder_struct('trajectories_stored')
    
    t0 = time.time()

    v_x, v_y, rpm, height = initialize(dtype = dtype_calc, squeeze = False,
                                       granularity_initial_ground = granularity_initial_ground, 
                                       granularity_initial_vert = granularity_initial_vert, 
                                       granularity_spin = granularity_spin,
                                       min_speed_ground = min_speed_ground, max_speed_ground = max_speed_ground,
                                       min_speed_vert = min_speed_vert, max_speed_vert = max_speed_vert,
                                       min_spin_rate = min_spin_rate, 
                                       max_spin_rate = max_spin_rate)
    dimensions = v_x.shape[0], v_y.shape[1], rpm.shape[2], height.shape[3]
    
    initial_x = np.zeros(dimensions, dtype = dtype_calc)
    initial_y = np.ones(dimensions, dtype = dtype_calc) * height
    initial_rpm = np.ones(dimensions, dtype = dtype_calc) * rpm
    
    xList = [initial_x] ### this thing will be (v_x, v_y, RPM, t)
    zList = [initial_y] ### this thing will be (v_x, v_y, RPM, t)
    rpmList = [initial_rpm]
    for frames in range(0, int(duration_of_trajectory / step_duration)):
        
        ### Iterate Location
        x = xList[-1] + v_x * step_duration
        z = zList[-1] + v_y * step_duration
        rpm = rpmList[-1]
        
        ### Calculate New Speeds and Spins
        # a) in flight
        v_x_fly, v_y_fly, spin_rates_fly = calculate_speed_step(v_x, v_y, rpm,
                                                                step_duration = step_duration,
                                                                k = k, C_D = C_D, g = g)
        
        # b) for bounce
        e_y, tan_theta, D = calculate_bounce_parameters(v_x, v_y, rpm)
        
        # b1) sliding bounce
        v_x_sld, v_y_sld, spin_rates_sld = calculate_sliding_bounce_step(v_x, v_y, rpm,
                                                                         e_y, tan_theta, D,
                                                                         mu = mu, R = R, alpha = alpha)
        # b2) gripping bounce
        v_x_grp, v_y_grp, spin_rates_grp = calculate_gripping_bounce_step(v_x, v_y, rpm,
                                                                          e_y, tan_theta, D, 
                                                                          e_x = e_x, R = R, alpha = alpha)
        
        ### Distinguish between bounces and flight
        z_lower_bound = np.clip(z, a_min = 0, a_max = None)
        is_bouncing = z_lower_bound == 0
        
        ### Distinguish between a sliding bounce and a gripping bounce
        is_sliding = R * RPM_to_omega(spin_rates_sld) < v_x_sld
        
        v_x = np.where(is_bouncing & is_sliding, v_x_sld, 
                       np.where(is_bouncing & ~is_sliding, v_x_grp,
                                v_x_fly))
        v_y = np.where(is_bouncing & is_sliding, v_y_sld, 
                       np.where(is_bouncing & ~is_sliding, v_y_grp,
                                v_y_fly))
        rpm = np.where(is_bouncing & is_sliding, spin_rates_sld, 
                       np.where(is_bouncing & ~is_sliding, spin_rates_grp,
                                spin_rates_fly))
        
        ### Add to Trajectories
        xList.append(x)
        zList.append(z_lower_bound)
        rpmList.append(rpm)
        
    xArr = np.stack(xList, axis = -1).astype(dtype_save)
    zArr = np.stack(zList, axis = -1).astype(dtype_save)
    rpmArr = np.stack(rpmList, axis = -1).astype(dtype_save)

    t1 = time.time()
    seconds_elapsed = f'{round(t1 - t0)} seconds'
    print(f'Done! Took {seconds_elapsed}')
    
    if name in folDict:
        print(f'Trajectories {name} exist. Will not overwrite.')
    else:
        save_trajectories(name, xArr, zArr, rpmArr,
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

    return xArr, zArr, rpmArr

def trajectory_to_3D(groundArr_opt, vertArr_opt, flightLn):
    groundArr_resc = groundArr_opt * const.SCALE / const.FOOT_TO_METER
    vertArr_resc = vertArr_opt * const.SCALE / const.FOOT_TO_METER
    
    st_x, st_y = flightLn.leftPt.coords[:2]
    fl_x, fl_y = flightLn.to_vector().coords[:2]
    
    lmbd = groundArr_resc / flightLn.length()
    
    x, y = st_x + lmbd * fl_x, st_y + lmbd * fl_y
    z = vertArr_resc
    
    return np.stack([x, y, z], axis = 1)
    

#%%############ SPIN ESTIMATION  ############

import ATMANpy.auxil.constants as const
import ATMANpy.models.HMM.HPost as HPost

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

def calculate_planar_ball_flight_vectorised(coordArr, add_in_initial_heights = True):
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
    
    vert_cum_dist_meters = vert_dist_meters.cumsum(axis = 1) ### add up all distances between two values
    # vert_cum_dist_meters = shift_array(vert_cum_dist_meters, 1, axis = 0, fill_value = 0) ### move first differences to the right, and add in zeros
    
    if add_in_initial_heights:
        initial_heights = transArr_nonnan[2, 0] / const.SCALE * const.FOOT_TO_METER
        vert_cum_dist_meters = vert_cum_dist_meters + initial_heights[np.newaxis, :]
    
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
    
def get_optimal_trajectory_indices_vectorised(groundArr_height, vertArr_height, 
                                              pos_ground_measured_h, pos_vert_measured_h,
                                              sum_of_squares = True):
    ### shape groundArr_height, vertArr_height: v_ground x v_vert x spin_rpm x time
    ### shape pos_ground_measured_h, pos_vert_measured_h: dim1 x dim2 x ... x t
    measured_shape = pos_ground_measured_h.shape
    extra_dims, num_frames = measured_shape[:-1], measured_shape[-1]
    
    pos_ground_theoretical_h = groundArr_height[:, :, :, :num_frames]
    pos_vert_theoretical_h = vertArr_height[:, :, :, :num_frames]
    
    for _ in extra_dims:
        pos_ground_theoretical_h = np.expand_dims(pos_ground_theoretical_h, axis = -2)
        pos_vert_theoretical_h = np.expand_dims(pos_vert_theoretical_h, axis = -2)
    
    for _ in range(3): ### initial condition space
        pos_ground_measured_h = np.expand_dims(pos_ground_measured_h, axis = 0)
        pos_vert_measured_h = np.expand_dims(pos_vert_measured_h, axis = 0)
    
    ground_dev = pos_ground_measured_h - pos_ground_theoretical_h
    vert_dev = pos_vert_measured_h - pos_vert_theoretical_h
    
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
        traj_conds.append((traj_ground_idx, traj_vert_idx, traj_rpm_idx, traj_best_dev))
    
    return np.array(traj_conds)
    
def get_locally_minimal_trajectory_indices_vectorised(groundArr_height, vertArr_height, 
                                                      pos_ground_measured_h, pos_vert_measured_h,
                                                      sum_of_squares = True, number_of_local_minima = 15):
    rng = np.random.default_rng()
    
    ### shape groundArr_height, vertArr_height: v_ground x v_vert x spin_rpm x time
    ### shape pos_ground_measured_h, pos_vert_measured_h: dim1 x dim2 x ... x t
    measured_shape = pos_ground_measured_h.shape
    extra_dims, num_frames = measured_shape[:-1], measured_shape[-1]
    
    pos_ground_theoretical_h = groundArr_height[:, :, :, :num_frames]
    pos_vert_theoretical_h = vertArr_height[:, :, :, :num_frames]
    
    for _ in extra_dims:
        pos_ground_theoretical_h = np.expand_dims(pos_ground_theoretical_h, axis = -2)
        pos_vert_theoretical_h = np.expand_dims(pos_vert_theoretical_h, axis = -2)
    
    for _ in range(3): ### initial condition space
        pos_ground_measured_h = np.expand_dims(pos_ground_measured_h, axis = 0)
        pos_vert_measured_h = np.expand_dims(pos_vert_measured_h, axis = 0)
    
    ground_dev = pos_ground_measured_h - pos_ground_theoretical_h
    vert_dev = pos_vert_measured_h - pos_vert_theoretical_h
    
    if sum_of_squares:
        my_dev = (ground_dev ** 2 + vert_dev ** 2) ** 0.5
    else:
        my_dev = np.abs(ground_dev) + np.abs(vert_dev)
        
    sum_dev = np.nansum(my_dev, axis = -1) ### shape: v_ground, v_vert, rpm, traj
    sum_dev = np.moveaxis(sum_dev, -1, 0) ### shape: traj, v_ground, v_vert, rpm
    
    local_minima = [detect_local_minima(traj_sum_dev) for traj_sum_dev in sum_dev] ### shape: trajs, x_idxs, z_idxs, rpm_idxs
    
    traj_arrays = []
    for traj_idx, local_minimum in enumerate(local_minima):
        ### gather the local minima in initial condition space for a given trajectory
        x_idxs, z_idxs, rpm_idxs = local_minimum
        traj_devs = sum_dev[traj_idx, x_idxs, z_idxs, rpm_idxs]
        
        ### resample local minima
        sample_ps = 1 / traj_devs
        sample_ps_norm = sample_ps / sample_ps.sum()
        loc_min_idxs = np.arange(len(traj_devs))
        
        if len(loc_min_idxs) < number_of_local_minima: ### in this case, get all of the minima and pad the array to fill the required local minima
            chosen_loc_min_idxs = np.tile(loc_min_idxs, number_of_local_minima)[:number_of_local_minima]
        else:
            chosen_loc_min_idxs = rng.choice(loc_min_idxs, size = number_of_local_minima, replace = False, p = sample_ps_norm)
        
        traj_ground_idxs = x_idxs[chosen_loc_min_idxs]
        traj_vert_idxs = z_idxs[chosen_loc_min_idxs]
        traj_rpm_idxs = rpm_idxs[chosen_loc_min_idxs]
        traj_devs_chosen = traj_devs[chosen_loc_min_idxs]
    
        traj_array = np.stack([traj_ground_idxs, traj_vert_idxs, traj_rpm_idxs, traj_devs_chosen], axis = 1)
        traj_arrays.append(traj_array)
        
    return np.stack(traj_arrays)
    
    
def get_locally_minimal_trajectory_indices(groundArr, vertArr, 
                                           pos_ground_measured, pos_vert_measured,
                                           sum_of_squares = True, number_of_local_minima = 15):
    rng = np.random.default_rng()
    
    ### shape groundArr_height, vertArr_height: v_ground x v_vert x spin_rpm x time
    ### shape pos_ground_measured_h, pos_vert_measured_h: dim1 x dim2 x ... x t
    num_trajs, num_frames = pos_ground_measured.shape
    
    pos_ground_theoretical = groundArr[:, :, :, :, :num_frames]
    pos_vert_theoretical = vertArr[:, :, :, :, :num_frames]
    
    pos_ground_theoretical_brdcst = pos_ground_theoretical[np.newaxis]
    pos_vert_theoretical_brdcst = pos_vert_theoretical[np.newaxis]
    
    pos_ground_measured_brdcst = pos_ground_measured.copy()
    pos_vert_measured_brdcst = pos_vert_measured.copy()
    for _ in range(4):
        pos_ground_measured_brdcst = pos_ground_measured_brdcst[:, np.newaxis]
        pos_vert_measured_brdcst = pos_vert_measured_brdcst[:, np.newaxis]
    
    ground_dev = pos_ground_measured_brdcst - pos_ground_theoretical_brdcst
    vert_dev = pos_vert_measured_brdcst - pos_vert_theoretical_brdcst
    
    sum_dev = np.nansum((ground_dev ** 2 + vert_dev ** 2) ** 0.5, axis = -1)
    del ground_dev, vert_dev
    
    local_minima = [detect_local_minima(traj_sum_dev) for traj_sum_dev in sum_dev] ### shape: trajs, x_idxs, z_idxs, rpm_idxs, height_idxs
    
    traj_arrays = []
    for traj_dev, local_minimum in zip(sum_dev, local_minima):
        ### gather the local minima in initial condition space for a given trajectory
        x_idxs, z_idxs, rpm_idxs, height_idxs = local_minimum
        min_devs = traj_dev[x_idxs, z_idxs, rpm_idxs, height_idxs]
        
        ### resample local minima
        sample_ps = 1 / min_devs
        sample_ps_norm = sample_ps / sample_ps.sum()
        loc_min_idxs = np.arange(len(min_devs))
        
        if len(loc_min_idxs) < number_of_local_minima: ### in this case, get all of the minima and pad the array to fill the required local minima
            chosen_loc_min_idxs = np.tile(loc_min_idxs, number_of_local_minima)[:number_of_local_minima]
        else:
            chosen_loc_min_idxs = rng.choice(loc_min_idxs, size = number_of_local_minima, replace = False, p = sample_ps_norm)
        
        traj_ground_idxs = x_idxs[chosen_loc_min_idxs]
        traj_vert_idxs = z_idxs[chosen_loc_min_idxs]
        traj_rpm_idxs = rpm_idxs[chosen_loc_min_idxs]
        traj_height_idxs = height_idxs[chosen_loc_min_idxs]
        traj_devs_chosen = min_devs[chosen_loc_min_idxs]
    
        traj_array = np.stack([traj_ground_idxs, traj_vert_idxs, traj_rpm_idxs, traj_height_idxs, traj_devs_chosen], axis = 1)
        traj_arrays.append(traj_array)
   
    return np.stack(traj_arrays)
    



def get_optimal_trajectory_indices_vectorised2(groundArr_brdcst, vertArr_brdcst, 
                                              pos_ground_measured, pos_vert_measured,
                                              sum_of_squares = True):
    ### shape groundArr, vertArr: v_ground x v_vert x spin_rpm x dim x time
    ### shape pos_ground_measured, pos_vert_measured: dim x time
    measured_shape = pos_ground_measured.shape
    extra_dims, num_frames = measured_shape[:-1], measured_shape[-1]
    
    pos_ground_theoretical = groundArr_brdcst[:, :, :, :, :num_frames]
    pos_vert_theoretical = vertArr_brdcst[:, :, :, :, :num_frames]
    
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
        traj_conds.append((traj_ground_idx, traj_vert_idx, traj_rpm_idx, traj_best_dev))
    
    return np.array(traj_conds)
    


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
    
    xArr, zArr, rpmArr = create_trajectories(NAME, 
                                             dtype_calc = 'float32', dtype_save = 'float16',
                                             step_duration = STEP_DURATION)
    
    ########## TESTING ##########
    initial_v_x, initial_v_y, initial_spins, initial_heights = initialize(squeeze = True)
    def get_match_idx(arr, val):
        return np.argmin(np.abs(arr - val))
    def limit_to_threshold(my_xArr, my_yArr, threshold = -1):
        filt = my_yArr >= threshold
        return my_xArr[filt], my_yArr[filt]
    
    
    ##### Replicate Fig. 6 in Rod Cross Paper
    spd0, ang0, rpm0, height0 = 30, 7.4, 0, 1
    spd1, ang1, rpm1, height1 = 30, 15.2, 4000, 1
    
    vx0, vy0 = spd0 * np.cos(ang0 * np.pi / 180), spd0 * np.sin(ang0 * np.pi / 180) 
    vx1, vy1 = spd1 * np.cos(ang1 * np.pi / 180), spd1 * np.sin(ang1 * np.pi / 180) 
    
    vx0_idx, vy0_idx = get_match_idx(initial_v_x, vx0), get_match_idx(initial_v_y, vy0)
    vx1_idx, vy1_idx = get_match_idx(initial_v_x, vx1), get_match_idx(initial_v_y, vy1)
    rpm0_idx, rpm1_idx = get_match_idx(initial_spins, rpm0), get_match_idx(initial_spins, rpm1)
    height0_idx, height1_idx = get_match_idx(initial_heights, height0), get_match_idx(initial_heights, height1)
    
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([0, 30])
    
    xArr0, zArr0 = xArr[vx0_idx, vy0_idx, rpm0_idx, height0_idx], zArr[vx0_idx, vy0_idx, rpm0_idx, height0_idx]
    xArr1, zArr1 = xArr[vx1_idx, vy1_idx, rpm1_idx, height1_idx], zArr[vx1_idx, vy1_idx, rpm1_idx, height1_idx]
    
    ax.plot(xArr0, zArr0, color = 'black')
    ax.plot(xArr1, zArr1, color = 'red')
    
    #### Replicate Fig. 7 
    
    
    
    
    
    # #### Visualize Spin
    # ground_speed = 30
    # vert_speed = 2
    # spins = np.array([-4000, 0, 4000])
    
    # initial_v_x_idx = np.argmin(np.abs(initial_v_x - ground_speed))
    # initial_v_y_idx = np.argmin(np.abs(initial_v_y - vert_speed))
    
    # xArr_chk, zArr_chk = xArr[initial_v_x_idx, initial_v_y_idx], zArr[initial_v_x_idx, initial_v_y_idx]
    
    # spin_idxs = []
    # for spin_devs in np.abs(spin_rate_space[np.newaxis, :] - spins[:, np.newaxis]):
    #     spin_idxs.append(np.argmin(spin_devs))
    
    # y_threshold = -1
    # fig, ax = plt.subplots(1, 1)
    # # ax.set_ylim(bottom = y_threshold, top = None)
    
    # for spin_idx in spin_idxs:
    #     xArr_plot, zArr_plot = xArr_chk[spin_idx], zArr_chk[spin_idx]
    #     xArr_plot, zArr_plot = xArr_plot[zArr_plot > y_threshold], zArr_plot[zArr_plot > y_threshold]
    
    #     ax.plot(xArr_plot, zArr_plot)
        


    
    
    
    
        