# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:49:29 2023

@author: gianl
"""

import os
import numpy as np
from ATMANpy.funcs.util_funcs import get_folder_struct

### SEARCH SPACE PARAMETERS ###
GRANULARITY_INITIAL_GROUND = 325 ### how many different initial conditions in x
GRANULARITY_INITIAL_VERT = 200 ### how many different initial conditions in z
GRANULARITY_SPIN = 301 ### how many different spin rates
MIN_SPEED_GROUND = 0 ## lowest possible ground speed of ball, in m/s
MAX_SPEED_GROUND = 75 ## highest possible ground speed of ball, in m/s
MIN_SPEED_VERT = -20
MAX_SPEED_VERT =  20
MIN_SPIN_RATE = -5000 ### highest topspin RPM (negative, because topspin causes negative lift)  
MAX_SPIN_RATE = 5000  ### higgest slice RPM (positive, because slice causes positive lift)
DURATION_OF_TRAJECTORY = 4 ### in seconds

### VIDEO PARAMETERS ####
FPS = 25
STEP_DURATION = 1 / FPS

### PHYSICAL PARAMETERS ####
p = 1.21 ### air density, in kg / m ^ 3
R = 66 / 10**4 ### radius of tennis ball in meters (66mm = 66m * 10**-4) 
m = 58 / 10**4 ### mass of tennis ball in kg (58 g = 58 kg * 10**-4)
g = 9.78 ### gravity of earth, in m/s**2

C_D = 0.51 ### drag coefficient
C_L_COEFFICIENT = 0.55
###################

k = 0.5 * p * np.pi * R**2 / m ### unit 1 / m

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
               max_spin_rate = MAX_SPIN_RATE):
        
    initial_v_x = np.linspace(min_speed_ground, max_speed_ground, granularity_initial_ground, dtype = dtype) # initial condition: horizontal speed
    initial_v_z = np.linspace(min_speed_vert, max_speed_vert, granularity_initial_vert, dtype = dtype) # initial condition: vertical speed
    spin_rate_space = np.linspace(min_spin_rate, max_spin_rate, granularity_spin, dtype = dtype)
    
    ### add pathological tails to spin rate space
    avg_spin_rate = (min_spin_rate + max_spin_rate) / 2
    pathological_lo_spin_rate = (min_spin_rate - avg_spin_rate) * 10
    pathological_hi_spin_rate = (max_spin_rate - avg_spin_rate) * 10
    one_tail_granularity = int(granularity_spin / 20) ### 5% extra space on each tail -> 10% extra total
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
    
    a_x = - kv * (C_D * v_x + lift_coeffs * v_z) 
    a_z = kv * (lift_coeffs * v_x - C_D * v_z) - g
    
    return v_x + a_x * STEP_DURATION, v_z + a_z * STEP_DURATION

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
      
    if name in folDict:
        print(f'Trajectories {name} exist. Will not overwrite.')
        return None, None

    t0 = time.time()

    k = 0.5 * p * np.pi * R**2 / m ### unit 1 / m
    
    v_x, v_z, lift_coeffs = initialize(dtype = dtype_calc, 
                                       granularity_initial_ground = granularity_initial_ground, 
                                       granularity_initial_vert = granularity_initial_vert, 
                                       granularity_spin = granularity_spin,
                                       min_speed_ground = min_speed_ground, max_speed_ground = max_speed_ground,
                                       min_speed_vert = min_speed_vert, max_speed_vert = max_speed_vert,
                                       min_spin_rate = min_spin_rate, 
                                       max_spin_rate = max_spin_rate)
    dimensions = v_x.shape[0], v_z.shape[1], lift_coeffs.shape[2]  
    
    xList = [np.zeros(dimensions, dtype = dtype_calc)] ### this thing will be (v_x, v_y, G_L, t)
    zList = [np.zeros(dimensions, dtype = dtype_calc)] ### this thing will be (v_x, v_y, G_L, t)
    for frames in range(0, duration_of_trajectory * FPS):
        
        x = xList[-1] + v_x * step_duration
        z = zList[-1] + v_z * step_duration
        
        xList.append(x)
        zList.append(z)
        
        v_x, v_z = calculate_speed_step(v_x, v_z, lift_coeffs,
                                        step_duration = step_duration,
                                        k = k, C_D = C_D, g = g)
        
    xArr = np.stack(xList, axis = -1).astype(dtype_save)
    zArr = np.stack(zList, axis = -1).astype(dtype_save)

    t1 = time.time()
    seconds_elapsed = f'{round(t1 - t0)} seconds'
    print(f'Done! Took {seconds_elapsed}')
    
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
    xDim, yDim, zDim = trajFrm.columns
    ground_dist = HPost.calc_ball_dist(trajFrm[[xDim, yDim]], norm = 2) ### in points
    ground_dist_meters = ground_dist / const.SCALE * const.FOOT_TO_METER ### in meters
    ground_dist_meters = ground_dist_meters.cumsum().shift(1).fillna(0) ### normalised, so that at hit, m = 0

    vert_dist = HPost.calc_ball_dist(trajFrm[[zDim]], norm = 1)
    vert_dist_meters = vert_dist / const.SCALE * const.FOOT_TO_METER ### in meters
    vert_dist_meters = vert_dist_meters.cumsum().shift(1).fillna(0) ### normalised, so that at hit, m = 0

    return ground_dist_meters.values, vert_dist_meters.values
    
def get_optimal_trajectory_indices(groundArr, vertArr, 
                                   pos_ground_measured, pos_vert_measured,
                                   sum_of_squares = True):
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
        
    sum_dev = my_dev.sum(axis = 3)
    best_idx_flat = sum_dev.argmin()
    
    v_ground_idx, v_vert_idx, rpm_idx = np.unravel_index(best_idx_flat, sum_dev.shape)
    
    return v_ground_idx, v_vert_idx, rpm_idx


#%% MAIN ##
if __name__ == '__main__':
    
    # for dtype in ['float64', 'float32', 'float16']:
    #     t = time.time()
    #     xArr, zArr = create_trajectories(f'TestTrajectories_{dtype}', 
    #                                      dtype_calc = dtype, dtype_save = dtype)
    #     print(f'Done with {dtype}! Took {time.time() - t} seconds')
        
    # #### NOTE HOW FLOAT16 TAKES LONGEST (APPARENTLY NO HARDWARE SUPPORT...)
    
    xArr, zArr = create_trajectories(f'TrajectoriesRPM', 
                                     dtype_calc = 'float32', dtype_save = 'float16')

    
    
    
    
        