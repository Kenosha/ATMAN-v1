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
GRANULARITY_LIFT = 301 ### how many different lift coeffs
MIN_SPEED_GROUND = 0 ## lowest possible ground speed of ball, in m/s
MAX_SPEED_GROUND = 75 ## highest possible ground speed of ball, in m/s
MIN_SPEED_VERT = -20
MAX_SPEED_VERT =  20
MIN_LIFT_COEFF = -1  
MAX_LIFT_COEFF = 1  
DURATION_OF_TRAJECTORY = 6 ### in seconds

### VIDEO PARAMETERS ####
FPS = 25
STEP_DURATION = 1 / FPS

### PHYSICAL PARAMETERS ####
p = 1.21 ### air density, in kg / mm ^ 3
R = 66 / 10**4 ### radius of tennis ball in meters (66mm * 10**-4) 
m = 58 / 10**4 ### mass of tennis ball in kg (58 g * 10**-4)
g = 9.78 ### gravity of earth, in m/s**2

C_D = 0.51 ### drag coefficient
C_L_COEFFICIENT = 0.55
###################

k = 0.5 * p * np.pi * R**2 / m ### unit 1 / m

def initialize(dtype = 'float64', squeeze = False,
               granularity_initial_ground = GRANULARITY_INITIAL_GROUND, 
               granularity_initial_vert = GRANULARITY_INITIAL_VERT, 
               granularity_lift = GRANULARITY_LIFT,
               min_speed_ground = MIN_SPEED_GROUND, max_speed_ground = MAX_SPEED_GROUND,
               min_speed_vert = MIN_SPEED_VERT, max_speed_vert = MAX_SPEED_VERT,
               min_lift_coeff = MIN_LIFT_COEFF, 
               max_lift_coeff = MAX_LIFT_COEFF):
        
    initial_v_x = np.linspace(min_speed_ground, max_speed_ground, granularity_initial_ground, dtype = dtype) # initial condition: horizontal speed
    initial_v_z = np.linspace(min_speed_vert, max_speed_vert, granularity_initial_vert, dtype = dtype) # initial condition: vertical speed
    lift_coeffs_space = np.linspace(min_lift_coeff, max_lift_coeff, granularity_lift, dtype = dtype)
    
    if squeeze:
        return initial_v_x, initial_v_z, lift_coeffs_space
    else:
        v_x, v_z = initial_v_x[:, np.newaxis, np.newaxis], initial_v_z[np.newaxis, :, np.newaxis]
        lift_coeffs = lift_coeffs_space[np.newaxis, np.newaxis, :]
        return v_x, v_z, lift_coeffs

def calculate_speed_step(v_x, v_z, lift_coeffs,
                         step_duration = STEP_DURATION,
                         k = k, C_D = C_D, g = g):
    ### using the trajectory differential equations, calculate the speed in the next second
    ### we are working with matrices of the shape v_x, v_y, C_L
    v = (v_x ** 2 + v_z ** 2) ** 0.5
    kv = k * v
    
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
                        granularity_lift = GRANULARITY_LIFT,
                        min_speed_ground = MIN_SPEED_GROUND, max_speed_ground = MAX_SPEED_GROUND,
                        min_speed_vert = MIN_SPEED_VERT, max_speed_vert = MAX_SPEED_VERT,
                        min_lift_coeff = MIN_LIFT_COEFF, max_lift_coeff = MAX_LIFT_COEFF,
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
                                       granularity_lift = granularity_lift,
                                       min_speed_ground = min_speed_ground, max_speed_ground = max_speed_ground,
                                       min_speed_vert = min_speed_vert, max_speed_vert = max_speed_vert,
                                       min_lift_coeff = min_lift_coeff, 
                                       max_lift_coeff = max_lift_coeff)
      
    xList = [np.zeros((granularity_initial_ground, granularity_initial_vert, granularity_lift), dtype = dtype_calc)] ### this thing will be (v_x, v_y, G_L, t)
    zList = [np.zeros((granularity_initial_ground, granularity_initial_vert, granularity_lift), dtype = dtype_calc)] ### this thing will be (v_x, v_y, G_L, t)
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
                      granularity_lift = granularity_lift,
                      min_speed_ground = min_speed_ground, max_speed_ground = max_speed_ground,
                      min_speed_vert = min_speed_vert, max_speed_vert = max_speed_vert,
                      min_lift_coeff = min_lift_coeff, 
                      max_lift_coeff = max_lift_coeff,
                      duration_of_trajectory = duration_of_trajectory,
                      step_duration = step_duration,
                      p = p, R = R, m = m, g = g, C_D = C_D,
                      time_taken = seconds_elapsed)

    return xArr, zArr

def lift_coeff_to_RPM(lift_coeff, ball_speed, 
                            C_L_coefficient = C_L_COEFFICIENT, R = R):
    S = lift_coeff / C_L_coefficient ### dimensionless ratio
    R_w = S * ball_speed ### meters per second (or whatever the unit of ball_speed is!)
    RPM = R_w / (2 * np.pi * R) * 60 ### revolutions per minute!
    
    return RPM


if __name__ == '__main__':
    
    # for dtype in ['float64', 'float32', 'float16']:
    #     t = time.time()
    #     xArr, zArr = create_trajectories(f'TestTrajectories_{dtype}', 
    #                                      dtype_calc = dtype, dtype_save = dtype)
    #     print(f'Done with {dtype}! Took {time.time() - t} seconds')
        
    # #### NOTE HOW FLOAT16 TAKES LONGEST (APPARENTLY NO HARDWARE SUPPORT...)
    
    xArr, zArr = create_trajectories(f'Trajectories2', 
                                     dtype_calc = 'float32', dtype_save = 'float16')

    
    
    
    
        