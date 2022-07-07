# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 02:12:26 2022

@author: gianl
"""

import pandas as pd
import numpy as np


def get_word_data(dataFrm, wordStr):
    
    ### Prep Word Data
    word_instances = dataFrm[dataFrm.Event == wordStr].drop(columns = 'Event')
    word_instances = word_instances.reset_index().set_index(['IndexSF', 'Frame'])
    word_instances.columns.name = 'Var'
    
    return word_instances
         
def calculate_emission_densities(words, avg, cov):
    """
    for each frame in each word occurence, we need the likelihood
    of said frame being emitted when in any of the HMM states.
    """
    
    ### PREPARE SOME HELPER VARS
    states = avg.index.unique()
    num_states = len(states)
    num_vars = len(avg.columns)
    
    ### PREPARE ARRAYS ###
    wordArr = words.values ### shape: num_obs, num_vars
    avgArr = avg.values.T  ### shape: num_vars, num_states
    covArr = cov.values.reshape(num_states, num_vars, num_vars) ### shape: num_states, num_vars, num_vars
    dists = wordArr[:, :, np.newaxis] - avgArr[np.newaxis, :, :] ### shape: num_obs, num_vars, num_states
    
    invArr = []
    detArr = []
    for mat in covArr:
        invArr.append(np.linalg.inv(mat))
        detArr.append(np.linalg.det(mat))
    invArr = np.array(invArr) ### shape: num_states, num_vars, num_vars
    detArr = np.array(detArr) ### shape: num_states
    
    ### CALCULATE NORMAL DENSITY LIKELIHOOD
    # a) Numerator
    trans_vec_mat_mulStr = 'nis, sij -> njs'
    dist_x_cov_inv = np.einsum(trans_vec_mat_mulStr, dists, invArr)
    
    vec_vec_mulStr = 'njs, njs -> ns'
    exp_term = np.einsum(vec_vec_mulStr,dist_x_cov_inv, dists)
    numerator = np.exp(-0.5 * exp_term) ### shape: num_obs, num_states
    
    # b) Denominator
    denominator = (((2 * np.pi) ** num_vars) * detArr) ** 0.5 ### shape: num_states
    
    # c) Result
    density = numerator / denominator[np.newaxis, :] ### shape: num_obs, num_states
    
    densityFrm = pd.DataFrame(density, index = words.index, columns = states)
    
    return densityFrm
    
def stop_condition(curr_log_likelihood, prev_log_likelihood, 
                   num_iterations, 
                   relative_change_threshold = 0.0001, 
                   min_iterations = 15, max_iterations = 100):
    try:
        round_likelihood = round(curr_log_likelihood, 2)
    except:
        round_likelihood = None
        
    print(f'Iteration {num_iterations}. Current likelihood: {round_likelihood}...')
    if num_iterations < min_iterations:
        # print(f'    ...Minimum of {min_iterations} not reached!')
        return False
    elif num_iterations > max_iterations:
        print(f'    ...Stopping: {max_iterations} iterations exceeded.')
        return True
    else:
        if num_iterations is None or prev_log_likelihood is None:
            return False ### don't stop!
        else:
            relative_diff_likelihood = abs((curr_log_likelihood - prev_log_likelihood) / curr_log_likelihood)
            if relative_diff_likelihood < relative_change_threshold:
                print('    ...Stopping: Convergence achieved.')
                return True
            else:
                return False
         