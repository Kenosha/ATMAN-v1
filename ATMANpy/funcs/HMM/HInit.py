# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 02:12:26 2022

@author: gianl
"""

import pandas as pd
import numpy as np

def segment_one_word(word, num_states = 3):
    
    out_word = word.copy()
    
    ### subdivide the word evenly into states
    state_allocation = []
    
    remaining_length = len(out_word)
    remaining_states = num_states
    for state in range(1, num_states + 1): ### emitting states start at 1!
        
        frames_per_state = round(remaining_length / remaining_states)
        stateList = [state] * frames_per_state
        
        remaining_length -= frames_per_state
        remaining_states -= 1
        
        state_allocation.extend(stateList)
        
    ### add the info to the dataframe
    out_word['State'] = state_allocation  
    
    ### and plug it into the index
    out_word = out_word.reset_index().set_index([*out_word.index.names, 'State'])
    
    return out_word

def find_transition_occurences(word, num_states = 3):
    
    out_word = word.copy()
    
    try:
        stateArr = out_word.index.get_level_values('State')
    except:
        stateArr = out_word['State']
        
    stateArr = [0, *stateArr, num_states + 1] ### add in entry (0) and exit (NUM_STATES + 1) state.
    
    from_state = stateArr[:-1]
    to_state = stateArr[1:]
    state_changes = np.array([from_state, to_state]).T ### -> shape: WORD_LENGTH (= TRANSITIONS) x 2 
    
    unique, counts = np.unique(state_changes, return_counts = True, axis = 0)
    
    transMat = np.zeros([num_states + 2, num_states + 2], dtype = int)
    
    for from_to_pair, number_of_transitions in zip(unique, counts):
        i, j = from_to_pair
        transMat[i, j] = number_of_transitions
    
    return transMat
    
def find_transition_matrix(words):
    
    sequences = words.index.get_level_values('IndexSF').unique()

    transMats = np.array([find_transition_occurences(words.loc[idx]) for idx in sequences])    
    all_transMat = transMats.sum(axis = 0)
    
    rowSum = all_transMat.sum(axis = 1)
    A = all_transMat / rowSum[:, np.newaxis] ### normalize by row sum
    
    ### it can happen that the row sum is zero -> if the state is never occupied non-terminally
    ### always happens with the exit state, btw -> there are no transitions away from it.
    A = np.nan_to_num(A)    
    
    return A 
    
def get_parameters(words):
    A = find_transition_matrix(words)
    
    avgs = words.groupby('IndexSF').apply(lambda x: x.groupby('State').mean())
    covs = words.groupby('IndexSF').apply(lambda x: x.groupby('State').cov())
    
    avg = avgs.groupby('State', sort=True).mean() ### State we want sorted, from left-to-right
    cov = covs.groupby(['State', 'Var'], sort=False).mean() ### to prevent the variable names to be sorted
    
    ### now, to ensure that state is also sorted here... 
    # cov = cov.sort_index(level = 'State') ## buggy: also sorts Var...
    cov = pd.concat([cov.loc[[state]] for state in avg.index.get_level_values('State').unique()])
    
    return A, avg, cov

def resolve_viterbi_grid(grid, final_state):
    grid_backwards = reversed(grid) ### first item belongs to the second-to-last state that was occupied
                                    ### last belongs to the first state that was occupied
                                    ### the elements express the likeliest move 
                                    ### an element of [1, 1, 3] says that 
                                        # if we landed in state 1, we likely came from state 1.
                                        # if we landed in state 2, we likely came from state 1.
                                        # if we landed in state 3, we likely came from state 3.
                                        
    stateList = [final_state]
    for likeliest_move in grid_backwards:
        landed_in = stateList[-1]
        came_from = likeliest_move[landed_in]
        stateList.append(came_from)
    
    return list(reversed(stateList))
        
        

def viterbi_segment(densFrm, A):
    sequences = densFrm.index.get_level_values('IndexSF').unique()
    
    ### Get the log quantities ###
    log_densFrm = np.log(densFrm)
    
    logA = np.log(A)
    log_transitions_from_initial = logA[0, 1:-1]
    log_transitions_within_states = logA[1:-1, 1:-1]
    log_transition_to_exit = logA[1:-1, -1]
    
    seq = sequences[0]
    seqList = []
    max_vals = []
    for seq in sequences:
        
        logFrm = log_densFrm.loc[[seq]]
        logArr = logFrm.values ### Shape: N x S
        
        ### initialize
        log_density_initial, log_densities = logArr[0], logArr[1:]
        phi = np.array([log_transitions_from_initial + log_density_initial]) ### Shape: 1 x S
        stateArr = []
        
        ### go through recursive algo
        for log_density in log_densities:
            last_phi_broadcast = phi[-1][:, np.newaxis] ### so the phi can be added to the log transitions.
            viterbi_step = last_phi_broadcast + log_transitions_within_states
            max_state, max_val = viterbi_step.argmax(axis = 0), viterbi_step.max(axis = 0)
            
            new_phi = max_val + log_density
            
            phi = np.append(phi, [new_phi], axis = 0)
            stateArr.append(max_state)
        
        ### last step -> movement into exit state!
        viterbi_step = phi[-1] + log_transition_to_exit
        max_state, max_val = viterbi_step.argmax(), viterbi_step.max()
        
        ### resolve the grid backward by looking at final state, and then finding second-to-last, etc.
        stateList = resolve_viterbi_grid(stateArr, max_state)
        
        seqList.extend(stateList)
        max_vals.append(max_val)
        
    outFrm = log_densFrm.index.to_frame(index = False)
    outFrm['State'] = seqList
    
    ### turn the statelist back into a one indexed thing -> the zeroth index is the entry state!!!
    outFrm['State'] += 1    
        
    return outFrm, sum(max_vals)

def HInit(words):
        
    from ATMANpy.funcs.HMM.utils import stop_condition, calculate_emission_densities
    
    ### Initialize a segmentation by evenly distributing word across states
    words = words.groupby('IndexSF', group_keys = False).apply(segment_one_word)
    
    curr_log_likelihood = None
    prev_log_likelihood = None
    best_log_likelihood = None
    num_iterations = 1
    while not stop_condition(curr_log_likelihood, prev_log_likelihood, num_iterations):
        
        prev_log_likelihood = curr_log_likelihood
        
        ### Recalculate parameters
        A, avg, cov = get_parameters(words)
        
        ### With new parameters, new densities
        densFrm = calculate_emission_densities(words, avg, cov)
        
        words.index = words.index.droplevel('State') ### we drop the states -> they will now be recalculated.
        state_sequences, curr_log_likelihood = viterbi_segment(densFrm, A)    
        words.index = pd.MultiIndex.from_frame(state_sequences)
    
        ### Save the parameters of the best iteration so far
        if best_log_likelihood is None:
            best_A, best_avg, best_cov = A, avg, cov
            best_log_likelihood = curr_log_likelihood
        else:
            if curr_log_likelihood > best_log_likelihood:
                best_A, best_avg, best_cov = A, avg, cov
                best_log_likelihood = curr_log_likelihood
    
        num_iterations += 1
    
    else:
        print(f'Best Likelihood: {round(best_log_likelihood, 2)}')
        return best_A, best_avg, best_cov 

