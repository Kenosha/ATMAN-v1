# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:30:11 2022

@author: gianl
"""


import os
import pandas as pd
import numpy as np

from ATMANpy.funcs.util_funcs import get_folder_struct


VIDEO_ID = 'pqxAmUWIoKc'
DATA_SET = 'events_final.csv'

hmmPath = get_folder_struct('ball_event_hmm')[VIDEO_ID]
dataPath = os.path.join(hmmPath, 'hmm_data_final')
dataFile = os.path.join(dataPath, DATA_SET)

dataFrm = pd.read_csv(dataFile, index_col = 0).drop(columns = 'SeqStatus')
dataFrm.index.name = 'Frame'
dataFrm = dataFrm[~dataFrm.Event.isna()]
all_words = dataFrm.Event.unique()

#### Choose a subset of sequences for single word initialization and baum-welch training
seqIdxs = dataFrm.IndexSF.unique()
print(f'Available Sequences: {len(seqIdxs)}')

how_many_words_min = 4
how_many_words_rel = 0.1
extend_word = 1

def choose_random_sample_words(dataFrm):
    all_words = dataFrm.Event.unique()
    
    from ATMANpy.funcs.HMM.utils import get_word_data
    from ATMANpy.funcs.util_funcs import prepend_index_level
    
    sampleList = []
    for wordStr in all_words:
        word_dataFrm = get_word_data(dataFrm, wordStr, extend_word = extend_word)

        words_available = word_dataFrm.index.get_level_values('IndexWord').unique()
        number_of_words = len(words_available)
        
        how_many_words = max(how_many_words_min, round(how_many_words_rel * number_of_words))
        
        assert number_of_words >= how_many_words, f'Need more training data for {wordStr}!'

        word_choice_idxs = np.random.choice(words_available, size = how_many_words, replace = False)

        word_choice_data = word_dataFrm.loc[word_choice_idxs]
        
        word_choice_data = prepend_index_level(word_choice_data, 'Event', wordStr)

        sampleList.append(word_choice_data)
    
    return pd.concat(sampleList)

word_training_sample = choose_random_sample_words(dataFrm)

#### Assign a number of states to each word ####

# -> for the moment, assume they're all three state hmms, except None, which is one state.
event_to_num_states = lambda wordStr: 1 if wordStr == 'None' else 3

#### Begin with the single word functions HInit and HRest

np.seterr('ignore')

initDict = {}
restDict = {}
for wordStr in all_words:

    abs_min_threshold = 3 # how many frames do we need to train
    train_threshold = 7 # how many words do we need to train    
                         # number of variables is six, so we need at least seven words.

    stop_iteration_threshold_init = 10 ** -10   
    stop_iteration_threshold_rest = 10 ** -7
    min_iterations = 15 
    max_iterations = 50
    
    extend_word = 1
    num_states = event_to_num_states(wordStr)
    density_floor = 10 ** -100
    cov_global = False

    print(f'Doing {wordStr}...')
    
    words = word_training_sample.loc[wordStr]
    num_words = len(words.index.get_level_values("IndexWord").unique())
    num_frames = len(words)
    print(f'Number of words to train on: {num_words}, occupying {num_frames} frames')
    
    from ATMANpy.funcs.HMM.HInit import HInit
    from ATMANpy.funcs.HMM.HRest import HRest
                   
    if num_words < train_threshold:  ### in this case, we initialize the parameters 
                                        # with even segmentation and global variance
        A, avg, cov = HInit(words, num_states = num_states,
                            min_iterations = 0, max_iterations = 1, cov_global = True)
        ### this is one iteration of HInit, i.e, a even-segmentation initialization.
        initDict[wordStr] = {'A': A, 'avg': avg, 'cov': cov}
        restDict[wordStr] = {'A': A, 'avg': avg, 'cov': cov}
    
    else:
        initial_A, initial_avg, initial_cov = HInit(words, num_states = num_states, 
                                                    stop_iteration_threshold = stop_iteration_threshold_init,
                                                    min_iterations = min_iterations, 
                                                    max_iterations = max_iterations,
                                                    density_floor = density_floor)
        
        initDict[wordStr] = {'A': initial_A, 'avg': initial_avg, 'cov': initial_cov}
        print('')
        
        A, avg, cov = HRest(words, 
                            initial_A = initial_A, initial_avg = initial_avg, initial_cov = initial_cov,
                            stop_iteration_threshold = stop_iteration_threshold_rest,
                            min_iterations = min_iterations, max_iterations = max_iterations,
                            density_floor = density_floor)
        
        restDict[wordStr] = {'A': A, 'avg': avg, 'cov': cov}
        
        print('... Done!')
        print('______________')
    
#%% HERest

from copy import deepcopy
paramDict = deepcopy(restDict)

from ATMANpy.funcs.HMM.HERest import get_sequence_data_and_transcription, get_sequence_densities
from ATMANpy.funcs.HMM.HERest import get_forward_probs, get_backward_probs

from ATMANpy.funcs.HMM.HERest import eexp, eln, elnproduct, elnsum, LOGZERO
from ATMANpy.funcs.HMM.HERest import log_gammas_and_model_prob

from ATMANpy.funcs.HMM.HERest import log_xi_entry_numerator, log_xi_emit_numerator, log_xi_exit_numerator 

from ATMANpy.funcs.HMM.HRest import weighted_denominator
from ATMANpy.funcs.HMM.HRest import weighted_emission_cov_numerator, weighted_emission_mean_numerator

from ATMANpy.funcs.HMM.HRest import join_entry_emit_and_exit_to_A

NUM_CYCLES = 4

for cycleNum in range(1, NUM_CYCLES + 1):
    ### Initialize Accumulators
    ## Initialize Accumulators for gamma, which will feature in many denominators
    gammaAccumulator = {word: 0 for word in all_words}
    gammaAccumulator_entry = {word: 0 for word in all_words}
    
    ## Initialize Accumulators for mean and standard deviation numerators
    meanAccumulator = {word: 0 for word in all_words}
    covAccumulator = {word: 0 for word in all_words}
    
    ## Initialize Accumulators for xi, the numerator in transition probabilities
    xientryAccumulator = {word: 0 for word in all_words}
    xiemitAccumulator = {word: 0 for word in all_words}
    xiexitAccumulator = {word: 0 for word in all_words}
    
    ## Initialize Accumulator for log-likelihood
    lklAccumulator = 0
    
    for i, sequenceIdx in enumerate(seqIdxs):
        
        if i % 10 == 0:
            print(f'Embedded re-estimation of sequence {i} out of {len(seqIdxs)}...')
        
        words, eventSeq = get_sequence_data_and_transcription(dataFrm, sequenceIdx)
        
        seqDict = get_sequence_densities(paramDict, eventSeq, words)
        
        elnas = get_forward_probs(paramDict, eventSeq, seqDict)
        elnbs = get_backward_probs(paramDict, eventSeq, seqDict)
        
        eln_gammas, model_probability = log_gammas_and_model_prob(elnas, elnbs)
        
        for idx, word in enumerate(eventSeq):
                
            elnaArr, elnbArr, seqArr = elnas[idx], elnbs[idx], seqDict[word]
            HMM = paramDict[word]
            A, avg, cov = HMM['A'], HMM['avg'], HMM['cov']
            avgArr = avg.values
            wordArr = words.values
            
            ### calculate emission accumulators
            elngammaArr = eln_gammas[idx][:, 1:-1] ### gammas for this index, for emitting states
            gammaAccumulator[word] += weighted_denominator(elngammaArr)
            meanAccumulator[word] += weighted_emission_mean_numerator(elngammaArr, wordArr)
            covAccumulator[word] += weighted_emission_cov_numerator(elngammaArr, wordArr, avgArr)
            
            ### calculate gamma accumulators for entry states (to normalize accumulator for entry xi)
            elngammaArr_entry = eln_gammas[idx][:, 0]
            gammaAccumulator_entry[word] += weighted_denominator(elngammaArr_entry)
            
            ### calculate xi numerators
            log_xi_entry_num = log_xi_entry_numerator(elnaArr, elnbArr, seqArr, A)
            log_xi_emit_num = log_xi_emit_numerator(elnaArr, elnbArr, seqArr, A)
            log_xi_exit_num = log_xi_exit_numerator(elnaArr, elnbArr, seqArr, A)
            
            ### divide by model probability
            log_xi_entry = elnproduct(log_xi_entry_num, -model_probability)
            log_xi_emit = elnproduct(log_xi_emit_num, -model_probability)
            log_xi_exit = elnproduct(log_xi_exit_num, -model_probability)
            
            ### calculate transition accumulators
            transition_entry_numerator = eexp(log_xi_entry).sum(axis = 0)
            transition_emit_numerator = eexp(log_xi_emit).sum(axis = 0)
            transition_exit_numerator = eexp(log_xi_exit).sum(axis = 0)
            
            xientryAccumulator[word] += transition_entry_numerator
            xiemitAccumulator[word] += transition_emit_numerator 
            xiexitAccumulator[word] += transition_exit_numerator
    
        ### accumulate likelihood 
        lklAccumulator = elnproduct(lklAccumulator, model_probability)
        
    
    print(f'Embedded re-estimation cycle {cycleNum} done! Total Likelihood: {lklAccumulator}')
        
    for word in all_words:
        
        ####### CALCULATE ITERATION RESULTS #####
        upd_avgArr = np.einsum('iv, i -> iv', meanAccumulator[word], 1 / gammaAccumulator[word])
        upd_covArr = np.einsum('ivw, i -> ivw', covAccumulator[word], 1 / gammaAccumulator[word])
        upd_entryArr = np.einsum('i,  -> i', xientryAccumulator[word], 1 / gammaAccumulator_entry[word])
        upd_emitArr = np.einsum('ij, i -> ij', xiemitAccumulator[word], 1 / gammaAccumulator[word])
        upd_exitArr = np.einsum('i, i -> i', xiexitAccumulator[word], 1 / gammaAccumulator[word])
        
        ##### UPDATE THE HMM PARAMETERS #####
        HMM = paramDict[word]
        A, avg, cov = HMM['A'], HMM['avg'], HMM['cov']
        
        ### bring them into shape so we get our target objects ....
        # ... avg
        avg = pd.DataFrame(upd_avgArr, index = avg.index, columns = avg.columns)
        
        # ... cov
        states = cov.index.get_level_values('State').unique()
        cov = pd.concat([pd.DataFrame(upd_covArr[stateIdx], 
                                      index = cov.loc[[stateVal]].index, 
                                      columns = cov.columns) for stateIdx, stateVal in enumerate(states)])
    
        # ... A
        A = join_entry_emit_and_exit_to_A(upd_entryArr, upd_emitArr, upd_exitArr)
    
        paramDict[word] = {'A': A, 'avg': avg, 'cov': cov}
        
        
    

        
        
    
    
    
    
    
    
    
    
    
    