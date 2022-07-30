# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 02:12:26 2022

@author: gianl
"""

import pandas as pd
import numpy as np


def get_word_data(dataFrm, wordStr, extend_word = None):
    
    ### Prep Word Data
    word_instances = dataFrm[dataFrm.Event == wordStr].drop(columns = 'Event')
    word_instances = word_instances.reset_index().set_index(['IndexSF', 'Frame'])
    word_instances.columns.name = 'Var'
    
    return word_instances
         
def get_contiguous_frame_indices(frameList):
    ### for a list of frames, returns a list of equal length containing indices,
    ### where contiguous frames have a common index -> essentially, this demarcates words 
    
    curr_frame, frames = frameList[0], frameList[1:]
    curr_index = 0 ### this is the first index for f
    
    outList = [curr_index] 
    for frame in frames:
        if frame > curr_frame + 1:
            curr_index += 1
        outList.append(curr_index)    
        curr_frame = frame
    
    return outList

def get_word_data(dataFrm, wordStr, extend_word = None):
    
    word_instances = dataFrm[dataFrm.Event == wordStr].drop(columns = 'Event')
    
    wordList = []
    seq = 0
    for seq in word_instances.IndexSF.unique(): ### we do this within sequence, just to be very safe.
        seq_words = word_instances[word_instances.IndexSF == seq]
        seq_words['IndexWord'] = get_contiguous_frame_indices(seq_words.index) 

        ### extend the words by a number of frames, if needed
        ### This part is so ugly because we want to be absolutely sure that
            # 1) extended word frames exist
            # 2) extended words are still contiguous
        if extend_word is not None:
            extend_all_words_in_sequence_list = []
            seq_whole = dataFrm[dataFrm.IndexSF == seq]
            for word_index in seq_words['IndexWord'].unique():
                single_word = seq_words[seq_words.IndexWord == word_index] 
                wordmin, wordmax = single_word.index.min(), single_word.index.max()
                
                extend_one_word_List = []
                ### add the earlier frames
                for extend_counter in range(1, extend_word + 1):
                    try:
                        extendSer = seq_whole.loc[wordmin - extend_counter]
                    except:
                        break ### if the frame before does not exist, we can stop looking for prev frames.
                    else:
                        extend_one_word_List.append(extendSer)
                ### add the later frames
                for extend_counter in range(1, extend_word + 1):
                    try:
                        extendSer = seq_whole.loc[wordmax + extend_counter]
                    except:
                        break ### if the frame before does not exist, we can stop looking for prev frames.
                    else:
                        extend_one_word_List.append(extendSer)
                
                word_extension_frame = pd.concat(extend_one_word_List, axis = 1).T
                word_extension_frame = word_extension_frame.drop(columns = 'Event')
                word_extension_frame['IndexWord'] = word_index
        
                extend_all_words_in_sequence_list.append(word_extension_frame)
            
            ext_words = pd.concat([*extend_all_words_in_sequence_list, seq_words])
        
            wordList.append(ext_words)
            
        else:
            wordList.append(seq_words)
     
    word_instances = pd.concat(wordList)
    word_instances.index.name = 'Frame'
    ### right now, IndexWord identifies words uniquely WITHIN each sequence.
    ### for simplicity: construct a one-dimensional, joint index identifying each word uniquely.
    ### this avoids having to find unique values of two columns, for instance...
    word_instances['IndexWord'] = word_instances['IndexSF'].astype(str) + '_' + word_instances['IndexWord'].astype(str)
    word_instances = word_instances.drop(columns = 'IndexSF')
    word_instances = word_instances.reset_index().set_index(['IndexWord', 'Frame']).sort_index()
    word_instances = word_instances.astype(float)
    word_instances.columns.name = 'Var'
        
    return word_instances
            
            

def calculate_emission_densities(words, avg, cov, density_floor = None):
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
        if any(np.linalg.eigvals(mat) <= 0): ### check for positive semidefiniteness
            print('Covariance matrix singular!')
            raise np.linalg.LinAlgError
        else:
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
    
    if density_floor is not None:
        densityFrm = densityFrm.clip(lower = density_floor)
    
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
    elif num_iterations >= max_iterations:
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
         
def choose_random_sample_words(dataFrm, how_many_words_min = 4, how_many_words_rel = 0.1, extend_word = 1):
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


def split_into_training_and_testing(dataFrm, relative_split = 0.8):
    all_words = dataFrm.Event.unique()
    
    sequenceIdxs = dataFrm.IndexSF.unique()
    num_training = round(relative_split * len(sequenceIdxs))
    
    training_sample_is_complete = False
    while not training_sample_is_complete:
        training_sample = np.random.choice(sequenceIdxs, size = num_training, replace = False)
        trainFrm = dataFrm[dataFrm.IndexSF.isin(training_sample)]
        testFrm = dataFrm[~dataFrm.IndexSF.isin(training_sample)]

        training_sample_is_complete = set(all_words) == set(trainFrm.Event.unique())

    return trainFrm, testFrm


def save_hmm_params(hmmDict, model_name, video_id):
    
    import os
    from ATMANpy.funcs.util_funcs import get_folder_struct
    
    videodataDir = get_folder_struct('ball_event_hmm')[video_id]
    hmmDir = os.path.join(videodataDir, 'saved_models')
    
    modDir = os.path.join(hmmDir, model_name)
    
    try:
        os.mkdir(modDir)
    except:
        pass
    
    for event, params in hmmDict.items():
        
        eventDir = os.path.join(modDir, event)
            
        try:
            os.mkdir(eventDir)
        except:
            pass
            
        A, avg, cov = params['A'], params['avg'], params['cov']
    
        ### Saving Transition Matrix
        APath = os.path.join(eventDir, 'A.csv')
        np.savetxt(APath, A, delimiter = ',')
        
        ### Saving Averages
        avgPath = os.path.join(eventDir, 'avg.csv')
        avg.to_csv(avgPath)
        
        ### Saving Covariance
        covPath = os.path.join(eventDir, 'cov.csv')
        cov.to_csv(covPath)    
    
def read_hmm_params(model_name, video_id):
    import os
    from ATMANpy.funcs.util_funcs import get_folder_struct
    
    videodataDir = get_folder_struct('ball_event_hmm')[video_id]
    hmmDir = os.path.join(videodataDir, 'saved_models')
    
    modelDir = os.path.join(hmmDir, model_name)
    
    outDict = {}
    for param in os.listdir(modelDir):
        if not param.endswith('.csv'):
            paramDir = os.path.join(modelDir, param)
    
            ### Read Transition Matrix
            ADir = os.path.join(paramDir, 'A.csv')
            A = np.loadtxt(ADir, delimiter = ',')
            
            ### Read Averages
            avgPath = os.path.join(paramDir, 'avg.csv')
            avg = pd.read_csv(avgPath, index_col = 0)
            avg.columns.name = 'Var'
            
            ### Read Covariance
            covPath = os.path.join(paramDir, 'cov.csv')
            cov = pd.read_csv(covPath, index_col = [0, 1])
            cov.columns.name = 'Var'
    
            entryDict = {'A': A, 'avg': avg, 'cov': cov}
            outDict[param] = entryDict
        
    return outDict

def save_train_test(train_data, test_data, model_name, video_id):
    import os
    from ATMANpy.funcs.util_funcs import get_folder_struct
    
    videodataDir = get_folder_struct('ball_event_hmm')[video_id]
    hmmDir = os.path.join(videodataDir, 'saved_models')
    modDir = os.path.join(hmmDir, model_name)
    
    trainPath = os.path.join(modDir, 'train_data.csv') 
    train_data.to_csv(trainPath)
    
    testPath = os.path.join(modDir, 'test_data.csv')
    test_data.to_csv(testPath)
    
def read_train_test(model_name, video_id):
    import os
    from ATMANpy.funcs.util_funcs import get_folder_struct
    
    videodataDir = get_folder_struct('ball_event_hmm')[video_id]
    hmmDir = os.path.join(videodataDir, 'saved_models')
    modDir = os.path.join(hmmDir, model_name)
    
    trainPath = os.path.join(modDir, 'train_data.csv') 
    train_data = pd.read_csv(trainPath, index_col = 0)
    
    testPath = os.path.join(modDir, 'test_data.csv')
    test_data = pd.read_csv(testPath, index_col = 0)

    return train_data, test_data


