# -*- coding: utf-8 -*-
"""
Created on Wed May 25 01:01:01 2022

Functions to compress, save and retrieve features to/from disc.

@author: gianl
"""

import numpy as np
import pandas as pd

import ATMANpy.funcs.util_funcs as atu 
from ATMANpy.types.Contours import Contour

##### Saving ################

def extract_contour_values(contour, compression = None):
    MISSING_SENTINEL_VALUE = 9999 ### if an integer is missing, we add this value. 
                                    # can't be np.nan, because np.nan is not an integer.
    if compression is None:
        print('Please give an integer value for the degree of compression')
    else:
        featValues = np.ravel(contour.compress(compression).points)
        number_of_missing_values = compression * 2 - len(featValues)
        if number_of_missing_values > 0:
            featValues = [*featValues, *[MISSING_SENTINEL_VALUE for _ in range(number_of_missing_values)]]
        return featValues

def extract_homography_values(homography):
    return np.ravel(homography)[:8]

def extract_particle_values(particleArr):
    # uniqueParticles = np.unique(particleArr, axis = 0) ### we don't do this, 
                                                         ### because the number of repetitions matters!
    return np.ravel(particleArr.astype(int))
                                                         

def save_feature_frame(featFrm, featureName, videoName):
        
    import os
    featDict = atu.get_folder_struct('features_stored')
    savePath = os.path.join(featDict['main'], videoName, featureName)
    
    try:
        os.mkdir(savePath)
    except FileExistsError:
        print(f'   Adding to existing {featureName} data')
        
    import datetime as dt
    timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    randomInt = np.random.randint(999999999)
    
    filePath = os.path.join(savePath, f'{timestamp}_{randomInt}.csv')
    
    featFrm.to_csv(filePath)   
    

def save_contours(stream_features, featureName, videoName, compression = None):
    myFeature = {frmNum: extract_contour_values(featDict[featureName], compression = compression) for frmNum, featDict in stream_features.items() if featDict[featureName] is not None}
    featFrm = pd.DataFrame(myFeature).T ### in rows: frame number. 
                                        ### in columns, the feature component (x and y values of each point, e.g.)
                                        ### in cells, the values for the feature components
    if not featFrm.empty:
        featLabels = np.ravel([[f'x_{idx}', f'y_{idx}'] for idx in range(compression)])
        featFrm.columns = featLabels
        
        save_feature_frame(featFrm, featureName, videoName)
    else:
        print(f'   {featureName}: nothing to save!')


def save_homographies(stream_features, videoName):
    myFeature = {frmNum: extract_homography_values(featDict['H']) for frmNum, featDict in stream_features.items() if featDict['H'] is not None}
    featFrm = pd.DataFrame(myFeature).T ### in rows: frame number. 
                                        ### in columns, the feature component (x and y values of each point, e.g.)
                                        ### in cells, the values for the feature components         
    featLabels = [f'{dim}_{idx}' for dim in ['x', 'y', 'z'] for idx in range(3)][:8]
    featFrm.columns = featLabels
    
    save_feature_frame(featFrm, 'H', videoName)

def save_particles(stream_features, featureName, videoName):
    myFeature = {frmNum: extract_particle_values(featDict[featureName]) for frmNum, featDict in stream_features.items() if featDict[featureName] is not None}
    featFrm = pd.DataFrame(myFeature).T ### in rows: frame number. 
                                        ### in columns, the feature component (x and y values of each point, e.g.)
                                        ### in cells, the values for the feature components
    number_of_particles = int(len(featFrm.columns) / 4)
    featLabels = np.ravel([[f'x_{idx}', f'y_{idx}', f'dx_{idx}', f'dy_{idx}'] for idx in range(number_of_particles)])
    featFrm.columns = featLabels
    
    save_feature_frame(featFrm, featureName, videoName)
    
    

###### Loading #############

def read_feature_frame(featureName, videoName, keep_which = 'last'):
       
    import os
    featDict = atu.get_folder_struct('features_stored')
    savePath = os.path.join(featDict['main'], videoName, featureName)
    
    frmList = []
    for file in os.listdir(savePath):
        filePath = os.path.join(savePath, file)
        myFrm = pd.read_csv(filePath, index_col = 0)
        frmList.append(myFrm)
        
    featFrm = pd.concat(frmList)
    featFrm = featFrm[~featFrm.reset_index().duplicated().values] ### drop exact duplicates (including the index = frame number)

    if not featFrm.index.duplicated().any():
        print(f'Feature merger for {featureName} successful.')
    else:
        featFrm = featFrm[~featFrm.index.duplicated(keep = keep_which)]
        print(f'Feature merger for {featureName} unsuccessful due to duplicated frames with ambiguous values. Kept the {keep_which} occurence, arbitrarily.')
        
    return featFrm

def single_feature_to_object(singlefeatSeries):
    ### Turns a dictionary corresponding to a row of a feature frame (a single picture frame)
    ### into the object that we can work with.
    
    singlefeatDict = singlefeatSeries.to_dict()
    
    MISSING_SENTINEL_VALUE = 9999 ### if an integer is missing, we add this value. 
                                # can't be np.nan, because np.nan is not an integer.
                                
    x_vals = [v for k, v in singlefeatDict.items() if k.startswith('x') if not v == MISSING_SENTINEL_VALUE]
    y_vals = [v for k, v in singlefeatDict.items() if k.startswith('y') if not v == MISSING_SENTINEL_VALUE]
    z_vals = [v for k, v in singlefeatDict.items() if k.startswith('z') if not v == MISSING_SENTINEL_VALUE]
    dx_vals = [v for k, v in singlefeatDict.items() if k.startswith('dx') if not v == MISSING_SENTINEL_VALUE]
    dy_vals = [v for k, v in singlefeatDict.items() if k.startswith('dy') if not v == MISSING_SENTINEL_VALUE]
    
    if z_vals == []:
        ### this means we have a bunch of points, (with x and y values), 
        if dx_vals == [] and dy_vals == []:
            ### we have no speed info, therefore a CONTOUR!
            assert len(x_vals) == len(y_vals)
            num_points = len(x_vals)
            myPoints = np.dstack([x_vals, y_vals]).reshape((num_points, 1, 2)) ### bring them into shape
            return Contour(myPoints)
        else:
            assert len(x_vals) == len(y_vals) == len(dx_vals) == len(dy_vals)
            return np.array([x_vals, y_vals, dx_vals, dy_vals])
    else:
        ### this means we have x, y and z -> a HOMOGRAPHY!
        assert len(x_vals) == len(y_vals) == 3
        assert len(z_vals) == 2
        
        z_vals = [*z_vals, 1]
        myH = np.array([x_vals, y_vals, z_vals])
        
        return myH
       
def load_stored_features(videoName, featnameList = None, keep_which = 'last'):
    
    ### if no list is given, read all available features
    if featnameList is None: 
        import os
        featDict = atu.get_folder_struct('features_stored')
        featPath = os.path.join(featDict['main'], videoName)

        featnameList = os.listdir(featPath)
    
    ### collect all feature frames
    frameDict = {}
    indexList = []
    for featName in featnameList:
        featureFrame = read_feature_frame(featName, videoName, keep_which = keep_which)
        frameDict[featName] = featureFrame
        indexList.extend(featureFrame.index.to_list())  ### construct the set of all frames (indices)
                                                        ### for which we have at least one feature.
        
    ### keep unique frames. These will be our dictionary keys!
    indexList = sorted(list(set(indexList)))
    
    stream_features = {}
    for idx in indexList:
        stream_features[idx] = {}
        for featName in featnameList:
            try:
                featSeries = frameDict[featName].loc[idx]
            except KeyError:
                stream_features[idx][featName] = None
            else:
                featObj = single_feature_to_object(featSeries)
                stream_features[idx][featName] = featObj
                
    return stream_features

# def split_joint_features(stream_features, min_feature_length = 0):
#     featList = []
    
#     currFeat = {}
    
#     lastFrame = None
#     for k in stream_features:
        
#         if lastFrame is None:
#             currFeat[k] = stream_features[k]
#         else:
#             if k == lastFrame + 1:
#                 currFeat[k] = stream_features[k]
#             else:
#                 if len(currFeat) > min_feature_length:
#                     featList.append(currFeat)
#                 currFeat = {}
#         lastFrame = k

#     return featList


def split_joint_features(stream_features, min_feature_length = 0):
    featList = []
    
    ks = sorted(list(stream_features.keys()))
    prevFrame, frames = ks[0], ks[1:]
    
    currFeat = {prevFrame: stream_features[prevFrame]} ### initialize with the first element.

    for k in frames:
        
        if k > prevFrame + 1: ### if there is a gap between last and current frame
            if len(currFeat) > min_feature_length: ### if the current accumulated sf is large enough
                featList.append(currFeat)    
            currFeat = {}
        
        currFeat[k] = stream_features[k]
        prevFrame = k
    
    ### finally, add in the remaining features (if long enough)
    if len(currFeat) > min_feature_length: 
        featList.append(currFeat)    
    
    return featList
        

def join_split_features(splitList):
    return {k: v for sf in splitList for k, v in sf.items()}
    

def filter_features_by_play_sequence_frame(stream_features, seqFrame):
    ### seqFrame -> two columns: start_frame and stop_frame    
    stream_features_filtered = {}
    for idx, start, stop in seqFrame.to_records():
        for frame in range(start, stop + 1):
            try:
                stream_features_filtered[frame] = stream_features[frame]
            except KeyError:
                pass
    return stream_features_filtered

def save_contiguous_play_sequences(split_sfs, VIDEO_ID):
    outList = []
    for sf in split_sfs:
        ks = list(sf.keys())
        start_frame, end_frame = min(ks), max(ks)
    
        outList.append((start_frame, end_frame))
        
    #### Write List to text ####
    import pandas as pd
    outFrm = pd.DataFrame(outList, columns = ['StartFrame', 'StopFrame'])
    
    import os
    ballDict = atu.get_folder_struct('ball_event_hmm')
    savePath = os.path.join(ballDict['main'], VIDEO_ID, 'play_sequences')
    
    try:
        os.mkdir(savePath)
    except FileExistsError:
        print('   Adding to existing play_sequence data')
    
    import datetime as dt
    timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    filePath = os.path.join(savePath, f'play_start_stop_{timestamp}.csv')
    
    outFrm.to_csv(filePath)   
    
    return outFrm
