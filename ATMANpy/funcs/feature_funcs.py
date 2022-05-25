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
    featLabels = np.ravel([[f'x_{idx}', f'y_{idx}'] for idx in range(compression)])
    featFrm.columns = featLabels
    
    save_feature_frame(featFrm, featureName, videoName)


def save_homographies(stream_features, videoName):
    myFeature = {frmNum: extract_homography_values(featDict['H']) for frmNum, featDict in stream_features.items() if featDict['H'] is not None}
    featFrm = pd.DataFrame(myFeature).T ### in rows: frame number. 
                                        ### in columns, the feature component (x and y values of each point, e.g.)
                                        ### in cells, the values for the feature components
    featLabels = np.ravel([[f'x_{idx}', f'y_{idx}', f'z_{idx}'] for idx in range(3)])[:8]
    featFrm.columns = featLabels
    
    save_feature_frame(featFrm, 'H', videoName)

#%% Loading 


def read_feature_frame(featureName, videoName):
       
    import os
    featDict = atu.get_folder_struct('features_stored')
    savePath = os.path.join(featDict['main'], videoName, featureName)
    
    frmList = []
    for file in os.listdir(savePath):
        filePath = os.path.join(savePath, file)
        myFrm = pd.read_csv(filePath, index_col = 0)
        frmList.append(myFrm)
        
    featFrm = pd.concat(frmList)
    featFrm = featFrm.drop_duplicates()

    if not featFrm.index.duplicated().any():
        print(f'Feature merger for {featureName} successful.')
    else:
        featFrm = featFrm[~featFrm.index.duplicated(keep = 'last')]
        print(f'Feature merger for {featureName} unsuccessful due to duplicated frames with ambiguous values. Kept the last occurence, arbitrarily.')
        
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
    
    if z_vals is not []:
        ### this means we have a bunch of points, (with x and y values), 
        ### and therefore a CONTOUR!
        assert len(x_vals) == len(y_vals)
        num_points = len(x_vals)
        myPoints = np.dstack([x_vals, y_vals]).reshape((num_points, 1, 2)) ### bring them into shape
        return Contour(myPoints)
    else:
        ### this means we have x, y and z -> a HOMOGRAPHY!
        assert len(x_vals) == len(y_vals) == 3
        assert len(z_vals) == 2
        
        z_vals = [*z_vals, 1]
        myH = np.dstack([x_vals, y_vals, z_vals])
        
        return myH
       
def load_stored_features(videoName, featnameList = None):
    
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
        featureFrame = read_feature_frame(featName, videoName)
        frameDict[featName] = featureFrame
        indexList.extend(featureFrame.index.to_list())  ### construct the set of all frames (indices)
                                                        ### for which we have at least one feature.
        
    ### keep unique frames. These will be our dictionary keys!
    indexList = list(set(indexList))
    
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

