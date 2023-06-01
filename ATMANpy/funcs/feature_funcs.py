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
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Region

from ATMANpy.types.ShapesTennis3D import Ball, Player

MISSING_SENTINEL_VALUE = 9999 ### if an integer is missing, we add this value. 
                                # can't be np.nan, because np.nan is not an integer.

##### Saving ################

def extract_contour_values(contour, compression = None):
    if compression is None:
        print('Please give an integer value for the degree of compression')
        
    if contour is None:
        featValues = []
    else:
        featValues = np.ravel(contour.compress(compression).points)
        
    number_of_missing_values = compression * 2 - len(featValues)
    if number_of_missing_values > 0:
        featValues = [*featValues, *[MISSING_SENTINEL_VALUE for _ in range(number_of_missing_values)]]
    return featValues

def extract_homography_values(H):
    if H is None:
        featValues = [MISSING_SENTINEL_VALUE] * 8
    else:
        featValues = np.ravel(H)[:8]
    return featValues

def extract_particle_values(particleArr):
    if particleArr is None:
        featValues = None ### figure something out here...
    else:
        featValues = np.ravel(particleArr.astype(int))
    # uniqueParticles = np.unique(particleArr, axis = 0) ### we don't do this, 
                                                         ### because the number of repetitions matters!
    return featValues
                                                         
def extract_region_values(region):
    if region is None:
        featValues = [MISSING_SENTINEL_VALUE] * 8 ### 4 points with 2 coords.
    else:
        coords = [pt.coords for pt in region.to_points()]
        featValues = np.ravel(coords)
    return featValues
    
def save_feature_frame(featFrm, featureName, videoName, storeName):
        
    import os
    featDict = atu.get_folder_struct('features_stored')
    storePath = os.path.join(featDict['main'], videoName, storeName)
    savePath = os.path.join(storePath, featureName)
    
    try:
        os.mkdir(savePath)
        print(f'    Creating new {featureName} data')
    except FileExistsError:
        print(f'    Adding to existing {featureName} data')
        
    import datetime as dt
    timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    randomInt = np.random.randint(999999999)
    
    filePath = os.path.join(savePath, f'{timestamp}_{randomInt}.csv')
    
    featFrm.to_csv(filePath)   
    
def save_contours(stream_features, featureName, videoName, storeName, compression = None):
    myFeature = {frmNum: extract_contour_values(featDict[featureName], compression = compression) for frmNum, featDict in stream_features.items() if featDict[featureName] is not None}
    featFrm = pd.DataFrame(myFeature).T ### in rows: frame number. 
                                        ### in columns, the feature component (x and y values of each point, e.g.)
                                        ### in cells, the values for the feature components
    if not featFrm.empty:
        featLabels = np.ravel([[f'x_{idx}', f'y_{idx}'] for idx in range(compression)])
        featFrm.columns = featLabels
        
        save_feature_frame(featFrm, featureName, videoName, storeName)
    else:
        print(f'   {featureName}: nothing to save!')

def save_homographies(stream_features, videoName, storeName):
    myFeature = {frmNum: extract_homography_values(featDict['H']) for frmNum, featDict in stream_features.items() if featDict['H'] is not None}
    featFrm = pd.DataFrame(myFeature).T ### in rows: frame number. 
                                        ### in columns, the feature component (x and y values of each point, e.g.)
                                        ### in cells, the values for the feature components         
    featLabels = [f'{dim}_{idx}' for dim in ['x', 'y', 'z'] for idx in range(3)][:8]
    featFrm.columns = featLabels
    
    save_feature_frame(featFrm, 'H', videoName, storeName)

def save_particles(stream_features, featureName, videoName, storeName):
    myFeature = {frmNum: extract_particle_values(featDict[featureName]) for frmNum, featDict in stream_features.items() if featDict[featureName] is not None}
    featFrm = pd.DataFrame(myFeature).T ### in rows: frame number. 
                                        ### in columns, the feature component (x and y values of each point, e.g.)
                                        ### in cells, the values for the feature components
    number_of_particles = int(len(featFrm.columns) / 4)
    featLabels = np.ravel([[f'x_{idx}', f'y_{idx}', f'dx_{idx}', f'dy_{idx}'] for idx in range(number_of_particles)])
    featFrm.columns = featLabels
    
    save_feature_frame(featFrm, featureName, videoName, storeName)
    
def save_region(stream_features, featureName, videoName, storeName):
    myFeature = {frmNum: extract_region_values(featDict[featureName]) for frmNum, featDict in stream_features.items() if featDict[featureName] is not None}
    featFrm = pd.DataFrame(myFeature).T
    
    if not featFrm.empty:
        featLabels = np.ravel([[f'x_{idx}', f'y_{idx}'] for idx in range(4)]) ### region has four corners.
        featFrm.columns = featLabels
        
        save_feature_frame(featFrm, featureName, videoName, storeName)
    else:
        print(f'   {featureName}: nothing to save!')
        
def save_events(stream_features, featureName, videoName, storeName):
    myFeature = {frmNum: featDict[featureName] for frmNum, featDict in stream_features.items() if featDict[featureName] is not None}
    featSer = pd.Series(myFeature, name = 'x_0')
    
    if not featSer.empty:
        save_feature_frame(featSer, featureName, videoName, storeName)
    else:
        print(f'   {featureName}: nothing to save!')

###### Loading #############

def read_feature_frame(featureName, videoName, storeName, keep_which = 'last'):
    
    print(f'Reading {featureName}...')
    
    import os
    featDict = atu.get_folder_struct('features_stored')
    savePath = os.path.join(featDict['main'], videoName, storeName, featureName)
    
    
    if os.path.isdir(savePath):
        frmList = []
        for file in os.listdir(savePath):
            filePath = os.path.join(savePath, file)
            myFrm = pd.read_csv(filePath, index_col = 0)
            frmList.append(myFrm)
        
        if frmList == []:
            print(f'   ... No data found!')
            return pd.DataFrame()
        
        featFrm = pd.concat(frmList)
        featFrm = featFrm[~featFrm.reset_index().duplicated().values] ### drop exact duplicates (including the index = frame number)
        
        # try:
        #     featFrm = featFrm.astype(int)
        # except:
        #     print(f'   ... Could not cast to integer!')
    
        if not featFrm.index.duplicated().any():
            print(f'   ... Feature merger for successful.')
        else:
            featFrm = featFrm[~featFrm.index.duplicated(keep = keep_which)]
            print(f'   ... Feature merger for {featureName} unsuccessful due to duplicated frames with ambiguous values. Kept the {keep_which} occurence, arbitrarily.')
            
        return featFrm
    else:
        print(f'   ... No data found!')
        return pd.DataFrame()

def single_2D_feature_to_object(singlefeatSeries):
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
    
    if y_vals == []:
        ### this means we have scalars --> array of values, or single value.
        # valArr = singlefeatSeries.values
        valArr = x_vals
        if len(valArr) == 0:
            return None
        elif len(valArr) == 1:
            return valArr[0]
        else:
            return valArr
    elif z_vals == []:
        ### this means we have a bunch of points, (with x and y values), 
        if dx_vals == [] and dy_vals == []:
            ### we have no speed info, therefore a CONTOUR!
            assert len(x_vals) == len(y_vals)
            num_points = len(x_vals) ### problem with this -> missing values are removed!
            if len(singlefeatDict) / 2 <= 4:
                ### -> have a region!
                myPoints = [Point2D((x, y)) for x, y in zip(x_vals, y_vals)]
                return Region.from_points(*myPoints)
            else:
                ### -> have a contour!
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
    
def single_3D_feature_to_object(singlefeatSeries, featName):
    if 'ball' in featName.lower(): ### is ball
        x, y, z = singlefeatSeries
        return Ball.make((x, y, z))
    elif 'player' in featName.lower():
        x, y, z = singlefeatSeries
        return Player.make((x, y, z)) 
    
       
def load_stored_features(videoName, storeName, featnameList = None, keep_which = 'last'):
    
    ### if no list is given, read all available features
    if featnameList is None: 
        import os
        featDict = atu.get_folder_struct('features_stored')
        featPath = os.path.join(featDict['main'], videoName, storeName)

        featnameList = [f for f in os.listdir(featPath) if not f.endswith('.csv')]
    
    ### collect all feature frames
    frameDict = {}
    indexList = []
    for featName in featnameList:
        featureFrame = read_feature_frame(featName, videoName, storeName, keep_which = keep_which)
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
                if featName.startswith('3D'):
                    featObj = single_3D_feature_to_object(featSeries, featName)
                else:    
                    featObj = single_2D_feature_to_object(featSeries)
                stream_features[idx][featName] = featObj
                
    return stream_features

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

#### PLAY SEQUENCE FUNCS ####

def save_contiguous_play_sequences(split_sfs, video_id, store_name):
    outList = []
    for sf in split_sfs:
        ks = list(sf.keys())
        start_frame, end_frame = min(ks), max(ks)
    
        outList.append((start_frame, end_frame))
        
    #### Write List to text ####
    import pandas as pd
    outFrm = pd.DataFrame(outList, columns = ['StartFrame', 'StopFrame'])
    
    import os
    storeDict = atu.get_folder_struct('features_stored')
    videoPath = os.path.join(storeDict['main'], video_id)
    savePath =  os.path.join(videoPath, store_name)
    
    try:
        print(f'    Creating new video path for {video_id}')
        os.mkdir(videoPath)
    except FileExistsError:
        print(f'    Adding to existing video data for {video_id}')
    
    try:
        print(f'    Creating new store path for {store_name}')
        os.mkdir(savePath)
    except FileExistsError:
        print(f'    Adding to existing data for {store_name}')
    
    import datetime as dt
    timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    filePath = os.path.join(savePath, f'{timestamp}.csv')
    
    outFrm.to_csv(filePath)   
    
    return outFrm

def read_play_sequences(videoName, storeName, keep_which = 'last'):
    import os
    featDict = atu.get_folder_struct('features_stored')
    featPath = os.path.join(featDict['main'], videoName, storeName)

    ### the only relevant items are the play sequences, which are the only thing in pure csv.
    seqList = [el for el in os.listdir(featPath) if el[-4:] == '.csv']
    
    if keep_which == 'last':
        seqName = seqList[-1]
    elif keep_which == 'first':
        seqName = seqList[0]
    elif keep_which.endswith('.csv'):
        seqName = keep_which
    else:
        seqName = keep_which + 'csv'
        
    return pd.read_csv(os.path.join(featPath, seqName), index_col = 0)
    

####### PARALLEL FUNCS

import cv2 as cv
from multiprocessing.pool import Pool
import time
from ATMANpy.types.MultiFrames import Frames
import psutil

def segment_to_stream_func_argument(segment, video_id):
    ### idea: turn a video segment of any kind, e.g.
        # tuple of start and stop frames
        # stream feature
    # into an argument for the stream function that is to be applied in parallel
    
    try:
        start, stop = segment
    except:
        frames = Frames.from_sf(video_id, segment)
        print(f'Starting on segment from {frames.start_frame} to {frames.stop_frame}')
        return (segment, frames)
    else:
        frames = Frames(video_id, start, stop, 'bgr')
        print(f'Starting on segment from {frames.start_frame} to {frames.stop_frame}')
        return (frames, )

def save_features_parallel(video_id, inList, my_stream_func, num_workers = None,
                           verbose = False):
    
    from copy import deepcopy
    video_segments = deepcopy(inList) 
    
    time_start = time.time()
    
    if num_workers is None:
        num_workers = cv.getNumberOfCPUs()

    ### INITIALIZATION ###
    # a) Multiprocessing
    pool = Pool(processes = num_workers)
    pending = []

    # b) initialize the process launching booleans
    jobs_are_open = len(pending) < num_workers
    memory_is_fine = psutil.virtual_memory().percent < 60
    cpu_is_fine =  psutil.cpu_percent() < 70
    
    necessary_to_launch = [jobs_are_open, memory_is_fine, cpu_is_fine]

    # c) message string to print
    messageStr = ''
    previous_messageStr = ''

    while video_segments != []:
        
        if all(necessary_to_launch):

            ### Collect the next segment in the queue
            video_segment, video_segments = video_segments[0], video_segments[1:]
            
            ### get the multiframe, serially! (so videoCapture does not block) ###
            parfunc_arg = segment_to_stream_func_argument(video_segment, video_id)
            
            ### Pass the segment to the worker.
            task = pool.apply_async(my_stream_func, parfunc_arg)
            pending.append(task)
        else:
            
            if verbose:
                ### Message reason for not launching job
                messList = ['no jobs available', 'memory threshold exceeded', 'cpu threshold exceeded']
                to_show = [messStr for messStr, messBool in zip(messList, necessary_to_launch) if not messBool]
                to_show_string = 'and'.join(to_show)
                messageStr = f'Did not start job: {to_show_string}'
                
                if previous_messageStr != messageStr:
                    print(messageStr)
            
            
        ### check if some tasks have finished and can be thrown out of pending!
        pending = [task for task in pending if not task.ready()]
        
        ### adjust the process launch booleans
        jobs_are_open = len(pending) < num_workers
        memory_is_fine = psutil.virtual_memory().percent < 70
        cpu_is_fine =  psutil.cpu_percent() < 95
        
        necessary_to_launch = [jobs_are_open, memory_is_fine, cpu_is_fine]
        previous_messageStr = messageStr
        
    print('Video Segments all launched! Waiting to finish jobs...')
    while pending != []:
        pending = [task for task in pending if not task.ready()]
        print(f'Waiting for {len(pending)} tasks...')
        time.sleep(5)
        
        
    time_elapsed = time.time() - time_start
    minutes_elapsed = round(time_elapsed / 60, 1)
    print(f'Done with {len(video_segments)} segments. Took {minutes_elapsed} minutes')








