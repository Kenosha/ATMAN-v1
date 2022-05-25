# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:45:04 2022

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

# image operation
import cv2 as cv
import numpy as np

import pandas as pd

import os
from random import randrange

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################


########## READ-IN PASSAGES OF PLAY ##########

evtDict = atu.get_folder_struct('ball_event_hmm')

try:
    readPath = evtDict[VIDEO_ID]
except KeyError:
    print('Data not found.')
    
playseqPath = os.path.join(readPath, 'play_sequences')    
start_stop_timestamp = '2022-04-28-19-09-45'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)

# frmList = []
# for file in os.listdir(playseqPath):
#     frm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)
#     frmList.append(frm)
# seqFrm = pd.concat(frmList, ignore_index = True)
    
# homography_timestamp = '2022-05-03-21-53-31'
# file = f'homographies_{homography_timestamp}.csv'

# homFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)


###############################################
        


#%% ############# STREAM FUNCTIONS ########

import ATMANpy.funcs.high_level_funcs as hlf
import ATMANpy.funcs.particle_funcs_vectorized as pf ### in the best case, we ONLY import from hlf

def init_featDict():
    # H = None
    # config = None
    # topplayerRegion, botplayerRegion = None, None
    # particles, idxHistories, closeHistories, modelHistories = None, None, None, None    
    
    featDict = {'H': None, 'config': None,
                'courtLines': None, 'netLines': None, 'trackFlag': None,
                'playerBot': None, 'playerTop': None,
                'ballContours': None, 'ballParticles': None, 'ballIdxHistory': None, 
                'ballCloseHistory': None, 'ballModelHistory': None}
    return featDict

### CACHE ######
FRAME_DIFF_LIST = [4, 6, 8]
CACHE_LENGTH = max(FRAME_DIFF_LIST) * 2 + 1 # we have a frame diff of 8 on both sides, so we need 8 * 2 + 1 = 17
CURR_FRAME_IDX = CACHE_LENGTH // 2 ### the middle frame
FRAME_DIFF_INDICES = [CURR_FRAME_IDX - f for f in reversed(FRAME_DIFF_LIST)] + [CURR_FRAME_IDX + f for f in FRAME_DIFF_LIST]
################

def init_frameDict(frmCache, 
                   curr_frame_idx = CURR_FRAME_IDX, 
                   frame_diff_indices = FRAME_DIFF_INDICES):
    
    cFrm = frmCache[curr_frame_idx]
    gFrm = cFrm.to_grey()
    hFrm = cFrm.to_HSV()
    
    diffFrames = [frmCache[idx].to_grey() for idx in frame_diff_indices]
    gFrm_before = frmCache[curr_frame_idx - 1].to_grey()
    
    frameDict = {'cFrm': cFrm, 'gFrm': gFrm, 'hFrm': hFrm, 
                 'diffFrames': diffFrames, 'gFrm_before': gFrm_before}
    
    return frameDict
    
def get_court_lines(frmDict, featDict, cached_lines = None):
    
    gFrm = frmDict['gFrm']
    H = featDict['H']
    config = featDict['config']
    
    if not cached_lines is None:
        courtLines, H, trackFlag = cached_lines
        from ATMANpy.funcs.court_funcs import find_netlines
        netLines = find_netlines(gFrm, courtLines)     
    else:
        try:
            courtLines, netLines, H, config, trackFlag = hlf.track_courtlines(gFrm, H, config,
                                                                              only_225 = True)
        except Exception as e:
            print('Exception in track_courtlines:')
            print(e)
            courtLines, netLines, H, config = None
            trackFlag = 'Not Found'
    
    featDict['courtLines'] = courtLines
    featDict['netLines'] = netLines
    featDict['H'] = H
    featDict['config'] = config
    featDict['trackFlag'] = trackFlag
    
    return featDict

def get_court_features(frmDict, featDict): ### these two quantities will be used in simulations
    
    ### handle the case if the court is not found ---> simply reset all features!
    if featDict['trackFlag'] == 'Not Found':
        outDict = init_featDict()
        outDict['trackFlag'] = 'Not Found'
        return outDict
        
    gFrm_before = frmDict['gFrm_before']
    cFrm = frmDict['cFrm']
    gFrm = frmDict['gFrm']
    hFrm = frmDict['hFrm']
    diffFrames = frmDict['diffFrames']
    
    H = featDict['H']
    courtLines = featDict['courtLines']
    netLines = featDict['netLines']
    
    particles = featDict['ballParticles']
    playerTop = featDict['playerTop']
    playerBot = featDict['playerBot']
    ballIdxHistory = featDict['ballIdxHistory']
    ballCloseHistory = featDict['ballCloseHistory']
    ballModelHistory = featDict['ballModelHistory']
    
    outDict = featDict.copy() 
    
    ### Player ROIs
    roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
    outDict['roiMask_bot'], outDict['roiMask_top'] = roiMask_bot, roiMask_top

    ### Based on previous frame (and previous players, if they exist...)
    # a) ... if necessary, initialize particles
    if particles is None:
        particles = pf.initialize_particles(gFrm_before, courtLines)

    # b) ... get particle propagation.
    propagation_models, closeBools, modelBools = hlf.decide_particle_propagation(particles, gFrm_before, 
                            topplayer = playerTop, botplayer = playerBot, 
                            closeHistories = ballCloseHistory, 
                            modelHistories = ballModelHistory)
    # c) ... display particle propagation
    # print(modelBools.sum())
    # primPts = pf.particles_to_points(particles[modelBools], particles_to_produce = 100)
    # normPts = pf.particles_to_points(particles[~modelBools], particles_to_produce = 100)

    # cFrm.put(primPts, color = (255, 0, 0)).put(normPts, color = (0, 0, 255)).show()

    ### Players with adaptive ROIs.
    topPlayer, botPlayer = hlf.find_both_players(hFrm, courtLines, netLines, H, remove_shadow = True,
                                                 roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
    outDict['playerTop'], outDict['playerBot'] = topPlayer, botPlayer

    
    ### find ball candidates
    ballContours = hlf.find_ball_contours(gFrm, diffFrames, topPlayer, botPlayer,
                                          cFrm = cFrm, hFrm = hFrm)
    outDict['ballContours'] = ballContours
    
    ### find ball particles
    particles, particleIdxs = hlf.update_particles(particles, propagation_models, ballContours)
    ballIdxHistory, ballCloseHistory, ballModelHistory = pf.update_histories(ballIdxHistory, ballCloseHistory, ballModelHistory, 
                                                                             particleIdxs, closeBools, modelBools)
    outDict['ballParticles'] = particles
    outDict['ballIdxHistory'] = ballIdxHistory
    outDict['ballCloseHistory'] = ballCloseHistory
    outDict['ballModelHistory'] = ballModelHistory
    
    return outDict
  

#%%### DRAW FUNCTIONS #####  
def make_sendframe(frmDict, featDict, frameNum = None):
    
    cFrm = frmDict['cFrm']
    
    sendFrm = cFrm
    
    if frameNum is not None:
        sendFrm = sendFrm.put(str(int(frameNum)), thickness = 2, color = (0, 255, 255), where = (1800, 50))
    
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = sendFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
        particles = featDict['ballParticles']
        playerTop = featDict['playerTop']
        playerBot = featDict['playerBot']
        ballContours = featDict['ballContours']
        
        pPts = pf.particles_to_points(particles, particles_to_produce = 100)
        
        sendFrm = sendFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
        sendFrm = sendFrm.put(pPts, thickness = 1, color = (0, 255, 0))
    
    return sendFrm

def draw_replay(frmDict, featDict):
    
    cFrm = frmDict['cFrm']
    
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
        selectedBall = featDict['ball']
        detectedBall = featDict['ballDetected']
        playerTop = featDict['playerTop']
        playerBot = featDict['playerBot']
        ballContours = featDict['ballContours']
        
        sendFrm = cFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
        
        if selectedBall is not None:
            if detectedBall: ### we have identified the contour as ball!
                sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
            else: ### 
                sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 255))
        
    return sendFrm


def draw_replay_onlyball(frmDict, featDict):
    
    cFrm = frmDict['cFrm']
    
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
        selectedBall = featDict['ball']
        detectedBall = featDict['ballDetected']
        ballContours = featDict['ballContours']
        ballParticles = featDict['ballParticles']
        
        sendFrm = cFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
        if not ballParticles is None:
            pPts = pf.particles_to_points(ballParticles, particles_to_produce = 200)
            sendFrm = sendFrm.put(pPts, thickness = 1, color = (0, 0, 255))
        
        if selectedBall is not None:
            if detectedBall: ### we have identified the contour as ball!
                sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 0))
            else: ### 
                sendFrm = sendFrm.put(selectedBall, thickness = 2, color = (0, 255, 255))
        
    return sendFrm



#%% STREAM AND EXTRACT BALL FEATURES


from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features, replay
from ATMANpy.funcs.high_level_funcs import smoothe_and_choose_ball_contour
from ATMANpy.funcs.high_level_funcs import interpolate_ball_contours, smart_interpolate_ball_contours 

my_stream_func = partial(get_stream_features, 
                        cache_length = CACHE_LENGTH,
                        init_feat_func = init_featDict, 
                        init_frame_func = init_frameDict,
                        court_line_func = get_court_lines, 
                        court_feature_func = get_court_features,
                        draw_func = make_sendframe,
                        verbose = False)



#### Simulation 
video_segments = seqFrm.head(5)
featList = []


for idx, start_stop_frames in video_segments.iterrows():
    start_frame, stop_frame = start_stop_frames
    stream_features_original = my_stream_func(cap, start_frame, stop_frame)
    stream_features_smooth = smoothe_and_choose_ball_contour(stream_features_original)
    featList.append(stream_features_smooth)
    replay(cap, stream_features_smooth, drawFunc = draw_replay,
           ms_between_frames = 30)
    


#%% Feature Compression

myFrame = 10602
featDict = stream_features_smooth[myFrame]
print(featDict.keys())

cFrm = BGRFrame.from_video(cap, myFrame)

#### Features to Compress
# H
# playerBot
# playerTop
# (ballContours) -> do we need these? They could be needed to verify selection by particle filter, 
                    # but we probably only want as few features as possible.
# ball

#### Features to re-build after reading in compressed data
# courtLines
# netLines

### how to compress a contour?

# Ball Contour
myFeature = featDict['ball']

myFeature.points.shape

target_number_of_points = 6
current_number_of_points = myFeature.points.shape[0]

sampleIndices = np.linspace(0, current_number_of_points, num = target_number_of_points, 
                            dtype = int, endpoint = False)
samplePoints = myFeature.points[sampleIndices]

compressedFeature = Contour(samplePoints)

cFrm.put(myFeature, thickness = 2).show()
cFrm.put(compressedFeature, thickness = 2).show()

# Player Contour
myFeature = featDict['playerBot']

myFeature.points.shape

target_number_of_points = 24
current_number_of_points = myFeature.points.shape[0]

sampleIndices = np.linspace(0, current_number_of_points, num = target_number_of_points, 
                            dtype = int, endpoint = False)
samplePoints = myFeature.points[sampleIndices]

compressedFeature = Contour(samplePoints)

cFrm.put(myFeature, thickness = 2).show()
cFrm.put(compressedFeature, thickness = 2).show()

#%% Feature Compression with .compress method

stream_features_smooth = featList[0]

ballPoints = 6
playerPoints = 24

myFrame = 10602
featDict = stream_features_smooth[myFrame]
print(featDict.keys())

cFrm = BGRFrame.from_video(cap, myFrame)

featName = 'ball'
myFeature = featDict[featName]
comprFeature = myFeature.compress(ballPoints)

cFrm.put(myFeature, thickness = 2).show()
cFrm.put(comprFeature, thickness = 2).show()

featName = 'playerBot'
myFeature = featDict[featName]
comprFeature = myFeature.compress(playerPoints)

cFrm.put(myFeature, thickness = 2).show()
cFrm.put(comprFeature, thickness = 2).show()


featName = 'playerTop'
myFeature = featDict[featName]
comprFeature = myFeature.compress(playerPoints)

cFrm.put(myFeature, thickness = 2).show()
cFrm.put(comprFeature, thickness = 2).show()

#%% Saving Features

# def save_feature(stream_features, feature_name, compress = None):
#     myFeatures = {frmNum: featDict[feature_name]}

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

# save_contours(stream_features_smooth, 'ball', VIDEO_ID, compression = 6)
# save_contours(stream_features_smooth, 'playerTop', VIDEO_ID, compression = 24)
# save_contours(stream_features_smooth, 'playerBot', VIDEO_ID, compression = 24)
# save_homographies(stream_features_smooth, VIDEO_ID)

for play_sequence_stream_features in featList:
        
    save_contours(play_sequence_stream_features, 'ball', VIDEO_ID, compression = 6)
    save_contours(play_sequence_stream_features, 'playerTop', VIDEO_ID, compression = 24)
    save_contours(play_sequence_stream_features, 'playerBot', VIDEO_ID, compression = 24)
    save_homographies(play_sequence_stream_features, VIDEO_ID)


#%% Reading these features back in

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


   
featnameList = ['ball', 'playerTop', 'playerBot', 'H']

stream_features_loaded = load_stored_features(VIDEO_ID, featnameList = featnameList)
 
stream_features_loaded = load_stored_features(VIDEO_ID, featnameList = ['H'])
 

    

#%%##### Small Investigation
H = None
config = None

playerTop = None
playerBot = None

for frame in range(10470, 10490):
    hFrm = HSVFrame.from_video(cap, frame)
    cFrm = hFrm.to_BGR()
    gFrm = hFrm.to_grey() 
    
    courtLines, netLines, H, config, trackFlag = hlf.track_courtlines(gFrm, H, config,
                                                                      only_225 = False)
    
    if not H is None:
        roiMask_top, roiMask_bot = hlf.get_adaptive_ROIs(playerTop, playerBot, H, hFrm)
        
        
        ### Players with adaptive ROIs.
        playerTop, playerBot = hlf.find_both_players(hFrm, courtLines, netLines, H, remove_shadow = True,
                                                     roiMask_top = roiMask_top, roiMask_bot = roiMask_bot)
        
        cFrm.mask(roiMask_bot).show()
        cFrm.put(playerTop, color = (0, 0, 255), thickness = 2).put(playerBot, color = (0, 0, 255), thickness = 2).show()












