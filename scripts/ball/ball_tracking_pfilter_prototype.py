# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:27:15 2022

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
from ATMANpy.Points import Point2D, SafePoint2D
from ATMANpy.Lines import Line, Lines, CourtLines, Region
from ATMANpy.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.Contours import Contour

import cv2 as cv
import pandas as pd
import numpy as np

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

#### READ IN THE BALL TRACKING SVM ######
import os
ballDict = atu.get_folder_struct('ball_svm')
fitPath = os.path.join(ballDict['main'], VIDEO_ID, 'fitted_svms')
svm_name = 'svm_dict_2022-04-13'

svmPath = os.path.join(fitPath, svm_name + '.pickle')

import pickle

with open(svmPath, 'rb') as f:
    svmDict = pickle.load(f)
    
    
# clf.predict([[1] * 8]) ## test if it works

######## GET SCHEMA ########
from ATMANpy.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)


###### DEFINE TRACK PROCEDURE ########
from ATMANpy.high_level_funcs import find_homography_225_only, update_homography
from ATMANpy.high_level_funcs import find_courtlines, find_netlines

def track_courtlines(gFrm, prev_H, prev_config, schemaLines = schemaLines):
    if prev_H is None:
        res = find_homography_225_only(gFrm)
        outFlag = 'Found'
    else:
        # print('Courtlines found, updating homography...')
        res = update_homography(gFrm, prev_H, prev_config)
        outFlag = 'Tracked'
        
    if not res is None:
        H, config = res
        courtLines = find_courtlines(H, schemaLines)
        netLines = find_netlines(gFrm, courtLines)       
    
        # outFrm = gFrm.to_BGR().put(courtLines, thickness = 3)
        # outFrm = outFrm.put(netLines, color = (255, 0, 0), thickness = 3)
    else:
        outFlag = 'Not Found'
        courtLines, netLines = None, None
        H, config = None, None
        
    return courtLines, netLines, H, config, outFlag

######## READ IN AND PREPARE BURN-IN ######
burnDict = atu.get_folder_struct('burn_in')

try:
    readPath = burnDict[VIDEO_ID]
except KeyError:
    print('Burn-In Data not found.')
    
medFrm = HSVFrame.from_csv(readPath, name = 'median')
stdFrm = HSVFrame.from_csv(readPath, name = 'std')

##### GET THE BURNED IN MASK ########
NORMALIZED_DEVIATION_THRESHOLD = 4
BURN_IN_HSV_CHANNEL_WEIGHTS = (1, 50, 1) #interested only in hue and value

from ATMANpy.player_funcs import get_burn_in_mask
from functools import partial
match_specific_burn_in_func = partial(get_burn_in_mask, medFrm, stdFrm,
                                      normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD,
                                      channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS) ### these are always the same!
#######################

########## GET REGIONS OF INTEREST ON SCHEMA ##########

WIDTH_FACTOR = 2 ### how much of the doubles alley width does a player maximally next to court
DEPTH_FACTOR = 0.2 ### how much of side length does a player maximally stand behind baseline
ADD_DEPTH_FACTOR = 1.25 ### top side depth adjustment - player figure is tall, so looks to be outside of ROI.
# we could adjust this with the camera matrix, but let's do something eyeballed here, for the moment.

from ATMANpy.player_funcs import get_initial_ROIs
ROI_lower, ROI_upper = get_initial_ROIs(schemaLines,
                                         width_factor = WIDTH_FACTOR, 
                                         depth_factor = DEPTH_FACTOR, 
                                         add_depth_factor_top = ADD_DEPTH_FACTOR)

def get_projected_ROIs(ROI_lower, ROI_upper, H):
    projROI_lower = ROI_lower.transform('apply_homography', H).transform('round_points')
    projROI_upper = ROI_upper.transform('apply_homography', H).transform('round_points')
    # cFrm.put(projROI_upper, color = (0, 0, 255), thickness = 2).put(projROI_lower, color = (0, 255, 0), thickness = 2).show()
    return projROI_lower, projROI_upper

schema_specific_roi_projection_func = partial(get_projected_ROIs, ROI_lower, ROI_upper)

# schemaFrm.to_BGR().put(ROI_upper, color = (0, 0, 255), thickness = 2).put(ROI_lower, color = (0, 255, 0), thickness = 2).show()
##############

#### now, we get a frame and an h
verbose = False
COURT_DILATION = 1.5
NET_DILATION = 2.5
ROI_DILATION = 1

from ATMANpy.high_level_funcs import COURT_LINE_WIDTH, WHITE_PIXEL_THRESHOLD
from ATMANpy.player_funcs import find_player_contour


def find_player(frame, courtLines, netLines, H, roiMask = None, verbose = False):
    
    ### get the burned in mask for this specific homography
    burn_foreMask = match_specific_burn_in_func(frame, H,
                                    normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD, 
                                    channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS, 
                                    verbose = verbose)
    
    playerRegion = find_player_contour(frame, burn_foreMask, roiMask)
    
    return playerRegion


################ ################

######### FIND BALL CANDIDATES ######

MAX_NUMBER_OF_CONTOURS = 40 # if there are too many contours, likely the camera was zooming or panning.
                            # this causes many call candidates (due to frame differencing).
                            # these spurious candidates we don't want in our training data
                            # also, these frames take ages to load due to labelling.

from ATMANpy.ball_funcs import get_motion_mask_from_frames, find_ball_candidate_mask, make_feature_frame

def find_ball_contours(gFrm, diffFrames, svmDict, topplayerRegion, botplayerRegion,
                       cFrm = None, hFrm = None):
    
    clf, features, mean, std = svmDict['svm'], svmDict['features'], svmDict['mean'], svmDict['std']
    
    motMask = get_motion_mask_from_frames(gFrm, diffFrames)
    intMask = find_ball_candidate_mask(cFrm, motMask, topplayerRegion, botplayerRegion)
    
    cnts, _ = intMask.find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = [Contour(c) for c in cnts if len(c) > 2] ### need at least three points for a contour
    
    if not 0 < len(cnts) < MAX_NUMBER_OF_CONTOURS:
        return None
        
    cntFrm = make_feature_frame(cnts, hFrm, gFrm, intMask).dropna()
    cntFrm = (cntFrm - mean) / std

    cntFrm['isBall'] = clf.predict(cntFrm[features].values)
    
    ballIdx = list(cntFrm[cntFrm.isBall == 1].index)
    
    return [c for i, c in enumerate(cnts) if i in ballIdx]

#############################
START_FRAME = 11000
STOP_FRAME = 11020


H = None
config = None
    
#### init cache ###
FRAME_DIFF_LIST = [4, 6, 8]
CACHE_LENGTH = max(FRAME_DIFF_LIST) * 2 + 1 # we have a frame diff of 8 on both sides, so we need 8 * 2 + 1 = 17
CURR_FRAME_IDX = CACHE_LENGTH // 2 ### the middle frame
FRAME_DIFF_INDICES = [CURR_FRAME_IDX - f for f in reversed(FRAME_DIFF_LIST)] + [CURR_FRAME_IDX + f for f in FRAME_DIFF_LIST]
frmCache = []

### init video stream ###
from ATMANpy.cv_funcs import video_stream
gen = video_stream(cap, start_frame = START_FRAME - CACHE_LENGTH, stop_frame = STOP_FRAME)
frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
######



while True:
    latestFrame = BGRFrame(rawArr)    
    ### cache frames, so we can frame difference 
    ### without having to re-generate the frames from the video capture
    frmCache.append(latestFrame)
    if len(frmCache) < 17:
        continue
    if len(frmCache) > 17:
        print(f'Cache FUBAR, {len(frmCache)} items') ### this happens if the cache drop at the end fails
        break
    
    cFrm = frmCache[CURR_FRAME_IDX]
    gFrm = cFrm.to_grey()
    hFrm = cFrm.to_HSV()
    
    diffFrames = [frmCache[idx].to_grey() for idx in FRAME_DIFF_INDICES]
    
    try:
        courtLines, netLines, H, config, trackFlag = track_courtlines(gFrm, H, config)
    except Exception as e:
        print('Exception in track_courtlines:')
        print(e)
        courtLines = None
    
    if trackFlag == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
        

    elif trackFlag == 'Found':
        
        ### Project Regions of Interest
        projROI_lower, projROI_upper = schema_specific_roi_projection_func(H)
        roiMask_lower, roiMask_upper = hFrm.within_region(projROI_lower), hFrm.within_region(projROI_upper)

        ### Top Player, with updated averages
        topplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_upper)
        
        ### Bottom Player, with updated averages
        botplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_lower)
        
        ### find ball
        ballContours = find_ball_contours(gFrm, diffFrames, svmDict, topplayerRegion, botplayerRegion,
                                          cFrm = cFrm, hFrm = hFrm)
        
        sendFrm = cFrm.put(topplayerRegion, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(botplayerRegion, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
        
    elif trackFlag == 'Tracked':
        ### If the court was tracked, we don't change the projected ROI, 
        #   and we don't change the color averages and standard deviations.
        
        topplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_upper)
        botplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_lower)
        ballContours = find_ball_contours(gFrm, diffFrames, svmDict, topplayerRegion, botplayerRegion,
                                          cFrm = cFrm, hFrm = hFrm)
        
        sendFrm = cFrm.put(topplayerRegion, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(botplayerRegion, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(ballContours, thickness = 2, color = (255, 0, 0))
    
    sendArr = sendFrm.array  
    try:
        frmNum, rawArr = gen.send(sendArr)
    except:
        cv.destroyAllWindows()
        break 
    
    ### cache drop  
    frmCache = frmCache[1:] # drop oldest item
    
else:
    cv.destroyAllWindows()



######### Particle Filter

sendFrm.show()

### Parameters ###

N = 1000

# the normal movement model
A = np.matrix([[1, 0, 1, 0], 
               [0, 1, 0, 1],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

# the close-to-player movement model
A_prime = np.matrix([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]) 

# the matrix that turns true state x into obversation z (which is only a location!)
H = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0]])

PROCESS_VARIANCE = 100
OBSERVATION_VARIANCE = 1000

import numpy.matlib
Q = np.matlib.identity(4) * PROCESS_VARIANCE
R = np.matlib.identity(2) * OBSERVATION_VARIANCE

LAMBDA = 0.00001 ### spacial poisson density of clutter... whatever that is. See isard & blake (1998)
P_NO_BALL = 0.1 ### probability that ball is not among tracked objects
beta = LAMBDA * P_NO_BALL

P_PRIME = 0.5

#### auxiliaries ###

weight_cov = R + H * Q * H.T
weight_cov_inv = weight_cov ** (-1)

denom = 2 * np.pi * np.linalg.det(weight_cov) ** 0.5
# HA = H * A

K = Q * H.T * weight_cov_inv
C = (np.matlib.identity(4) - K * H) * Q

########

### initialize prior particles.

def initialize_particles(gFrm, courtLines, N = 1000):
    # particles are four dimensional vectors, representing location and speed in x and y coords resp.
    MIN_LOC_X, MIN_LOC_Y = 0, 0
    MAX_LOC_X, MAX_LOC_Y = gFrm.cols, gFrm.rows
    MIN_SPEED_X, MIN_SPEED_Y = 0, 0
    
    conv_X, conv_Y = courtLines.pixel_in_meter(horiLine = 'Top Baseline', vertLine = 'Top Center Line')  
    meter_per_frame = 70 / cap.get(cv.CAP_PROP_FPS) ### 260 km/h -> 70 m/s
    MAX_SPEED_X, MAX_SPEED_Y = conv_X * meter_per_frame, conv_Y * meter_per_frame
    
    min_vec = [MIN_LOC_X, MIN_LOC_Y, MIN_SPEED_X, MIN_SPEED_Y]
    max_vec = [MAX_LOC_X, MAX_LOC_Y, MAX_SPEED_X, MAX_SPEED_Y]
    
    initial_particles = np.random.uniform(min_vec, max_vec, (N, 4))
    return initial_particles

# chamfer distance
def chamfer_distance(gFrm, topplayer, botplayer, threshold = 100):
    pMask = gFrm.within_contour(topplayerRegion) | gFrm.within_contour(botplayerRegion)    
    dist = cv.distanceTransform((~pMask).array, cv.DIST_L2, 3,  dstType=cv.CV_32F)
    dMask = (Frame(dist) < threshold)
    if verbose: dMask.show()
    return dMask


##### define markov process and observation rules #####
def get_motion_model(S, chamferMask):
    ### Find Chamfer-Appropriate Model
    As = []
    for s in S:
        ### determine model to use
        x, y = s[:2].squeeze()
        sPt = Point2D((round(x), round(y)))
        if not chamferMask.eval_at(sPt): ### not close to player
            A_star = A
        else:
            ### this is actually more involved, but for the moment...
            if np.random.rand() < P_PRIME:
                A_star = A
            else:
                A_star = A_prime
                
        As.append(A_star)
    
    return As

def get_means_and_weights(S, ballContours):
    ### Baseline Clutter Weights
    
    wDict = {}
    mDict = {}
    
    wDict[0] = [beta for i in range(N)]
    mDict[0] = [myA * s for myA, s in zip(As, S)] ### start here! this is ok, but very slow!
    
    ### Measurement
    
    
    for idx_z, z in enumerate(ballContours):
        z_coords = z.center().coords
        
        # wList = np.zeros(N)
        # mList = np.zeros(N)
        wList = []
        mList = []
        for s, A_star in zip(S, As):
            ### weight
            updParts = H * A_star * s
            dist = z_coords - updParts.T
            chk = -0.5 * dist * weight_cov_inv * dist.T
            w = np.exp(chk.item()) / denom
            wList.append(w.squeeze())
            
            ### m
            m = A_star * s + K * dist.T
            mList.append(m)
            
        wDict[idx_z + 1] = wList
        mDict[idx_z + 1] = mList    
        
    return mDict, wDict
        
def normalize_weights(wDict):
    wArr = np.concatenate([l for l in wDict.values()])
    return wArr / wArr.sum()


def resample(mDict, wDict):
    wArr = normalize_weights(wDict)
    
    ### Resample
    contourList = []
    motherList = []
    sampleList = []
    for k in range(N):
        sampleIdx = np.random.choice(N * 2, p = wArr)
        i, j = sampleIdx % N, sampleIdx // N
    
        sample_mean = mDict[j][i]
        sample_cov = Q if j == 0 else C
        
        sample = np.random.multivariate_normal(np.asarray(sample_mean).squeeze(), sample_cov)
        sampleList.append(sample)
        motherList.append(i)
        contourList.append(j)
    
    sampleList = np.expand_dims(np.array(sampleList), axis = 2)
    motherList = np.array(motherList)    
    contourList = np.array(contourList)
    
    return sampleList, motherList, contourList


N = 10000
S_in = initialize_particles(gFrm, courtLines, N = N)
S_in = np.expand_dims(S_in, axis = 2)



chamferMask = chamfer_distance(gFrm, topplayerRegion, botplayerRegion)
As = get_motion_model(S_in, chamferMask)        
mDict, wDict = get_means_and_weights(S_in, ballContours)
S_out, I_out, J_out = resample(mDict, wDict)

print(f'Evidence Based Filtering Rate: {round(J_out.sum() / len(J_out), 3)}')


def particles_to_points(S, particles_to_produce = None):
    def round_tup(tup):
        return round(tup[0]), round(tup[1])
    
    if not particles_to_produce is None:
        to_draw = S[:particles_to_produce]
    else:
        to_draw = S.copy()
    coords = to_draw[:, :2, :]
    points = [Point2D(round_tup(c.squeeze())) for c in coords]
    return points

def average_particles(S, particles_to_produce = 100):
    bucket_length = len(S) // particles_to_produce
    bucketLists = [S[b:b + bucket_length] for b in range(particles_to_produce)]
    avgLists = np.array([np.mean(bucket, axis = 0) for bucket in bucketLists])
    return avgLists

def plot_particles(S, particles_to_produce = 500):
    return particles_to_points(average_particles(S, particles_to_produce = particles_to_produce))


inPt = particles_to_points(S_in, particles_to_produce = 100)
outPt = particles_to_points(S_out, particles_to_produce = 100)

gFrm.to_BGR().put(inPt).show()
gFrm.to_BGR().put(outPt).show()






# updParts = H * A * S.T
# updParts.shape

# dist = z1 - updParts.T
# dist = np.expand_dims(dist, axis = 1)
# dist_T = np.expand_dims(dist, axis = 2).T

# np.exp(-0.5 * dist * weight_cov_inv * dist_T)







