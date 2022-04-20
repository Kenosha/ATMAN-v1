# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:55:01 2022

Faster, vectorized versions of the particle filter in ball_tracking_pfilter_prototype

@author: gianl
"""

import numpy as np
import numpy.matlib

import cv2 as cv

from ATMANpy.types.Frames import Frame
from ATMANpy.types.Points import Point


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

# the close-to-player movement model with reversed thrust
A_primeprime = np.matrix([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]]) 

# the matrix that turns true state x into obversation z (which is only a location!)
H = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0]])

PROCESS_VARIANCE = 100
OBSERVATION_VARIANCE = 100

Q = np.matlib.identity(4) * PROCESS_VARIANCE
R = np.matlib.identity(2) * OBSERVATION_VARIANCE

LAMBDA = 0.0000001 ### spacial poisson density of clutter... whatever that is. See isard & blake (1998)
P_NO_BALL = 0.25 ### probability that ball is not among tracked objects
beta = LAMBDA * P_NO_BALL

P_PRIME = 0.3

#### auxiliaries ###

weight_cov = R + H * Q * H.T
weight_cov_inv = weight_cov ** (-1)

denom = 2 * np.pi * np.linalg.det(weight_cov) ** 0.5
# HA = H * A

K = Q * H.T * weight_cov_inv
C = (np.matlib.identity(4) - K * H) * Q


########## FUNCTIONS ###########


def initialize_particles(gFrm, courtLines, N = 1000, fps = 25):
    # particles are four dimensional vectors, representing location and speed in x and y coords resp.
    MIN_LOC_X, MIN_LOC_Y = 0, 0
    MAX_LOC_X, MAX_LOC_Y = gFrm.cols, gFrm.rows
    MIN_SPEED_X, MIN_SPEED_Y = 0, 0
    
    conv_X, conv_Y = courtLines.pixel_in_meter(horiLine = 'Top Baseline', vertLine = 'Top Center Line')  
    meter_per_frame = 70 / fps ### 260 km/h -> 70 m/s
    MAX_SPEED_X, MAX_SPEED_Y = conv_X * meter_per_frame, conv_Y * meter_per_frame
    
    min_vec = [MIN_LOC_X, MIN_LOC_Y, MIN_SPEED_X, MIN_SPEED_Y]
    max_vec = [MAX_LOC_X, MAX_LOC_Y, MAX_SPEED_X, MAX_SPEED_Y]
    
    initial_particles = np.random.uniform(min_vec, max_vec, (N, 4))
    
    initial_particles = np.expand_dims(initial_particles, axis = 2)
    
    return initial_particles

def compare_histories(closeArr, modelArr):
    """
    If the particle was always close, 
        if it has never been moved with A'
            -> move it with A' with probability p'                  (CASE 1)
        if it has been moved with A'
            -> move it with A                                       (CASE 2)
    If the particle was not always close,
        if it has never been moved with A'
            -> move it with A' with probability p'                  (CASE 3)
        it it has been moved with A'
            -> as in Yan Christmas Kittler: t1 versus t             (CASE 4)
            (only this is not an edge case. everything else is an edge case.
             
    Returns a boolean vector of length N, 
        with True denoting possible model switch (A or A'), 
        and False denoting to use model A always
        
    Note that this function uses ONLY the histories! an additional condition for 
    using A' is that the particle is actually close right now! 
    this is checked in get motion model directly. 
    """
    
    ### determine bools for the edge cases
    always_close = np.min(closeArr, axis = 0) # was a particle always close?
                                              # because minimum of closeArr is only true 
                                              # if particle has always true
    always_normal_model = np.max(modelArr, axis = 0) == False # was particle always normal?
                                                              # if it was, then there are only false in the history, 
                                                              # and our maximum is false
        
    ### for the normal case, which is described in Yan Christmas Kittler...
    t1 = np.argmin(closeArr, axis = 0) ### for each particle, most recent time at not close.
    t2 = np.argmax(modelArr, axis = 0) ### for each particle, most recent time at which A' used.
    
    possible_model_switch = t1 < t2 ### note that this is inverted versus paper
                                    ### this is because paper counts time forward, 
                                    ### but we count backwards!
    
    propagation_rules = np.where(always_normal_model, True, # CASE 1 and CASE 3
                                 np.where(always_close, False,  # CASE 2
                                          possible_model_switch)) # CASE 4
    return propagation_rules



# chamfer distance
def chamfer_distance(gFrm, topplayer, botplayer, threshold = 100, verbose = False):
    pMask = gFrm.within_contour(topplayer) | gFrm.within_contour(botplayer)    
    dist = cv.distanceTransform((~pMask).array, cv.DIST_L2, 3,  dstType=cv.CV_32F)
    dMask = (Frame(dist) < threshold)
    if verbose: dMask.show()
    return dMask


##### define markov process and observation rules #####
def get_motion_model(S, chamferMask, closeHistories = None, modelHistories = None):
    ### Check the histories for the particle, to see if it would qualify for a model switch
    if closeHistories is None and modelHistories is None:
        propagation_rules = np.ones(len(S), dtype = 'bool') ### if we have no history,
                                                            ### every particle can potentially 
                                                            ### be moved with A or A' (if it's close)
    else:
        propagation_rules = compare_histories(closeHistories, modelHistories)
    
    ### Find Chamfer-Appropriate Model
    As = []
    closeBools = np.array([], dtype = 'bool')
    modelBools = np.array([], dtype = 'bool')
    for s, switchBool in zip(S, propagation_rules):
        ### determine closeness
        x, y = s[:2].squeeze()
        sPt = Point((round(x), round(y)))
        if not chamferMask.eval_at(sPt): ### not close to player
            ball_is_close = False
            model_is_prime = False    
            A_star = A    
        else:
            ball_is_close = True
            
            if switchBool: ### particle qualifies for a model switch! 
                if np.random.rand() < P_PRIME:
                    A_star = A_primeprime
                    model_is_prime = True
                else:
                    A_star = A
                    model_is_prime = False
            else:          ### independent of whether the particle is close or not, 
                           ### it must be moved with the normal model A
                A_star = A
                model_is_prime = False
                         
        As.append(A_star) ### do not turn this into an np.array! appending a matrix to a list
                            # flattens the matrix...
        closeBools = np.append(closeBools, ball_is_close)
        modelBools = np.append(modelBools, model_is_prime)

    return As, closeBools, modelBools

def get_means_and_weights(S, As, ballContours):
    N = len(S)
    
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
    N = len(wDict[0])
    wArr = normalize_weights(wDict)
    
    ### Resample
    contourList = []
    motherList = []
    sampleList = []
    for k in range(N):
        sampleIdx = np.random.choice(N * len(wDict.keys()), p = wArr)
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


def zero_weight_for_external_particles(mDict, wDict, rows = 1080, cols = 1920):
    ### quick fix for particle dispersion issue
    newweights = []
    for m, w in zip(mDict[0], wDict[0]):
        x, y = np.asarray(m[:2]).squeeze()
        if (0 < x < cols) and (0 < y < rows):
            newweights.append(w)
        else:
            newweights.append(0)
    wDict[0] = newweights
    return wDict

def init_or_append_array(inArr, addList, dtype = 'bool'):
    if inArr is None:
        return np.array([addList], dtype = dtype)
    else:
        return np.vstack([addList, inArr]) ### really important: 
                                            ## for 'compare_histories' to work, 
                                            ## we need the most recent observations at the top
                                            ## since we want to look thru history backwards,
                                            ## and argmax does just that.

def update_histories(motherArr, closeArr, modelArr,  ### the histories, so far.
                     I_out, closeBools, modelBools): ### the new additions

    ### parent particle index history
    motherArr = init_or_append_array(motherArr, I_out)
    
    ### select appropriate close and model choice indicators
    closeBools_resampled = closeBools[I_out]
    modelBools_resampled = modelBools[I_out]
    
    ### update close and model choice histories
    closeArr = init_or_append_array(closeArr, closeBools_resampled)
    modelArr = init_or_append_array(modelArr, modelBools_resampled)

    return motherArr, closeArr, modelArr

def particles_to_points(S, particles_to_produce = None):
    def round_tup(tup):
        return round(tup[0]), round(tup[1])
    
    if not particles_to_produce is None:
        to_draw = S[:particles_to_produce]
    else:
        to_draw = S.copy()
    coords = to_draw[:, :2, :]
    points = [Point(round_tup(c.squeeze())) for c in coords]
    return points

def average_particles(S, particles_to_produce = 100):
    bucket_length = len(S) // particles_to_produce
    bucketLists = [S[b:b + bucket_length] for b in range(particles_to_produce)]
    avgLists = np.array([np.mean(bucket, axis = 0) for bucket in bucketLists])
    return avgLists

def plot_particles(S, particles_to_produce = 500):
    return particles_to_points(average_particles(S, particles_to_produce = particles_to_produce))


