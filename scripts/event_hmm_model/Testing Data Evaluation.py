# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 03:33:20 2022

@author: gianl
"""

import os
import pandas as pd
import numpy as np

from ATMANpy.funcs.HMM.utils import read_hmm_params, read_train_test

from ATMANpy.funcs.util_funcs import get_video        
from ATMANpy.funcs.cv_funcs import show_video, get_frame, show, show_continuous

from ATMANpy.funcs.draw_funcs import draw

from ATMANpy.funcs.cv_funcs import replay


VIDEO_ID = 'pqxAmUWIoKc'
MODEL_NAME = 'embedded_cyc3_nrel85'

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = get_video(VIDEO_ID)
################

### STORENAME ###
STORE_NAME = '10to100_ok2'
#################

train_data, test_data = read_train_test(MODEL_NAME)
hmmDict = read_hmm_params(MODEL_NAME)

from ATMANpy.funcs.HMM.TokenPass import read_grammar, infer, ModelShell
grammarStr = 'full_grammar_test'
grammar = read_grammar(grammarStr)

from ATMANpy.funcs.HMM.HVite import token_pass_sequence

#%% MAKE INFERENCE ON TEST CASES

### GET STREAM FEATURES ###
from ATMANpy.funcs.feature_funcs import load_stored_features
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)

### FIGURE OUT WHAT WAS INFERRED ALREADY, AND REMOVE FROM TO_DO
inferred_features = load_stored_features(VIDEO_ID, STORE_NAME, 
                                         featnameList = ['arbitrary_event', 'grammatic_event'])

test_data_to_do = test_data[~test_data.index.isin(inferred_features)]
sequences_to_do = test_data_to_do.IndexSF.unique()
to_infer = [sequences_to_do[-1]] ### just do one at a time


### ENRICH STREAM FEATURES WITH INFERENCE ###
from ATMANpy.funcs.HMM.HPost import expand_inferred_sequence

infer_arbitrary_dict = {}
infer_grammatic_dict = {}
for test_sequence in to_infer:
    print(f'Doing Test Sequence {test_sequence}...')
    
    randSeq = test_data[test_data.IndexSF == test_sequence]
    words = randSeq.drop(columns = ['Event', 'IndexSF'])

    print('    Doing arbitrary inference...')
    arbitrary_sequence = token_pass_sequence(hmmDict, words)
    print('    Doing grammatical inference...')
    grammatic_sequence = infer(words, hmmDict, grammar)    

    infer_arbitrary_dict[test_sequence] = arbitrary_sequence
    infer_grammatic_dict[test_sequence] = grammatic_sequence
    
    frame_min, frame_max = words.index.min(), words.index.max()
    
    arbitrary_expanded = expand_inferred_sequence(arbitrary_sequence, 
                                                  word_beginning_frame = frame_min)
    grammatic_expanded = expand_inferred_sequence(grammatic_sequence, 
                                                  word_beginning_frame = frame_min)

    for frame, inferred_event in arbitrary_expanded.items():
        stream_features[frame]['arbitrary_event'] = inferred_event
        # if frame in arbitrary_sequence:
        #     stream_features[frame]['arbitrary_event_unexpanded'] = arbitrary_sequence[frame]
        
    for frame, inferred_event in grammatic_expanded.items():
        stream_features[frame]['grammatic_event'] = inferred_event
        # if frame in arbitrary_sequence:
        #     stream_features[frame]['grammatic_event_unexpanded'] = grammatic_sequence[frame]
    

#%% WRITE STREAM FEATURES

from ATMANpy.funcs.feature_funcs import save_events

inferred_features = {frame: featDict for frame, featDict in stream_features.items() if 'grammatic_event' in featDict}
save_events(inferred_features, 'grammatic_event', VIDEO_ID, STORE_NAME)

inferred_features = {frame: featDict for frame, featDict in stream_features.items() if 'arbitrary_event' in featDict}
save_events(inferred_features, 'arbitrary_event', VIDEO_ID, STORE_NAME)


#%% VISUAL INSPECTION
import cv2 as cv

######## GET SCHEMA ########
from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)
    
### CHOOSE A SEQUENCE ###
test_data.IndexSF.unique()
TEST_SEQUENCE = 94
#########################

## PREPARE 3D FEATURES ##
testSeq = test_data[test_data.IndexSF == TEST_SEQUENCE]
words = testSeq.drop(columns = ['Event', 'IndexSF'])

# inferred_events = infer_grammatic_dict[TEST_SEQUENCE]
    
test_sequence_features = {frame: featDict for frame, featDict in stream_features.items() if frame in words.index}
event_features = {frame: featDict['grammatic_event'] for frame, featDict in test_sequence_features.items()}

import ATMANpy.funcs.HMM.HPost as HPost

projDict = HPost.make_projection_matrices(cap, test_sequence_features, schemaLines)

players_3D = HPost.find_player_positions_3D(projDict, test_sequence_features)

# inf_words = HPost.merge_words_and_inference(words, inferred_events)
inf_words = HPost.merge_words_and_event_features(words, event_features)
inf_words = HPost.init_spatial_dimensions(inf_words)
inf_words = HPost.impose_serve(inf_words, schemaLines)
inf_words = HPost.impose_net(inf_words, schemaLines)
inf_words = HPost.impose_bounce(inf_words, bounce_height = 0.2)
inf_words = HPost.impose_hit(inf_words, players_3D)

ball_3D = HPost.find_known_ball_positions_3D(inf_words, projDict)
ball_3D = HPost.interpolate_ball_positions_3D(ball_3D)

features_3D = HPost.make_3D_features(inf_words, ball_3D, players_3D)
########################

#%% Smarter Interpolation of Ball in Flight

# Idea: Ball is on a ray for each frame between points where the ball location is known
# Additionally, we know that the ball, projected onto the ground, has to be on the line between
# the two known ball locations --> no sidespin!
# With this, we should be able to identify the ball height.
ball_3D_smart = HPost.find_known_ball_positions_3D(inf_words, projDict)

internalDict, externalDict = HPost.make_internal_external_matrices(projDict)
#### many of these are nan... we could do a two-pass thing: 
    # on the first pass: find the internal params by averaging over what is not nan (since they should never change )
    # on the second pass: divide proj matrix by average.
    ### but now i see that usually, only external matrix is nan... weird.
    
internalDict = HPost.interpolate_matrices(internalDict)
externalDict = HPost.interpolate_matrices(externalDict)

    
def get_camera_ray_params(somePt, M_int):
    ### this could potentially be improved by exploiting that M_int[0,1] is NOT zero, 
    ### as it usually is in the pinhole camera model.
    
    u, v = somePt.coords
    o_x, o_y = M_int[0, 2], M_int[1, 2]
    f_x, f_y = M_int[0, 0], M_int[1, 1]
    z_norm = M_int[2, 2]
    
    coeff_x_c = (z_norm * (u - o_x)) / f_x
    coeff_y_c = (z_norm * (v - o_y)) / f_y
    coeff_z_c = 1       ### since z_c is distance from camera aperture in camera coordinates, 
                        # and we need a parameter for the line, we choose z_c for convenience. 
                        # Note: z_c > 0 -> line_parameter > 0
    
    slope = np.array([coeff_x_c, coeff_y_c, coeff_z_c])

    intercept = np.zeros(3)
    return slope, intercept
    
def get_world_ray_params(m_c, M_ext):
    R, t = M_ext[:-1, :-1], M_ext[:-1:, -1]
    
    slope = R.T @ m_c
    intercept = - R.T @ t

    return slope, intercept

def calculate_interpolation_lambda(m_w, b_w, lsPt, nxPt):
    x1, y1 = lsPt['x_3D'], lsPt['y_3D']
    xn, yn = nxPt['x_3D'], nxPt['y_3D']

    m1, m2 = m_w[0], m_w[1]
    b1, b2 = b_w[0], b_w[1]

    num = y1 - m2/m1*x1 + m2/m1*b1 - b2
    den = m2/m1*(xn-x1) - (yn-y1)
    
    lmbd = num/den
    
    return lmbd

def calculate_interpolated_point(m_w, b_w, lsPt, nxPt):
    lmbd = calculate_interpolation_lambda(m_w, b_w, lsPt, nxPt)
    
    x1, y1 = lsPt['x_3D'], lsPt['y_3D']
    xn, yn = nxPt['x_3D'], nxPt['y_3D']
    
    m1, m2 = m_w[0], m_w[1]
    b1, b2 = b_w[0], b_w[1]
    
    x = x1 + lmbd * (xn - x1)
    y = y1 + lmbd * (yn - y1)
    
    z_c = (x - b1) / m1 ### this is still in camera coords!
    # z_c = (y - b2) / m2 ### also possible, results should be pretty much the same.
    z = z_c * m_w[2] + b_w[2]
    
    return x, y, z

from ATMANpy.types.Points import Point
from ATMANpy.funcs.util_funcs import find_last_and_next_list_member

known_frames = ball_3D_smart.dropna().index
for frame, row in ball_3D_smart.iterrows():
    if not frame in known_frames:
        ### here, we need to interpolate!
        ls_frame, nx_frame = find_last_and_next_list_member(frame, known_frames)
        if ls_frame is not None and nx_frame is not None:
            ### if both last and next position are known, we can interpolate
            lsPt = ball_3D_smart.loc[ls_frame]
            nxPt = ball_3D_smart.loc[nx_frame]
            
            ### now, use the internal and external camera matrices on the frame in question...
            M_int, M_ext = internalDict[frame], externalDict[frame]
            ball_data = inf_words.loc[frame]
            ballPt = Point((ball_data['x'], ball_data['y']))
            
            ### ...to derive the ray on which the ball has to lie
            m_c, b_c = get_camera_ray_params(ballPt, M_int)
            m_w, b_w = get_world_ray_params(m_c, M_ext)
        
            ### together with knowledge about the last and the next point that is known, 
            ### we can interpolate!
            print(calculate_interpolation_lambda(m_w, b_w, lsPt, nxPt))
            
            ball_3D_smart.loc[frame] = calculate_interpolated_point(m_w, b_w, lsPt, nxPt)
                
ball_3D_smart = HPost.interpolate_ball_positions_3D(ball_3D_smart) ### just for backward and forward interpolation    

features_3D_smart = HPost.make_3D_features(inf_words, ball_3D_smart, players_3D)

#%% #### VISUALIZE #########
frame_min, frame_max = min(words.index), max(words.index)
show_video(cap, 
           start_frame = frame_min, stop_frame = frame_max,
           ms_between_frames = 50)

for frame in features_3D.keys():
    
    backgroundFrm = schemaFrm.to_BGR()
    frmDict = {'cFrm': backgroundFrm}
    featDict = features_3D[frame]
    
    drawFunc = draw(['ball', 'playerTop', 'playerBot', 
                     'goodBounces', 'badBounces',
                     'ballSpeed', 'stroke']) ### takes frmDict and featDict, returns show frame
    
    showFrm = drawFunc(frmDict, featDict)
    
    show_continuous(showFrm.array, ms_between_frames = 75)
    
cv.destroyAllWindows()

for frame in features_3D_smart.keys():
    
    backgroundFrm = schemaFrm.to_BGR()
    frmDict = {'cFrm': backgroundFrm}
    featDict = features_3D[frame]
    
    drawFunc = draw(['ball', 'playerTop', 'playerBot', 
                     'goodBounces', 'badBounces',
                     'ballSpeed', 'stroke']) ### takes frmDict and featDict, returns show frame
    
    showFrm = drawFunc(frmDict, featDict)
    
    show_continuous(showFrm.array, ms_between_frames = 75)
    
cv.destroyAllWindows()
    

#%% Smarter Interpolation of Ball in Flight -- Testing

# Idea: Ball is on a ray for each frame between points where the ball location is known
# Additionally, we know that the ball, projected onto the ground, has to be on the line between
# the two known ball locations --> no sidespin!
# With this, we should be able to identify the ball height.

frame = 94100
P = projDict[frame]

np.linalg.qr(P.T)

def rq_decompose(P):
    ### based on ChatGPT
    
    R, Q = np.linalg.qr(P.T)
    
    K, R = R, Q.T ## rq -> right triangular, orthonormal
                    # Calibration Matrix K is right triangular
                    # Rotation Matrix R is orthonormal ---> inverse is transpose!

    return K, R

## Testing:
K, R = rq_decompose(P)

P_reconstructed = np.dot(K, R).T
print("Reconstructed P matrix:")
print(P_reconstructed)
print("Is P equal to reconstructed P?")
print(np.allclose(P, P_reconstructed))
#### -> not good...

import scipy as sp
def find_intrinsic_extrinsic_M(P, normalize_K = True):
    
    P_left, P_last = P[:, :-1], P[:, -1]
    
    K, R = sp.linalg.rq(P_left)  ## rq -> right triangular, orthonormal
                                    # Calibration Matrix K is right triangular
                                    # Rotation Matrix R is orthonormal ---> inverse is transpose!
    if normalize_K: K = K / K[2,2]
                                    
    t = np.linalg.inv(K) @ P_last
    
    M_int = np.append(K, np.zeros((3, 1)), axis = 1)
    M_ext = np.append(R, t[:, np.newaxis], axis = 1)
    M_ext = np.append(M_ext, np.array([[0, 0, 0, 1]]), axis = 0)
    
    return M_int, M_ext

M_int, M_ext = find_intrinsic_extrinsic_M(P, normalize_K=False)
print(M_int)

np.allclose(M_int @ M_ext, P)

def find_intrinsic_extrinsic_M(P, normalize_K = True):
        
    import cv2 as cv
    res = cv.decomposeProjectionMatrix(P)
    
    K, R, T = res[0], res[1], res[2]
    if normalize_K: K = K / K[2,2]
    t = T[:3] / T[3]    
    
    M_int = np.append(K, np.zeros((3, 1)), axis = 1)
    M_ext = np.append(R, t, axis = 1)
    M_ext = np.append(M_ext, np.array([[0, 0, 0, 1]]), axis = 0)
    
    return M_int, M_ext

M_int, M_ext = find_intrinsic_extrinsic_M(P, normalize_K=True)
print(M_int)
print(M_ext)

np.allclose(M_int @ M_ext, P)

def decomposeP(P, normalize_K = True):
    import numpy as np
    M = P[0:3,0:3]
    Q = np.eye(3)[::-1]
    P_b = Q @ M @ M.T @ Q
    K = Q @ np.linalg.cholesky(P_b) @ Q
    if normalize_K: K = K / K[2,2]
    A = np.linalg.inv(K) @ M
    l = (1/np.linalg.det(A)) ** (1/3)
    R = l * A
    t = l * np.linalg.inv(K) @ P[0:3,3]
    return K, R, t

K, R, t = decomposeP(P)
K @ R
P[:, :-1]

def find_intrinsic_extrinsic_M(P, normalize_K = True):
        
    K, R, t = decomposeP(P, normalize_K = normalize_K)
    
    M_int = np.append(K, np.zeros((3, 1)), axis = 1)
    M_ext = np.append(R, t[:, np.newaxis], axis = 1)
    M_ext = np.append(M_ext, np.array([[0, 0, 0, 1]]), axis = 0)
    
    return M_int, M_ext

M_int, M_ext = find_intrinsic_extrinsic_M(P, normalize_K=True) ### if normalize_K is put to false,
                                                                ## we get the exact p matrix!
print(M_int)
print(M_ext)

## np.allclose(M_int @ M_ext, P) -> only true if normalize_K is False.

def get_camera_ray_params(somePt, M_int):
    ### this could potentially be improved by exploiting that M_int[0,1] is NOT zero, 
    ### as it usually is in the pinhole camera model.
    
    u, v = somePt.coords
    o_x, o_y = M_int[0, 2], M_int[1, 2]
    f_x, f_y = M_int[0, 0], M_int[1, 1]
    z_norm = M_int[2, 2]
    
    coeff_x_c = (z_norm * (u - o_x)) / f_x
    coeff_y_c = (z_norm * (v - o_y)) / f_y
    coeff_z_c = 1       ### since z_c is distance from camera aperture in camera coordinates, 
                        # and we need a parameter for the line, we choose z_c for convenience. 
                        # Note: z_c > 0 -> line_parameter > 0
    
    slope = np.array([coeff_x_c, coeff_y_c, coeff_z_c])

    intercept = np.zeros(3)
    return slope, intercept
    
from ATMANpy.types.Points import Point

somePt = Point((inf_words.loc[frame, 'x'], inf_words.loc[frame, 'y']))

m_c, b_c = get_camera_ray_params(somePt, M_int)

def get_world_ray_params(m_c, M_ext):
    R, t = M_ext[:-1, :-1], M_ext[:-1:, -1]
    
    slope = R.T @ m_c
    intercept = - R.T @ t

    return slope, intercept

m_w, b_w = get_world_ray_params(m_c, M_ext)




### TEST ###
pt_true = ball_3D.loc[frame].values

z_imputed = (pt_true - b_w) / m_w ### these values need to be close together!
                                    # z is a scalar, after all.

z = z_imputed[1]

pt_imputed = m_w * z + b_w

print(pt_imputed)
print(pt_true)












