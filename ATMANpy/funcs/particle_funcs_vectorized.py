# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:55:01 2022

Faster, vectorized versions of the particle filter in ball_tracking_pfilter_prototype


@author: gianl
"""

import numpy as np

import cv2 as cv

from ATMANpy.types.Frames import Frame
from ATMANpy.types.Points import Point
from ATMANpy.types.Contours import Contour


# the normal movement model
A = np.array([[1, 0, 1, 0], 
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

# the close-to-player movement model
A_prime = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]) 

# the close-to-player movement model with reversed thrust
A_primeprime = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]]) 

# the matrix that turns true state x into obversation z (which is only a location!)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

PROCESS_VARIANCE = 100
OBSERVATION_VARIANCE = 100

Q = np.eye(4) * PROCESS_VARIANCE
R = np.eye(2) * OBSERVATION_VARIANCE

LAMBDA = 0.0000001 ### spacial poisson density of clutter... whatever that is. See isard & blake (1998)
P_NO_BALL = 0.25 ### probability that ball is not among tracked objects
beta = LAMBDA * P_NO_BALL

P_PRIME = 0.3

#### auxiliaries ###

weight_cov = R + H @ Q @ H.T
weight_cov_inv = np.linalg.inv(weight_cov)

weight_denominator = 2 * np.pi * np.linalg.det(weight_cov) ** 0.5 ### not quite correct! 
                                                                  # pi must be raised to the power of number of dimensions, 
                                                                  # and then also be square-rooted. 
# HA = H * A

K = Q @ H.T @ weight_cov_inv
C = (np.eye(4) - K @ H) @ Q

R_inv = np.linalg.inv(R)
likelihood_denominator = 2 * np.pi * np.linalg.det(R) ** 0.5 ### not quite correct! 
                                                             # pi must be raised to the power of number of dimensions, 
                                                             # and then also be square-rooted. 

#### CHAMFER ###
CHAMFER_THRESH = 50


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
def chamfer_distance(gFrm, topplayer, botplayer, 
                     threshold = None, 
                     verbose = False):
    if threshold is None:
        threshold = CHAMFER_THRESH
    
    pMask = gFrm.within_contour(topplayer) | gFrm.within_contour(botplayer)    
    dist = cv.distanceTransform((~pMask).array, cv.DIST_L2, 3,  dstType=cv.CV_32F)
    dMask = (Frame(dist) < threshold)
    if verbose: dMask.show()
    return dMask


##### define markov process and observation rules #####
def get_motion_model(S, chamferMask, closeHistories = None, modelHistories = None,
                     normal_propagation = None, close_propagation = None, switch_probability = None):
    if normal_propagation is None:
        normal_propagation = A
    if close_propagation is None:
        close_propagation = A_primeprime
    if switch_probability is None:
        switch_probability = P_PRIME
    
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
            A_star = normal_propagation   
        else:
            ball_is_close = True
            
            if switchBool: ### particle qualifies for a model switch! 
                if np.random.rand() < switch_probability:
                    A_star = close_propagation
                    model_is_prime = True
                else:
                    A_star = normal_propagation
                    model_is_prime = False
            else:          ### independent of whether the particle is close or not, 
                           ### it must be moved with the normal model A
                A_star = normal_propagation
                model_is_prime = False
                         
        As.append(A_star)
        closeBools = np.append(closeBools, ball_is_close)
        modelBools = np.append(modelBools, model_is_prime)

    As = np.array(As)
    return As, closeBools, modelBools


# S = initialize_particles(gFrm, courtLines)
# chamferMask = chamfer_distance(gFrm, topplayerRegion, botplayerRegion)
# As, closeBools, modelBools = get_motion_model(S, chamferMask)


def get_means_and_weights(S, As, ballContours,
                          beta = beta, H = H, K = K,
                          weight_cov_inv = weight_cov_inv, 
                          weight_denominator = weight_denominator,
                          rows = 1080, cols = 1920):
    # S.shape ### n,j -> j = 4
    # As.shape ### n,i,j -> i = j = 4. But note: for einstein sums, we need to multiply the vector rows with the matrix columns
    
    ### the m0 vector -> particles updated with appropriate A_star model.
    matVecMulStr = 'nij,nj -> ni'
    m0 = np.einsum(matVecMulStr, As, S)
    
    ### the upper two elements of the m0 vector -> location of updated particles.
    m0_loc = m0[:, :2]
    
    if ballContours == []:
        m = m0[:, :, np.newaxis]
        w = np.zeros((m.shape[0], 1)) + beta
    
    else:
        z_coords = [list(cnt.center().coords) for cnt in ballContours]
        
        dist = np.dstack([m0_loc - xy for xy in z_coords])
        
        ###### Construct the matrix of weights #######
        # weight_cov_inv.shape ### 2 x 2
        # dist.shape ### n, 2, num_contours
        
        trans_vec_mat_mulStr = 'niz,ij->njz'
        dist_x_weight_cov_inv = np.einsum(trans_vec_mat_mulStr, dist, weight_cov_inv)
        
        vec_vec_mulStr = 'njz,njz->nz'
        weight_exp_term = np.einsum(vec_vec_mulStr,dist_x_weight_cov_inv,dist)
        
        w = np.exp(-0.5 * weight_exp_term) / weight_denominator
        w = np.insert(w,0,beta,axis=1) ### axis 1 is z, index = 0 -> prepend the betas!
        # w.shape ### -> n,z+1
        
        #### Construct the matrix of particle centers
        K.shape ### 4,2
        dist.shape ### n, 2, num_contours
        mat_vec_mulStr = 'ij,njz->niz'
        K_dist = np.einsum(mat_vec_mulStr,K,dist)
        K_dist.shape
        
        m = np.add(m0[:,:,np.newaxis], K_dist)
        m = np.insert(m,0,m0,axis=2) ### axis 2 is z, index = 0 -> prepend the m0 vector!
        # m.shape ### -> n,4,z+1  
    
    #### Weight Rectification and Normalization
    # if particle is out of bounds...
    # bad_coordinate = (m[:, :2, :] < 0) | (m[:, :2, :] > [rows - 1, cols - 1])
    x, y = m[:, 0, :], m[:, 1, :]
    bad_point = (x < 0) | (y < 0) | (x > cols) | (y > rows)
    # bad_point.shape -> n,z+1
    
    ### ... we set the weight to zero
    w = w * ~bad_point
    
    ### then, normalize weights so they add up to one
    w = w / w.sum() 
    
    return m,w


def resample(m, w, Q = Q, C = C):
    cholQ, cholC = np.linalg.cholesky(Q), np.linalg.cholesky(C)
    
    wArr = w.flatten('F') ### fortran-style ordering -> columns first!
    N,Z = w.shape
    sampleIdx = np.random.choice(N * Z, p = wArr, size = N)
    i, j = sampleIdx % N, sampleIdx // N
    
    seeds = np.random.multivariate_normal(np.zeros(4), np.eye(4), size = N)
    
    ### get appropriate lower-triangle covariance using j vector
        # j == 0 -> covariance is Q
        # j > 0 -> covariance is C = (I - KH)Q
    concatStr = 'n,ij->nij'
    cov_L = np.einsum(concatStr, j == 0, cholQ) + np.einsum(concatStr, j > 0, cholC)
    
    ### means + scaled covariances 
    means = np.array([m[n, :, z] for n,z in zip(i,j)])
    # means.shape ### -> N,Z
    
    n_matmulStr = 'nij,nj->ni' ### matrix multiplication, n times
    devs_from_mean = np.einsum(n_matmulStr, cov_L, seeds)
    # devs_from_mean.shape ### -> N,Z
    
    samples = means + devs_from_mean
    
    return samples, i, j


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

######### SMOOTHING FUNCTIONS ########

def kitagawa_smooth(stream_features_original):
    
##### Let's go backwards.
## -> u = T - t - 1: Essentially, we now kitagawa-smooth from recent to old.
## this is easier since we can get the direct parent recursively.
    
    from copy import deepcopy
    stream_features_smooth = deepcopy(stream_features_original)
    
    last_frame = sorted(list(stream_features_smooth.keys()))[-1]
    
    idxHistory = stream_features_smooth[last_frame]['ballIdxHistory'] ### the final history of each particle
    
    T, N = idxHistory.shape
    
    ### initialize for last period 
    k_iT_u = np.arange(start = 0, stop = N) # -> think: u=0
    
    for u in range(T):
        s_i = stream_features_smooth[last_frame - u]['ballParticles']
        selected_s_i = s_i[k_iT_u] ### select particles. in u = 0, selected_s_i = s_i
    
        if not s_i is None:
            stream_features_smooth[last_frame - u]['ballParticles'] = selected_s_i
        
        parentIdxs = idxHistory[u] ## the indices of direct particle parents at time u
        
        k_iT_u = parentIdxs[k_iT_u] ### find parents of permuted particles

    return stream_features_smooth


# start_features = stream_features_smooth[START_FRAME + 40]
# S, ballContours = start_features['ballParticles'], start_features['ballContours']


# start_features = stream_features_smooth[START_FRAME + 7]
# S2, ballContours2 = start_features['ballParticles'], start_features['ballContours']


def choose_likely_ball_contour(S, ballContours, l_threshold = 1/10,
                               R_inv = R_inv, 
                               likelihood_denominator = likelihood_denominator,
                               return_likelihood = False,
                               verbose = False):
    if ballContours == []:
        if verbose: print('No Contours!')
        if return_likelihood:
            return None, None, None
        else:
            return None
    else:
        m0_loc = S[:, :2]
        z_coords = [list(cnt.center().coords) for cnt in ballContours]
        dist = np.dstack([m0_loc - xy for xy in z_coords])
    
        trans_vec_mat_mulStr = 'niz,ij->njz'
        dist_x_obs_cov_inv = np.einsum(trans_vec_mat_mulStr, dist, R_inv)
        
        vec_vec_mulStr = 'njz,njz->nz'
        likelihood_exp_term = np.einsum(vec_vec_mulStr,dist_x_obs_cov_inv,dist)
        
        likelihoods = np.exp(-0.5 * likelihood_exp_term)
        likelihood = likelihoods.mean(axis = 0) / likelihood_denominator
        max_likelihood = np.max(likelihood)
        
        ### normalise the threshold, so the likelihood denominator cancels out.
        normalised_threshold = l_threshold / likelihood_denominator
        
        if max_likelihood > normalised_threshold:
            likeliest_idx = np.argmax(likelihood)
            likeliest_z = ballContours[likeliest_idx]
            if verbose: print(f'Detected a ball contour with likelihood {max_likelihood}')
            if return_likelihood:
                return likeliest_z, max_likelihood, normalised_threshold
            else:
                return likeliest_z
        else:
            if verbose: print(f'No ball, best candidate has likelihood {max_likelihood}')
            if return_likelihood:
                return None, max_likelihood, normalised_threshold
            else:
                return None

def expand_to_ellipse(myCnt):
    ### so that we may form an ellipse from both contours,
    ### we have to have at least five points on the contour
    outPts = myCnt.points
    min_x, min_y = outPts.min(axis = (0, 1))
    max_x, max_y = outPts.max(axis = (0, 1))
    
    cand_points = [[max_x, max_y], 
                   [max_x, min_y], 
                   [min_x, max_y], 
                   [min_x, min_y]] ### these points are guaranteed to be at edge
    
    add_points_emergency = [[max_x + 1, (min_y + max_y) // 2],
                            [(min_x + max_x) // 2, max_y + 1]]
    ### these are not in the contour by definition. we add them if we can't reach the required number of points.
    
    while len(outPts) < 6:
        
        for pt in cand_points:
            if [pt] in outPts.tolist():
                cand_points.remove(pt)
                continue
            else:
                outPts = np.vstack([outPts, [[pt]]])
                break
          
        if len(cand_points) == 0:
            break
          
    if len(outPts) == 4:
        outPts = np.vstack([outPts, [[add_points_emergency[0]]]])
    if len(outPts) == 5:
        outPts = np.vstack([outPts, [[add_points_emergency[1]]]])
            
    return Contour(outPts)


def interpolate_ball_features(cnt_before, cnt_after, interpolation_points = 8,
                              frames_before = None, frames_after = None):
    
    ###### Ensure Ellipticality ####
    ### so that we may form an ellipse from both contours,
    ### we have to have at least five points on the contour
    if len(cnt_before.points) < 6:
        ell_cnt_before = expand_to_ellipse(cnt_before)
    else:
        ell_cnt_before = cnt_before
    
    if len(cnt_after.points) < 6:
        ell_cnt_after = expand_to_ellipse(cnt_after)
    else:
        ell_cnt_after = cnt_after
    
    #### Sourt out Frame Count for Weighting ###
    if frames_before is None:
        frames_before = 1 ### how many frames before is the contour before from?
        # (i.e: if we interpolate a contour at t=3, and cnt_bef is from t=2, 
            # then time before is 1
        # if we interpolate a contour at t=3, and cnt_bef is from t=0,
            # then time before is 3
    if frames_after is None:
        frames_after = 1
        
    tot_frames = frames_before + frames_after
    #########    
    
    ###### Fit Ellipse Parameters #####
    xy_before, majmin_before, theta_before = ell_cnt_before.fit_ellipse()
    xy_after, majmin_after, theta_after = ell_cnt_after.fit_ellipse()
    
    x_before, y_before = xy_before
    maj_before, min_before = majmin_before
    
    x_after, y_after = xy_after
    maj_after, min_after = majmin_after
    #####
    
    ###### Weighted average of ellipse parameters #####
    def weight_avg(bef, aft):
        ### the chiasm is sic! the further away the frame after is,
        ### the more weight we give to the frame before, and vice versa.
        return bef * frames_after / tot_frames + aft * frames_before / tot_frames 
    
    x_int, y_int = weight_avg(x_before, x_after), weight_avg(y_before, y_after)
    maj_int, min_int = weight_avg(maj_before, maj_after), weight_avg(min_before, min_after)
    theta_int = weight_avg(theta_before, theta_after)
    
    semimaj_int = maj_int / 2
    semimin_int = min_int / 2
    theta_rad_int = theta_int / 180 * np.pi

    ### to go from ellipse to contour: sample points on the ellipse!
    t = np.linspace(0, 2 * np.pi, num = interpolation_points, endpoint = False)
    sin_t, cos_t = np.sin(t), np.cos(t) 
    sin_theta, cos_theta = np.sin(theta_rad_int), np.cos(theta_rad_int) 
    x = semimaj_int * cos_theta * cos_t - semimin_int * sin_theta * sin_t
    y = semimaj_int * sin_theta * cos_t + semimin_int * cos_theta * sin_t
    
    coords = np.dstack([x + x_int, y + y_int]) ### add back center points!!
    # coords.shape # -> (1, num_points, 2)
    coords = coords.reshape((interpolation_points, 1, 2)) ### this way, they look like contours
    coords = coords.astype('int32')
    
    return Contour(coords)

def interpolate_ball_features_smart(cnt_before, cnt_after, 
                                    interpolation_points = 8,
                                    speed_before = None, speed_after = None,
                                    accel_before = None, accel_after = None,
                                    frames_before = None, frames_after = None,
                                    acceptable_lambda_difference = 2,
                                    verbose = False):
    
    ###### Ensure Ellipticality ####
    ### so that we may form an ellipse from both contours,
    ### we have to have at least five points on the contour
    if len(cnt_before.points) < 5:
        ell_cnt_before = expand_to_ellipse(cnt_before)
    else:
        ell_cnt_before = cnt_before
    
    if len(cnt_after.points) < 5:
        ell_cnt_after = expand_to_ellipse(cnt_after)
    else:
        ell_cnt_after = cnt_after
    
    #### Sourt out Frame Count for Weighting ###
    if frames_before is None:
        frames_before = 1 ### how many frames before is the contour before from?
        # (i.e: if we interpolate a contour at t=3, and cnt_bef is from t=2, 
            # then time before is 1
        # if we interpolate a contour at t=3, and cnt_bef is from t=0,
            # then time before is 3
    if frames_after is None:
        frames_after = 1
        
    tot_frames = frames_before + frames_after
    #########    
    
    ###### Fit Ellipse Parameters #####
    xy_before, majmin_before, theta_before = ell_cnt_before.fit_ellipse()
    xy_after, majmin_after, theta_after = ell_cnt_after.fit_ellipse()
    
    x_before, y_before = xy_before
    maj_before, min_before = majmin_before
    
    x_after, y_after = xy_after
    maj_after, min_after = majmin_after
    #####
    
    #### Correct speeds for acceleration
    if accel_before is not None:
        speed_before_star = speed_before + accel_before ### essentially, the interpolated starting speed ...
    else:
        speed_before_star = speed_before
    
    if accel_after is not None:
        speed_after_star = speed_after - accel_after ### ... and this is the interpolated final speed.
    else:
        speed_after_star = speed_after
    
    #### Solve for Lambda, the switching point! 
    # see notes, but in principle: solve p_bef + lambda * s_bef = p_aft - (tot_frames - lambda) * s_aft
    # in principle, this will yield a vector of lambdas. We need a scalar, though. 
    # If the interpolation model is good, the vector values will be close, and we can average them.
    # if they aren't the interpolation will not be great. 
    # another danger: if speed before and after is the same, then lambda is undefined.
    # the nice thing here: in this case, the value of lambda is irrelevant. 
    # since the speeds are the same, there is no switching point.
    
    ## a) check speed difference
    speed_diff = (speed_before_star - speed_after_star)
    if speed_diff.x != 0:
        l_x = (x_after - x_before - tot_frames * speed_after_star.x) / speed_diff.x
        
        ### plausibility check
        if not 0 < l_x < tot_frames:
            return interpolate_ball_features(cnt_before, cnt_after, 
                                             interpolation_points = interpolation_points,
                                             frames_before = frames_before, frames_after = frames_after)
    else:
        l_x = None
        
    if speed_diff.y != 0:
        l_y = (y_after - y_before - tot_frames * speed_after_star.y) / speed_diff.y
        
        ### plausibility check
        if not 0 < l_y < tot_frames:
            return interpolate_ball_features(cnt_before, cnt_after, 
                                             interpolation_points = interpolation_points,
                                             frames_before = frames_before, frames_after = frames_after)
    else:
        l_y = None
    
    ## b) combine lambdas
    if l_x is None and l_y is None:
        l = tot_frames / 2
    elif l_x is None:
        l = l_y
    elif l_y is None:
        l = l_x
    else:
        # weight_x = abs(speed_diff.x)
        # weight_y = abs(speed_diff.y)
        # l = (weight_x * l_x + weight_y * l_y) / (weight_x + weight_y)
        l = (l_x + l_y) / 2
    
        ## check plausibility 
        if abs(l_x - l_y) > acceptable_lambda_difference:
            return interpolate_ball_features(cnt_before, cnt_after, 
                                             interpolation_points = interpolation_points,
                                             frames_before = frames_before, frames_after = frames_after) 
        
    
    if verbose:
        print('Interpolating Smart')
        print(f'   l_x: {l_x}, l_y: {l_y}, l: {l}')
        print(f'   tot_frames: {tot_frames}')
        print('____')
    
    ##### decide which 'regime' we are in: previous detected ball, or next detected ball.
    switching_point, switching_prop = l // 1 + 1, l % 1          ### divide integer and decimal, 
                                                                 ### except add one to the integer.
                                                                 ### this is because if lambda is 1.5,
                                                                 ### the appropriate switching point
                                                                 ### is when frames_before = 2.
                                                                 
    ### calculate location (and ellipse parameters) depending on switching point and proportion.
    if frames_before < switching_point:
        x_int = x_before + frames_before * speed_before_star.x
        y_int = y_before + frames_before * speed_before_star.y
        maj_int, min_int, theta_int = maj_before, min_before, theta_before
    if frames_before > switching_point:
        x_int = x_after - frames_after * speed_after_star.x
        y_int = y_after - frames_after * speed_after_star.y
        maj_int, min_int, theta_int = maj_after, min_after, theta_after
    if frames_before == switching_point:
        x_prev = x_before + (frames_before - 1) * speed_before_star.x
        y_prev = y_before + (frames_before - 1) * speed_before_star.y
        
        prop_add_on_x = switching_prop * speed_before_star.x + (1 - switching_prop) * speed_after_star.x
        prop_add_on_y = switching_prop * speed_before_star.y + (1 - switching_prop) * speed_after_star.y
        
        x_int = x_prev + prop_add_on_x
        y_int = y_prev + prop_add_on_y
        
        maj_int = switching_prop * maj_before + (1 - switching_prop) * maj_after
        min_int = switching_prop * min_before + (1 - switching_prop) * min_after
        theta_int = switching_prop * theta_before + (1 - switching_prop) * theta_after
    
    semimaj_int = maj_int / 2
    semimin_int = min_int / 2
    theta_rad_int = theta_int / 180 * np.pi

    ### to go from ellipse to contour: sample points on the ellipse!
    t = np.linspace(0, 2 * np.pi, num = interpolation_points, endpoint = False)
    sin_t, cos_t = np.sin(t), np.cos(t) 
    sin_theta, cos_theta = np.sin(theta_rad_int), np.cos(theta_rad_int) 
    x = semimaj_int * cos_theta * cos_t - semimin_int * sin_theta * sin_t
    y = semimaj_int * sin_theta * cos_t + semimin_int * cos_theta * sin_t
    
    coords = np.dstack([x + x_int, y + y_int]) ### add back center points!!
    # coords.shape # -> (1, num_points, 2)
    coords = coords.reshape((interpolation_points, 1, 2)) ### this way, they look like contours
    coords = coords.astype('int32')

    return Contour(coords)

def interpolate_ball_features_circle(cnt_before, cnt_after, interpolation_points = 6,
                                     frames_before = None, frames_after = None):
    
    #### Sourt out Frame Count for Weighting ###
    if frames_before is None:
        frames_before = 1 ### how many frames before is the contour before from?
        # (i.e: if we interpolate a contour at t=3, and cnt_bef is from t=2, 
            # then time before is 1
        # if we interpolate a contour at t=3, and cnt_bef is from t=0,
            # then time before is 3
    if frames_after is None:
        frames_after = 1
        
    tot_frames = frames_before + frames_after
    #########    
    
    ###### Fit Circle Parameters #####
    xy_before, area_before = cnt_before.center(), cnt_before.area()
    xy_after, area_after = cnt_after.center(), cnt_after.area()
    
    x_before, y_before = xy_before.coords
    x_after, y_after = xy_after.coords
    #####
    
    ###### Weighted average of ellipse parameters #####
    def weight_avg(bef, aft):
        ### the chiasm is sic! the further away the frame after is,
        ### the more weight we give to the frame before, and vice versa.
        return bef * frames_after / tot_frames + aft * frames_before / tot_frames 
    
    x_int, y_int = weight_avg(x_before, x_after), weight_avg(y_before, y_after)
    area_int = weight_avg(area_before, area_after)

    BALL_MIN_RADIUS = 3
    r_int = int(max((area_int / np.pi) ** 0.5, BALL_MIN_RADIUS))
    
    ### to go from ellipse to contour: sample points on the ellipse!
    t = np.linspace(0, 2 * np.pi, num = interpolation_points, endpoint = False)
    sin_t, cos_t = np.sin(t), np.cos(t) 
    x = x_int + r_int * cos_t
    y = y_int + r_int * sin_t
    
    coords = np.dstack([x, y]) ### add back center points!!
    # coords.shape # -> (1, num_points, 2)
    coords = coords.reshape((interpolation_points, 1, 2)) ### this way, they look like contours
    coords = coords.astype('int32')
    
    return Contour(coords)

def interpolate_ball_features_smart_circle(cnt_before, cnt_after, 
                                           interpolation_points = 6,
                                           speed_before = None, speed_after = None,
                                           accel_before = None, accel_after = None,
                                           frames_before = None, frames_after = None,
                                           acceptable_lambda_difference = 3,
                                           verbose = False):
    
    #### Sourt out Frame Count for Weighting ###
    if frames_before is None:
        frames_before = 1 ### how many frames before is the contour before from?
        # (i.e: if we interpolate a contour at t=3, and cnt_bef is from t=2, 
            # then time before is 1
        # if we interpolate a contour at t=3, and cnt_bef is from t=0,
            # then time before is 3
    if frames_after is None:
        frames_after = 1
        
    tot_frames = frames_before + frames_after
    #########    
    
    ###### Fit Circle Parameters #####
    xy_before, area_before = cnt_before.center(), cnt_before.area()
    xy_after, area_after = cnt_after.center(), cnt_after.area()
    
    x_before, y_before = xy_before.coords
    x_after, y_after = xy_after.coords
    #####
    
    #### Correct speeds for acceleration
    if accel_before is not None:
        speed_before_star = speed_before + accel_before ### essentially, the interpolated starting speed ...
    else:
        speed_before_star = speed_before
    
    if accel_after is not None:
        speed_after_star = speed_after - accel_after ### ... and this is the interpolated final speed.
    else:
        speed_after_star = speed_after
    
    #### Solve for Lambda, the switching point! 
    # see notes, but in principle: solve p_bef + lambda * s_bef = p_aft - (tot_frames - lambda) * s_aft
    # in principle, this will yield a vector of lambdas. We need a scalar, though. 
    # If the interpolation model is good, the vector values will be close, and we can average them.
    # if they aren't the interpolation will not be great. 
    # another danger: if speed before and after is the same, then lambda is undefined.
    # the nice thing here: in this case, the value of lambda is irrelevant. 
    # since the speeds are the same, there is no switching point.
    
    ## a) check speed difference
    speed_diff = (speed_before_star - speed_after_star)
    if speed_diff.x != 0:
        l_x = (x_after - x_before - tot_frames * speed_after_star.x) / speed_diff.x
        
        ### plausibility check
        if not 0 <= l_x <= tot_frames:
            if verbose: print(f'l_x {l_x} out of scope ({tot_frames})')
            return interpolate_ball_features_circle(cnt_before, cnt_after, 
                                             interpolation_points = interpolation_points,
                                             frames_before = frames_before, frames_after = frames_after)
    else:
        l_x = None
        
    if speed_diff.y != 0:
        l_y = (y_after - y_before - tot_frames * speed_after_star.y) / speed_diff.y
        
        ### plausibility check
        if not 0 <= l_y <= tot_frames:
            if verbose: print(f'l_y {l_y} out of scope ({tot_frames})')
            return interpolate_ball_features_circle(cnt_before, cnt_after, 
                                             interpolation_points = interpolation_points,
                                             frames_before = frames_before, frames_after = frames_after)
    else:
        l_y = None
    
    ## b) combine lambdas
    if l_x is None and l_y is None:
        l = tot_frames / 2
    elif l_x is None:
        l = l_y
    elif l_y is None:
        l = l_x
    else:
        # weight_x = abs(speed_diff.x)
        # weight_y = abs(speed_diff.y)
        # l = (weight_x * l_x + weight_y * l_y) / (weight_x + weight_y)
        l = (l_x + l_y) / 2
    
        ## check plausibility 
        if abs(l_x - l_y) > acceptable_lambda_difference:
            if verbose: print(f'Difference between l_x ({l_x}) and l_y ({l_y}) too large.')
            return interpolate_ball_features_circle(cnt_before, cnt_after, 
                                                    interpolation_points = interpolation_points,
                                                    frames_before = frames_before, frames_after = frames_after) 
        
    
    if verbose:
        print('Interpolating Smart')
        print(f'   l_x: {l_x}, l_y: {l_y}, l: {l}')
        print(f'   tot_frames: {tot_frames}')
        print('____')
    
    ##### decide which 'regime' we are in: previous detected ball, or next detected ball.
    switching_point, switching_prop = l // 1 + 1, l % 1          ### divide integer and decimal, 
                                                                 ### except add one to the integer.
                                                                 ### this is because if lambda is 1.5,
                                                                 ### the appropriate switching point
                                                                 ### is when frames_before = 2.
                                                                 
    ### calculate location (and ellipse parameters) depending on switching point and proportion.
    if frames_before < switching_point:
        x_int = x_before + frames_before * speed_before_star.x
        y_int = y_before + frames_before * speed_before_star.y
        area_int = area_before
    if frames_before > switching_point:
        x_int = x_after - frames_after * speed_after_star.x
        y_int = y_after - frames_after * speed_after_star.y
        area_int = area_after
    if frames_before == switching_point:
        x_prev = x_before + (frames_before - 1) * speed_before_star.x
        y_prev = y_before + (frames_before - 1) * speed_before_star.y
        
        prop_add_on_x = switching_prop * speed_before_star.x + (1 - switching_prop) * speed_after_star.x
        prop_add_on_y = switching_prop * speed_before_star.y + (1 - switching_prop) * speed_after_star.y
        
        x_int = x_prev + prop_add_on_x
        y_int = y_prev + prop_add_on_y
        
        area_int = switching_prop * area_before + (1 - switching_prop) * area_after
        
    BALL_MIN_RADIUS = 3
    r_int = int(max((area_int / np.pi) ** 0.5, BALL_MIN_RADIUS))
        
    ### to go from ellipse to contour: sample points on the ellipse!
    t = np.linspace(0, 2 * np.pi, num = interpolation_points, endpoint = False)
    sin_t, cos_t = np.sin(t), np.cos(t) 
    x = x_int + r_int * cos_t
    y = y_int + r_int * sin_t
    
    coords = np.dstack([x, y]) ### add back center points!!
    # coords.shape # -> (1, num_points, 2)
    coords = coords.reshape((interpolation_points, 1, 2)) ### this way, they look like contours
    coords = coords.astype('int32')

    return Contour(coords)


########## DRAW FUNCTIONS #########

def particles_to_points(S, particles_to_produce = None):
    def round_tup(tup):
        return round(tup[0]), round(tup[1])
    
    if not particles_to_produce is None:
        to_draw = S[:particles_to_produce]
    else:
        to_draw = S.copy()
    coords = to_draw[:, :2]
    points = [Point(round_tup(c.squeeze())) for c in coords]
    return points


def particles_to_area(S, particles_to_produce = None):
    def round_tup(tup):
        return round(tup[0]), round(tup[1])
    
    if not particles_to_produce is None:
        to_draw = S[:particles_to_produce]
    else:
        to_draw = S.copy()
    coords = to_draw[:, :2]
    mean, std = np.mean(coords, axis = 0), np.std(coords, axis = 0)
    
    return tuple(mean), tuple(std)

def average_particles(S, particles_to_produce = 100):
    bucket_length = len(S) // particles_to_produce
    bucketLists = [S[b:b + bucket_length] for b in range(particles_to_produce)]
    avgLists = np.array([np.mean(bucket, axis = 0) for bucket in bucketLists])
    return avgLists

def plot_particles(S, particles_to_produce = 500):
    return particles_to_points(average_particles(S, particles_to_produce = particles_to_produce))


