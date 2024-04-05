# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 22:40:29 2023

@author: gianl
"""


import numpy as np
import cv2 as cv

from ATMANpy.types.Frames import Frame
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Contours import Contour
from ATMANpy.types.Features import StreamFeature

class ParticleFilter:
    """
    An implementation of Yann, Christmas and Kittlers (2005) (henceforth, YCK)'s particle filter
    used for ball tracking.
    
    This class at the moment simply contains all functions previously held by particle_funcs_vectorized,
    and therefore instances do not really change state (except at initialization, when all necessary constants are created).
    For example, the function "initialize_particles" does not initialize particles on the instance, 
    but yields an array of initial particles. 
    Feel free to make this class less of a glorified module if you are bored.
    """
    
    
    A = np.array([[1, 0, 1, 0], 
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # the close-to-player movement model
    A_prime = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]) 

    ### The following three models are used jointly
    # close-to-player model with no thrust -> if the ball stays on the racket for a frame
    A_primeprime0 = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]) 
    

    # the close-to-player movement model with 1-to-1 reversed thrust
    A_primeprime1 = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, -1]]) 
    
    # the close-to-player movement model with 1-to-2 reversed thrust 
    # (motivated by observation that ball tracking is most often lost if the ball is hit away very forcefully, i.e.
    # it was still for one frame, "on the racket", and then accelerates out. In this case, we have to exaggerate the thrust reversal.)
    A_primeprime2 = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -2, 0],
                              [0, 0, 0, -2]]) 
    #########

    # the matrix that turns true state x into obversation z (which is only a location!)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    def __init__(self, 
                 process_standard_deviation = 10, ### in pixels -> one dimensional!
                 observation_standard_deviation = 10, ### in pixels -> one dimensional
                 spatial_lambda = 10 ** -7,  ### spatial poisson density of clutter... whatever that is. See isard & blake (1998)
                 p_no_ball = 0.25, ### probability that ball is not among tracked objects
                 p_prime = 0.3, ### probability of switching model if close to player (from A to A_prime or A_primeprime)
                 verbose = False,
                 ):
        
        ### Write initial parameters to self
        self.process_standard_deviation = process_standard_deviation
        self.observation_standard_deviation = observation_standard_deviation
        self.spatial_lambda = spatial_lambda
        self.p_no_ball = p_no_ball
        self.p_prime = p_prime
        
        ### Initialize Covariance Matrices of Process and Observation
        # -> Process and Observation features are for the moment assumed to be uncorrelated
        self.Q = np.eye(4) * process_standard_deviation ** 2
        self.R = np.eye(2) * observation_standard_deviation ** 2
    
        ### Initialize Beta, the "observation density caused by clutter" 
        self.beta = spatial_lambda * p_no_ball 
        
        ### Initialize some often used weights
        self.weight_cov = self.R + self.H @ self.Q @ self.H.T
        self.weight_cov_inv = np.linalg.inv(self.weight_cov)
        self.weight_denominator = 2 * np.pi * np.linalg.det(self.weight_cov) ** 0.5 ### not quite correct! 
                                                                                      # pi must be raised to the power of number of dimensions, 
                                                                                      # and then also be square-rooted. 
                                              
        ### Initialize some constants, as used in Yann Christmas Kittler
        self.K = self.Q @ self.H.T @ self.weight_cov_inv
        self.C = (np.eye(4) - self.K @ self.H) @ self.Q
        
        self.R_inv = np.linalg.inv(self.R)
        self.likelihood_denominator = 2 * np.pi * np.linalg.det(self.R) ** 0.5 ### not quite correct! 
                                                                                # pi must be raised to the power of number of dimensions, 
                                                                                # and then also be square-rooted. 
        self.verbose = verbose

    @staticmethod
    def initialize_particles(N = 1000,
                             initialization_point = None,
                             initialization_distance = None,
                             frame_width = 1920, 
                             frame_height = 1080, 
                             pixel_per_meter_horizontal = 200,
                             pixel_per_meter_vertical = 100,
                             fps = 25,
                             verbose = False):
        #### particles are four dimensional vectors, 
        #### representing location and speed in x and y coords resp.
        #### x and y are used in the traditional sense: x is horizontal, y is vertical
        if initialization_point is None:
            min_loc_x, min_loc_y = 0, 0
            max_loc_x, max_loc_y = frame_width, frame_height
        else:
            x, y = initialization_point.coords
            min_loc_x, min_loc_y = max(0, x - initialization_distance), max(0, y - initialization_distance)
            max_loc_x, max_loc_y = min(x + initialization_distance, frame_width), min(y + initialization_distance, frame_height)

        max_meter_per_frame = 70 / fps ### 260 km/h -> 70 m/s
        min_speed_x, min_speed_y = 0, 0
        max_speed_x, max_speed_y = pixel_per_meter_horizontal * max_meter_per_frame, pixel_per_meter_vertical * max_meter_per_frame
        
        min_vec = [min_loc_x, min_loc_y, min_speed_x, min_speed_y]
        max_vec = [max_loc_x, max_loc_y, max_speed_x, max_speed_y]
        
        initial_particles = np.random.uniform(min_vec, max_vec, (N, 4))
        
        if verbose: print(f'Initialized {N} particles.')

        return initial_particles
    
    
    @staticmethod
    def compare_histories(closeArr, modelArr, verbose = False):
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
        t1 = np.argmin(closeArr, axis = 0) ### for each particle, most recent time at which not close.
        t2 = np.argmax(modelArr, axis = 0) ### for each particle, most recent time at which A' used.
        
        possible_model_switch = t1 < t2 ### note that this is inverted versus paper
                                        ### this is because paper counts time forward, 
                                        ### but we count backwards!
        
        propagation_rules = np.where(always_normal_model, True, # CASE 1 and CASE 3
                                     np.where(always_close, False,  # CASE 2
                                              possible_model_switch)) # CASE 4
        
        if verbose: print(f'{propagation_rules.sum()} out of {len(propagation_rules)} particles eligible for switch!')

        return propagation_rules
        
    def get_motion_model_with_contours(self, S, chamferCnts, closeHistories = None, modelHistories = None):
        MODEL_THRESHOLD_A0 = 0.25
        MODEL_THRESHOLD_A1 = 0.75 ### these are to be calibrated if necessary!
        
        ### Check the histories for the particle, to see if it would qualify for a model switch
        if closeHistories is None and modelHistories is None:
            propagation_rules = np.ones(len(S), dtype = 'bool') ### if we have no history,
                                                                ### every particle can potentially 
                                                                ### be moved with A or A' (if it's close)
        else:
            propagation_rules = ParticleFilter.compare_histories(closeHistories, modelHistories)
        
        ### Find Chamfer-Appropriate Model
        As = []
        closeBools = np.array([], dtype = 'bool')
        modelBools = np.array([], dtype = 'bool')
        for s, switchBool in zip(S, propagation_rules):
            ### determine closeness
            x, y = s[:2].squeeze()
            sPt = Point2D((round(x), round(y)))
            
            if chamferCnts is None:
                ball_is_close = False
            else:
                ball_is_close = chamferCnts.contains(sPt)
                
            ### assign models depending on closeness    
            if not ball_is_close:   ### not close to player
                model_is_prime = False    
                A_star = self.A   
            else:                   ### close to player
                if switchBool: ### particle qualifies for a model switch! 
                    rand_0_1 = np.random.rand()
                    if rand_0_1 < self.p_prime * MODEL_THRESHOLD_A0:
                        A_star = self.A_primeprime0 ### note that our prime model of choice is actually primeprime -> thrust reversal!
                        model_is_prime = True
                    elif MODEL_THRESHOLD_A0 * self.p_prime <= rand_0_1 <  self.p_prime * MODEL_THRESHOLD_A1:
                        A_star = self.A_primeprime1 ### note that our prime model of choice is actually primeprime -> thrust reversal!
                        model_is_prime = True
                    elif MODEL_THRESHOLD_A1 * self.p_prime <= rand_0_1 <  self.p_prime:
                        A_star = self.A_primeprime2 ### note that our prime model of choice is actually primeprime -> thrust reversal!
                        model_is_prime = True
                    else:
                        A_star = self.A
                        model_is_prime = False
                else:          ### independent of whether the particle is close or not, 
                               ### it must be moved with the normal model A
                    A_star = self.A
                    model_is_prime = False
                             
            As.append(A_star)
            closeBools = np.append(closeBools, ball_is_close)
            modelBools = np.append(modelBools, model_is_prime)

        As = np.array(As)
        Bs = np.tile(np.zeros(4), (len(S), 1)) ### no add ons necessary in chamfered mode
        return As, Bs, closeBools, modelBools
    
    def get_motion_model_precalculated(self, S, A, B, weight, closeHistories = None, modelHistories = None,
                                       verbose = False):
        ### weight is None iff the particle propagation here is always linear -> we are NOT at or around the hit frame
        ### weight > 0 iff the particle propagation here is potentially precalculated -> we are at or around hit frame
        ### the interpretation of close histories here is: has the particle been in a situation where the weight is greater than zero?
        ### This occurs if we are at frames around which a hit has been detected!
        N = len(S)
        linear_As = np.tile(self.A, (N, 1, 1))
        linear_Bs = np.tile(np.zeros(4), (N, 1))
        linear_modelBools = np.zeros(N, dtype = 'bool')
        linear_closeBools = np.zeros(N, dtype = 'bool')

        ### Check the histories for the particle, to see if it would qualify for a model switch
        if closeHistories is None and modelHistories is None:
            propagation_rules = np.ones(N, dtype = 'bool') ### if we have no history,
                                                                ### every particle can potentially 
                                                                ### be moved with A or A' (if it's close)
        else:
            propagation_rules = ParticleFilter.compare_histories(closeHistories, modelHistories)

        if weight is None: ### linear propagation!
            return linear_As, linear_Bs, linear_modelBools, linear_closeBools
        
        else:
            assert weight > 0
            
            rng = np.random.default_rng()
            eligible_to_switch_indices = np.where(propagation_rules)[0]
            switch_indices = rng.choice(eligible_to_switch_indices, size = int(weight * len(eligible_to_switch_indices)), shuffle = False, replace = False)

            if verbose:
                len_done = len(switch_indices)
                len_elig = len(eligible_to_switch_indices)
                if len_elig == 0:
                    pctStr = ''
                else:
                    pct = round(len_done / len_elig * 100, 1)
                    pctStr = f'({pct}%)'
                print(f'Switching {len_done} out of {len_elig} eligible particles {pctStr}!')

            As = linear_As
            Bs = linear_Bs
            As[switch_indices] = A

            ### sample some multipliers for B, so we have a broader range of particle switches
            ### this ought to make model switching more reactive to fast shots and broad, 
            ### since the additive error doesn't do well with fast flying balls.
            b_multipliers = rng.chisquare(1, size = len(switch_indices) * 4).reshape((len(switch_indices), 4))
            ### median at k = degrees of freedom is roughly: k * (1 - 2 / (9k)) ** 3. 
            ### for k = 1, this is roughly equal to 0.47. 
            ### Therefore, adding 0.55 to the whole thing causes particles never to slow down completely, 
            ### and pushes the median to around 1. 
            b_multipliers = b_multipliers + 0.55

            Bs[switch_indices] = B * b_multipliers

            modelBools = linear_modelBools
            modelBools[switch_indices] = True

            closeBools = np.ones(N, dtype = 'bool')
            return As, Bs, closeBools, modelBools

    def get_means_and_weights(self, S, As, Bs, ballContours,
                              frame_width = 1920, 
                              frame_height = 1080,
                              rectify = True):
        # S.shape ### n,j -> j = 4
        # As.shape ### n,i,j -> i = j = 4. But note: for einstein sums, we need to multiply the vector rows with the matrix columns
        
        ### the m0 vector -> particles updated with appropriate A_star model.
        matVecMulStr = 'nij,nj -> ni'
        m0 = np.einsum(matVecMulStr, As, S) + Bs
        
        ### the upper two elements of the m0 vector -> location of updated particles.
        m0_loc = m0[:, :2]
        
        if ballContours is None or ballContours.is_empty():
            m = m0[:, :, np.newaxis]
            w = np.zeros((m.shape[0], 1)) + self.beta
        
        else:
            z_coords = [list(cnt.center().coords) for cnt in ballContours]
            
            dist = np.dstack([xy - m0_loc for xy in z_coords])
            
            ###### Construct the matrix of weights #######
            # weight_cov_inv.shape ### 2 x 2
            # dist.shape ### n, 2, num_contours
            
            trans_vec_mat_mulStr = 'niz,ij->njz'
            dist_x_weight_cov_inv = np.einsum(trans_vec_mat_mulStr, dist, self.weight_cov_inv)
            
            vec_vec_mulStr = 'njz,njz->nz'
            weight_exp_term = np.einsum(vec_vec_mulStr,dist_x_weight_cov_inv,dist)
            
            w = np.exp(-0.5 * weight_exp_term) / self.weight_denominator
            w = np.insert(w,0, self.beta, axis=1) ### axis 1 is z, index = 0 -> prepend the betas!
            # w.shape ### -> n,z+1
            
            #### Construct the matrix of particle centers
            # K.shape ### 4,2
            # dist.shape ### n, 2, num_contours
            mat_vec_mulStr = 'ij,njz->niz'
            K_dist = np.einsum(mat_vec_mulStr, self.K, dist)
            # K_dist.shape ### -> n,4,num_contours
            
            m = np.add(m0[:,:,np.newaxis], K_dist)
            m = np.insert(m,0,m0,axis=2) ### axis 2 is z, index = 0 -> prepend the m0 vector!
            # m.shape ### -> n,4,z+1  
        
        if rectify:
            #### Weight Rectification and Normalization
            # if particle is out of bounds...
            x, y = m[:, 0, :], m[:, 1, :]
            bad_point = (x < 0) | (y < 0) | (x > frame_width) | (y > frame_height)
            # bad_point.shape -> n,z+1
            
            ### ... we set the weight to zero
            w = w * ~bad_point
        
        ### then, normalize weights so they add up to one
        w = w / w.sum() 
        
        return m, w
    
    def resample(self, m, w):
        cholQ, cholC = np.linalg.cholesky(self.Q), np.linalg.cholesky(self.C)
        
        wArr = w.flatten('F') ### fortran-style ordering -> columns first!
        N, Z = w.shape
        sampleIdx = np.random.choice(N * Z, p = wArr, size = N)
        i, j = sampleIdx % N, sampleIdx // N
        
        seeds = np.random.multivariate_normal(np.zeros(4), np.eye(4), size = N)
        
        ### get appropriate lower-triangle covariance using j vector
            # j == 0 -> covariance is Q
            # j > 0 -> covariance is C = (I - KH)Q
        concatStr = 'n,ij->nij'
        cov_L = np.einsum(concatStr, j == 0, cholQ) + np.einsum(concatStr, j > 0, cholC)
        
        ### samples are means + scaled covariances 
        means = m[i, :, j]
        # means.shape ### -> N,Z
        
        n_matmulStr = 'nij,nj->ni' ### matrix multiplication, n times
        devs_from_mean = np.einsum(n_matmulStr, cov_L, seeds)
        # devs_from_mean.shape ### -> N,Z
        
        samples = means + devs_from_mean
        
        return samples, i, j


    @staticmethod
    def init_or_append_array(inArr, addList, dtype = 'bool'):
        if inArr is None:
            return np.array([addList], dtype = dtype)
        else:
            return np.vstack([addList, inArr]) ### really important: 
                                                ## for 'compare_histories' to work, 
                                                ## we need the most recent observations at the top
                                                ## since we want to look thru history backwards,
                                                ## and argmax does just that.
    
    @staticmethod
    def update_histories(motherArr, closeArr, modelArr,  ### the histories, so far.
                         I_out, closeBools, modelBools): ### the new additions
    
        ### parent particle index history
        motherArr = ParticleFilter.init_or_append_array(motherArr, I_out)
        
        ### select appropriate close and model choice indicators
        closeBools_resampled = closeBools[I_out]
        modelBools_resampled = modelBools[I_out]
        
        ### update close and model choice histories
        closeArr = ParticleFilter.init_or_append_array(closeArr, closeBools_resampled)
        modelArr = ParticleFilter.init_or_append_array(modelArr, modelBools_resampled)
    
        return motherArr, closeArr, modelArr
        
        
    def choose_likely_ball_contour(self, S, ballContours, standard_deviation_threshold = 2):
        
        if ballContours is None or ballContours.is_empty():
            return None
        else:
            m0_loc = S[:, :2]
            z_coords = [list(cnt.center().coords) for cnt in ballContours]
            dist = np.dstack([xy - m0_loc for xy in z_coords])
        
            trans_vec_mat_mulStr = 'niz,ij->njz'
            dist_x_obs_cov_inv = np.einsum(trans_vec_mat_mulStr, dist, self.R_inv)
            
            vec_vec_mulStr = 'njz,njz->nz'
            likelihood_exp_term = np.einsum(vec_vec_mulStr, dist_x_obs_cov_inv, dist)
            
            mean_unnormalised_likelihoods = np.exp(-0.5 * likelihood_exp_term).mean(axis = 0)
            
            ### Figure out threshold value ###
            # assuming uncorrelated observation error with a single standard deviation for x and y coords, we get...
            threshold_unnormalised_likelihood = np.exp(- standard_deviation_threshold ** 2)
            
            ### Figure out if there is at least one contour within the threshold
            at_least_one_contour = (mean_unnormalised_likelihoods > threshold_unnormalised_likelihood).any()
            
            if at_least_one_contour:
                likeliest_idx = np.argmax(mean_unnormalised_likelihoods)
                return ballContours[likeliest_idx]
                
            else:
                return None

    ### The StreamFeature function to run the particle algorithm
    def find_particles(self, mySF, 
                       N = 1000,
                       frame_width = 1920, 
                       frame_height = 1080, 
                       pixel_per_meter_horizontal = 200,
                       pixel_per_meter_vertical = 100,
                       fps = 25,
                       initialization_distance = 30,
                       rectify = True,
                       verbose = False):
        
        chamfered_mode = 'chamferCnts' in mySF.get_features()
        tight_init_mode = 'hitCnt' in mySF.get_features()

        indexHistories = None
        closeHistories = None
        modelHistories = None

        if tight_init_mode:
            srv_frame, srv_cnt = mySF['hitCnt'].dropna().first()
            srv_point = srv_cnt.center()
            initialization_point = srv_point
        else:
            initialization_point = None

        particles = ParticleFilter.initialize_particles(N = N,
                                                        initialization_point = initialization_point,
                                                        initialization_distance = initialization_distance,
                                                        frame_width = frame_width, 
                                                        frame_height = frame_height, 
                                                        pixel_per_meter_horizontal = pixel_per_meter_horizontal,
                                                        pixel_per_meter_vertical = pixel_per_meter_vertical,
                                                        fps = fps,
                                                        verbose = verbose)
        
        partDict = {} ### we need particles, first and foremost
        histDict = {} ### but we also need the final index histories, for smoothing
        for frame, featDict in mySF.items():
            if verbose: print(f'Frame: {frame}')

            ### write to frame!
            partDict[frame - 1] = particles.astype(int) ### for memory reasons, save as integer
                                                        ### EDIT: particles belong to previous period!

            ###### DETERMINE PARTICLE PROPAGATION RULES ########
            if chamfered_mode:    
                chamferCnts = featDict['chamferCnts']
                As, Bs, closeBools, modelBools = self.get_motion_model_with_contours(particles, chamferCnts,
                                                                                closeHistories = closeHistories,
                                                                                modelHistories = modelHistories)    
            else:
                A, B, weight = featDict['A'], featDict['B'], featDict['weight']
                As, Bs, closeBools, modelBools = self.get_motion_model_precalculated(particles, A, B, weight, 
                                                                                     closeHistories = closeHistories, 
                                                                                     modelHistories = modelHistories,
                                                                                     verbose = verbose)

            
            ########### UPDATE PARTICLES #########
            ### means_and_weights updates last periods particles, ...
            ballContours = featDict['candCnts']
            m, w = self.get_means_and_weights(particles, As, Bs, ballContours,
                                              frame_width = frame_width, frame_height = frame_height,
                                              rectify = rectify) 
            ### ... and resample compares to evidence in this period!
            particles, particleIdxs, _ = self.resample(m, w)
            
            ########### UPDATE HISTORIES #########
            indexHistories, closeHistories, modelHistories = ParticleFilter.update_histories(indexHistories, closeHistories, modelHistories,
                                                                                             particleIdxs, closeBools, modelBools) 
    
        partDict[frame] = particles.astype(int) ### for memory reasons, save as integer
        histDict[frame] = indexHistories ### these are only needed at the very end, 
                                               # since indexHistories update and accumulate
        
        partSF = StreamFeature(partDict, name = 'particles')
        histSF = StreamFeature(histDict, name = 'idxHistories')
        return partSF.merge(histSF)
