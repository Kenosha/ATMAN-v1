# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 02:26:32 2024

@author: gianl
"""

import numpy as np

class Parameters:
    def __init__(self):
        pass
    
    def __repr__(self):
        parameter_attributes = [el for el in dir(self) if el.endswith('params')]
        if len(parameter_attributes) == 0:
            return 'Parameters collection: empty'
        else:
            return f'Parameters collection holding {parameter_attributes}'
    
    def get_parameter_dictionary(self):
        parDict = {el: getattr(self, el) for el in dir(self) if el.endswith('params')}
        return parDict
    
##### GENERAL PARAMETERS

_scale_factor = 1 / 3
_width_of_line = 8 * _scale_factor

###### COURT LINES #########
    
court_lines = Parameters()

court_lines.courtline_params = {'width_of_line': _width_of_line, 
                                'int_diff_thresh': 10,
                                'white_thresh': 150,
                                'rho': 3, 'theta': np.pi / 180, 'min_votes': int(500 * _scale_factor ** 2),
                                'minLineLength': _width_of_line + 1, 'maxLineGap': 1,
                                'merge_threshold': 2 * _width_of_line, 
                                'horizontal_filter_margin': 40 * _scale_factor,
                                'vertical_filter_margin': 40 * _scale_factor,
                                'only_225': False, 
                                'verbose': False
                                }

######## COURT ROI #########

court_roi = Parameters()


court_roi.courtroi_params = {'p1': (200, 12),
                             'p2': (440, 12),
                             'p3': (0, 295),
                             'p4': (640, 295)}
    
court_roi.serveroi_params = {'top_down': 20,
                             'top_up': 60,
                             'bot_down': 30,
                             'bot_up': 82}

######## SEGMENTATION WITH LINES ######

segmentation_with_lines = Parameters()

segmentation_with_lines.court_match_params = {'width_of_line': _width_of_line, 
                                              'int_diff_thresh': 10,
                                              'white_thresh': 150,
                                              'zoom_factor': 1.2,
                                              'dil_kernel': (2, 2)}

segmentation_with_lines.threshold_params = {'fp_threshold': None, 'fn_threshold': 0.3}

segmentation_with_lines.filter_params = {'min_seq_length': 100}

###### HITS_CLASSIFY_SOUND #######

hits_classify_sound = Parameters()

hits_classify_sound.model_params = {'model_name': 'test_model_20240109_150057',
                                    'sample_rate': 44100,
                                    'fps': 25}

######## SEGMENTATION_WITH_SOUND #####

segmentation_with_sound = Parameters()

segmentation_with_sound.smooth_params = {'smoothing_kernel_audio': np.array([0.25, 0.5, 1, 0.5, 0.25])}

segmentation_with_sound.prob_params = {'next_hit_should_come_soon': 20,  ### can be early, for serves. set on the basis of above boxplot
                                       'next_hit_should_have_come': 120, ### can be late, for high lobs for example. set on the basis of above boxplot
                                       'decay_mild': 0.99,
                                       'decay_strong': 0.5,
                                       'initial_rally_probability': 0}

segmentation_with_sound.thresh_params = {'rally_prob_threshold': 0.2}

segmentation_with_sound.shift_params = {'shift_frames': 15}


########## COURT_BACKGROUND #######

court_background = Parameters()

court_background.params = {'window_range': 5, 'widen_play_by_frames': 50, 'sample_frames': 1000}

######## COURT_MASKS #########

court_masks = Parameters()

### Burn-In Mask
court_masks.burn_in_params = {'h_threshold': 16, 
                              'v_threshold': 16, 
                              'gaussian_kernel': (round(21 * _scale_factor), round(21 * _scale_factor)), 
                              'closing_kernel': (round(10 * _scale_factor), round(18 * _scale_factor)),
                              'verbose': False}

### Shadow Mask
court_masks.shadow_params = {'h_threshold': 5,
                             's_threshold': None,
                             'v_threshold_lower': 0.75,
                             'v_threshold_upper': 0.97,
                             'opening_kernel': (round(15 * _scale_factor), round(15 * _scale_factor)),
                             'closing_kernel': (round(15 * _scale_factor), round(15 * _scale_factor))}

# a) a fine grained mask, for ball detection
court_masks.motion_params = {'intensity_threshold': 5,
                             'relevant_indices': [4, 6, 8],
                             'open_kernel': None,
                             'close_kernel': (3, 3)}


####### PLAYER CANDIDATES #######

player_candidates = Parameters()
player_candidates.params = {'min_player_size': 50, 'frame_blur': 1}

###### PLAYER TRACKING #######

player_tracking = Parameters()

_frame_offset = 5
player_tracking.link_params = {'frame_offset': _frame_offset}
player_tracking.split_track_params = {'frame_offset': _frame_offset,
                                      'critical_number_of_tracks': 50000,
                                      'verbose': False,
                                      'super_verbose': False}
player_tracking.match_params = {'frame_offset': _frame_offset,
                                'num_points': 6,
                                'threshold_distance': 20}
player_tracking.eval_params = {'granularity': 10,
                               'area_outlier_iqr_multiple': 1.5,
                               'overlap_optimum': 0.9 ** _frame_offset,
                               'overlap_strictness': 5,
                               'area_outlier_iqr_multiple': 1.5,
                               'metric_weights': {'dist': 4, 'dist_time': 4, 'dist_serve': 0, 'ovlp': 2, 'area': 1, 'move': 2}}
player_tracking.itpl_params = {'frame_offset': _frame_offset,
                               'minimum_overlap_between_candidate_and_interpolation_position': 0.2}

##### BALL FILTERED MOTION MASK #####

ball_filtered_motion_mask = Parameters()
### Filter out Players from fine motion mask
ball_filtered_motion_mask.filter_params = { 'top_player_dil_kernel': (12, 7),
                                'bot_player_dil_kernel': (20, 12)}

##### BALL CONTOURS ######

ball_contours = Parameters()
ball_contours.contour_params = {'ROI': None, 
                                'min_area_of_contours': 0, 
                                'max_number_of_contours': 200}


##### BALL SVM #########

ball_inference_svm = Parameters()

ball_inference_svm.svm_params = {'svm_name': 'svm_proba_dict_r75_p60_2023-08-15'}    
ball_inference_svm.inference_params = { 'use_training_moments': True,
                                        'threshold_low': 0.025, 
                                        'threshold_mid': 0.1, 
                                        'threshold_hi': None}

###### HITS CLASSIFY VIDEO #####

hits_classify_video = Parameters()

hits_classify_video.model_params = {'top': 'top_foremotion_T16_S8_randcrop5_dim56_small_fullconv_model_foremotion_20231121_235426',
                                    'bot': 'bot_foremotion_T16_S8_randcrop5_dim88_small_fullconv_model_foremotion_20231122_095800'}

##### HITS_CLASSIFY_JOIN #######

hits_classify_join = Parameters()



###### BALL_INFERENCE_CHAMFER #########

ball_inference_chamfer = Parameters()
ball_inference_chamfer.params = {'threshold': 50 * _scale_factor}
    
#### BALL_INFERENCE_PARTICLES ######

ball_inference_particles = Parameters()

ball_inference_particles.filter_init_params = {'process_standard_deviation': 1.5,
                                                'observation_standard_deviation': 2,
                                                'spatial_lambda': 10 ** -4}

ball_inference_particles.filter_run_params = {'N': 10000,
                                            'fps': 25,
                                            'rectify': True}

ball_inference_particles.ball_choice_params = {'standard_deviation_threshold': 2}

ball_inference_particles.propagation_params = {'share_kernel': [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.75]}

######## SEGMENTATION_WITH_SOUND_AND_VISION #######

segmentation_with_sound_and_vision = Parameters()

segmentation_with_sound_and_vision.viterbi_params = {   'initial_state_probabilities': [0.9, 0.5], 
                                                        'hit_state_probabilities': [0, 0, 0, 0, 0],
                                                        'flight_state_probabilities': [0.5, 0.9, 0.5], 
                                                        'exit_state_probabilities': [1],
                                                        'initial_emission_exponents': [2, 0], 
                                                        'flight_emission_exponents': [0, 2, 0], 
                                                        'exit_emission_exponents': [1], 
                                                        'high_hit_emission_exponents': [4, 4, 4, 4, 4], 
                                                        'low_hit_emission_exponents': [1/2, 1/2, 1/2, 1/2, 1/2]
                                                    }
segmentation_with_sound_and_vision.weight_params = {'confidence_audio_vs_video': 0.7} ### we trust the audio a bit more than the video. 
    
segmentation_with_sound_and_vision.smooth_params = {'smoothing_kernel_audio': np.array([0.25, 0.5, 1, 0.5, 0.25]),
                                                    'smoothing_kernel_video': np.array([0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 1, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1])}
       
segmentation_with_sound_and_vision.widen_params = {'add_before': 5, 'add_after': 30}

###### BALL_INFERENCE_HIT_CANDIDATES #########

ball_inference_hit_candidates = Parameters()

ball_inference_hit_candidates.params = {'stroke_px_r_top': 10,
                                        'stroke_px_r_bot': 20,
                                        'stroke_px_u_top': -2,
                                        'stroke_px_u_bot': 10,
                                        'serve_px_u_top': 20,
                                        'serve_px_u_bot': 40,
                                        'do_bot': True,
                                        'do_top': True,
                                        'hand_top': 'right',
                                        'hand_bot': 'right',
                                        'resolution': 6}


##### BALL INTERPOLATION PLANE #######

ball_interpolation_plane = Parameters()
ball_interpolation_plane.trajectory_params = {'traj_name': 'ShorterTrajectories_g60_v41_s41_h46', 
                                              'observation_threshold': 3, 'sum_of_squares': True, 'number_of_local_minima': 30,
                                              'groundstrokes_are_below': 2.5, 'serves_and_overheads_are_above': 2}
ball_interpolation_plane.plane_point_params = {'max_sideways_reach_in_feet': 5, 'resolution': 1, 'serve_height_modifier': 1}
ball_interpolation_plane.hit_point_params = {'height_in_feet': 5, 'resolution': 0.5, 'radius_in_feet': 15}
ball_interpolation_plane.bounce_params = {'prior_rel_weight_dist': 1, 'prior_rel_weight_max': 0} ### TODO: check for higher prior_rel_weight_max!
ball_interpolation_plane.downsample_params = {'logit_coeff': 10, 'sampling_number': 25}
ball_interpolation_plane.viterbi_net_params = {'p': 0.95, 'epsilon': 0.01, 'q': 0.1, 
                                               'stretch_factor': 0.25, 'net_bottom_adjustment_in_pixels': 5}
ball_interpolation_plane.penalty_params = {'frame_cutoff': 5, 'threshold_angle': np.pi / 8, 'epsilon': 0.1,
                                           'below_weight': 1, 'monot_weight': 1, 'decel_weight': 1, 'net_weight': 1, 'angle_weight': 0}
ball_interpolation_plane.viterbi_traj_params = {'dst_factor': 0.15, 'dev_factor': 0.2, 
                                                'bnc_factor': 0.2, 'net_factor': 0.15, 
                                                'fair_factor': 0.15, 'spin_factor': 0.15}    
